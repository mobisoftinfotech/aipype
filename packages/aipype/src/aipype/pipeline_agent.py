"""PipelineAgent - automatic task orchestration based on dependencies."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Set, Optional

from typing import override
from .base_task import BaseTask
from .task_result import TaskResult
from .task_context import TaskContext
from .task_dependencies import DependencyResolver
from .agent_run_result import AgentRunResult
from .utils.common import setup_logger


class TaskExecutionPlan:
    """Execution plan that organizes tasks into phases based on dependencies."""

    def __init__(self, tasks: List[BaseTask]):
        """Create execution plan from list of tasks.

        Args:
            tasks: List of tasks to organize into execution phases

        Raises:
            ValueError: If circular dependencies detected or dependencies cannot be satisfied
        """
        self.tasks = tasks
        self.logger = setup_logger("task_execution_plan")
        self.phases: List[List[BaseTask]] = []
        self._build_execution_phases()

    def _build_execution_phases(self) -> None:
        """Build execution phases by analyzing task dependencies."""
        if not self.tasks:
            return

        # Create task name to task mapping
        task_map = {task.name: task for task in self.tasks}

        # Build dependency graph
        dependency_graph = self._build_dependency_graph(task_map)

        # Check for circular dependencies
        self._check_circular_dependencies(dependency_graph)

        # Organize into phases
        self._organize_into_phases(dependency_graph, task_map)

    def _build_dependency_graph(
        self, task_map: Dict[str, BaseTask]
    ) -> Dict[str, Set[str]]:
        """Build dependency graph mapping task names to their dependencies.

        Args:
            task_map: Mapping of task names to task objects

        Returns:
            Dictionary mapping task names to sets of dependency task names
        """
        dependency_graph: Dict[str, Set[str]] = {}

        for task_name, task in task_map.items():
            dependencies: Set[str] = set()

            for dep in task.get_dependencies():
                # Extract task name from source path
                if dep.source_path and "." in dep.source_path:
                    dep_task_name: str = dep.source_path.split(".")[0]
                    if dep_task_name in task_map:
                        dependencies.add(dep_task_name)
                    elif dep.is_required():
                        raise ValueError(
                            f"PipelineAgent dependency resolution failed: Cannot satisfy dependency: task '{dep_task_name}' not found for task '{task_name}'"
                        )

            dependency_graph[task_name] = dependencies

        return dependency_graph

    def _check_circular_dependencies(
        self, dependency_graph: Dict[str, Set[str]]
    ) -> None:
        """Check for circular dependencies in the graph.

        Args:
            dependency_graph: Task dependency graph

        Raises:
            ValueError: If circular dependency detected
        """
        # Use DFS to detect cycles
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def has_cycle(node: str) -> bool:
            if node in rec_stack:
                return True
            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)

            for neighbor in dependency_graph.get(node, set()):
                if has_cycle(neighbor):
                    return True

            rec_stack.remove(node)
            return False

        for task_name in dependency_graph:
            if task_name not in visited:
                if has_cycle(task_name):
                    raise ValueError(
                        f"PipelineAgent dependency validation failed: Circular dependency detected involving task '{task_name}'"
                    )

    def _organize_into_phases(
        self, dependency_graph: Dict[str, Set[str]], task_map: Dict[str, BaseTask]
    ) -> None:
        """Organize tasks into execution phases for optimal parallelism.

        Args:
            dependency_graph: Task dependency graph
            task_map: Mapping of task names to task objects
        """
        remaining_tasks = set(dependency_graph.keys())

        while remaining_tasks:
            # Find tasks with no remaining dependencies
            ready_tasks: List[BaseTask] = []
            for task_name in remaining_tasks:
                deps = dependency_graph[task_name]
                if not deps or all(dep not in remaining_tasks for dep in deps):
                    ready_tasks.append(task_map[task_name])

            if not ready_tasks:
                # Should not happen if circular dependency check passed
                raise ValueError(
                    "PipelineAgent task organization failed: Unable to resolve task dependencies - possible circular dependency"
                )

            # Add ready tasks as a new phase
            self.phases.append(ready_tasks)

            # Remove completed tasks from remaining set
            completed_task_names: Set[str] = {task.name for task in ready_tasks}
            remaining_tasks -= completed_task_names

    def total_phases(self) -> int:
        """Get total number of execution phases.

        Returns:
            Number of phases
        """
        return len(self.phases)

    def get_phase(self, phase_index: int) -> List[BaseTask]:
        """Get tasks in a specific phase.

        Args:
            phase_index: Index of the phase

        Returns:
            List of tasks in the phase
        """
        if 0 <= phase_index < len(self.phases):
            return self.phases[phase_index]
        return []

    def get_total_tasks(self) -> int:
        """Get total number of tasks in the plan.

        Returns:
            Total number of tasks
        """
        return sum(len(phase) for phase in self.phases)


class PipelineAgent:
    """Agent that automatically orchestrates task execution based on dependencies."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize pipeline agent.

        Args:
            name: Agent name
            config: Agent configuration
        """
        self.name = name
        self.config = config or {}
        self.tasks: List[BaseTask] = []
        self.logger = setup_logger(f"pipeline_agent_{name}")
        self._is_running = False

        self.context = TaskContext()
        self.dependency_resolver = DependencyResolver(self.context)
        self.execution_plan: Optional[TaskExecutionPlan] = None
        self.enable_parallel = config.get("enable_parallel", True) if config else True

        # Automatically setup tasks if not already done
        if not self.tasks:
            self._auto_setup_tasks()

    def add_tasks(self, tasks: List[BaseTask]) -> None:
        """Add tasks to the agent.

        Args:
            tasks: List of tasks to add
        """
        for task in tasks:
            task.set_agent_name(self.name)
            self.tasks.append(task)
            self.logger.debug(f"Added task: {task.name}")

    def setup_tasks(self) -> List[BaseTask]:
        """Set up tasks for this agent. Must be implemented by subclasses.

        Returns:
            List of configured tasks
        """
        raise NotImplementedError("Subclasses must implement setup_tasks()")

    def create_context(self) -> TaskContext:
        """Create and return the task context for this agent.

        Returns:
            TaskContext instance
        """
        return self.context

    def _log_task_start(
        self, task_name: str, task_index: int, total_tasks: int
    ) -> None:
        """Log task start."""
        self.logger.info(
            f"[STATUS] Starting task {task_index}/{total_tasks}: {task_name}"
        )

    def _log_task_end(
        self, task_name: str, task_index: int, total_tasks: int, success: bool
    ) -> None:
        """Log task completion."""
        status = "[SUCCESS] SUCCESS" if success else "[ERROR] FAILED"
        self.logger.info(
            f"[STATUS] Completed task {task_index}/{total_tasks}: {task_name} - {status}"
        )

    def _log_task_output(self, task_name: str, result: Any) -> None:
        """Log task output with distinctive formatting."""
        if result is not None:
            result_str = str(result)[:5000]
            self.logger.info("*" + "-" * 60 + "*")
            self.logger.info(f"[STATUS] TASK OUTPUT: '{task_name}'")
            self.logger.info("*" + "-" * 60 + "*")
            self.logger.info(result_str)
            self.logger.info("*" + "-" * 60 + "*")
        else:
            self.logger.info(f"[STATUS] Task '{task_name}' completed with no output")

    def _auto_setup_tasks(self) -> None:
        """Automatically setup tasks by calling setup_tasks() and adding them."""
        try:
            tasks = self.setup_tasks()
            if tasks:
                self.add_tasks(tasks)
                self.logger.info(f"Auto-setup completed: added {len(tasks)} tasks")
        except Exception as e:
            self.logger.warning(f"Auto-setup failed: {e}")

    def run(self) -> AgentRunResult:
        """Run all tasks using automatic dependency-based orchestration.

        Returns:
            AgentRunResult containing execution status and results
        """
        if self._is_running:
            self.logger.warning(f"PipelineAgent '{self.name}' is already running")
            return AgentRunResult.running(self.name)

        self._is_running = True
        self.logger.info(
            f"Starting PipelineAgent '{self.name}' with {len(self.tasks)} tasks"
        )

        try:
            # Build execution plan
            self.execution_plan = self._build_execution_plan()

            # Execute phases
            results = self._execute_phases()

            # Count successful and failed tasks
            completed_tasks = len([r for r in results if r.get("status") == "success"])
            failed_tasks = len(self.tasks) - completed_tasks
            total_phases = self.execution_plan.total_phases()

            # Determine status based on task completion
            if completed_tasks == len(self.tasks):
                # All tasks completed successfully
                return AgentRunResult.success(
                    agent_name=self.name,
                    total_tasks=len(self.tasks),
                    completed_tasks=completed_tasks,
                    total_phases=total_phases,
                )
            elif completed_tasks > 0:
                # Some tasks succeeded, some failed
                return AgentRunResult.partial(
                    agent_name=self.name,
                    total_tasks=len(self.tasks),
                    completed_tasks=completed_tasks,
                    failed_tasks=failed_tasks,
                    total_phases=total_phases,
                )
            else:
                # No tasks succeeded (all failed)
                return AgentRunResult.failure(
                    agent_name=self.name,
                    total_tasks=len(self.tasks),
                    failed_tasks=failed_tasks,
                    error_message="All tasks failed to complete",
                    total_phases=total_phases,
                )

        finally:
            self._is_running = False

    def _build_execution_plan(self) -> TaskExecutionPlan:
        """Build execution plan for current tasks.

        Returns:
            TaskExecutionPlan instance
        """
        if not self.tasks:
            return TaskExecutionPlan([])

        self.logger.info(f"Building execution plan for {len(self.tasks)} tasks")
        plan = TaskExecutionPlan(self.tasks)
        self.logger.info(f"Execution plan created with {plan.total_phases()} phases")

        return plan

    def _execute_phases(self) -> List[Dict[str, Any]]:
        """Execute all phases of the execution plan.

        Returns:
            List of task execution results
        """
        results: List[Dict[str, Any]] = []
        total_tasks = len(self.tasks)
        completed_tasks = 0

        self.logger.info("")
        self.logger.info("")
        self.logger.info("[TARGET]" + "=" * 78 + "[TARGET]")
        self.logger.info(f"[BOT] STARTING PIPELINE AGENT: {self.name}")
        self.logger.info(f"[SUMMARY] TOTAL TASKS: {total_tasks}")
        self.logger.info("[TARGET]" + "=" * 78 + "[TARGET]")
        self.logger.info("")

        if self.execution_plan is None:
            self.logger.error("No execution plan available")
            return []

        for phase_index in range(self.execution_plan.total_phases()):
            phase_tasks = self.execution_plan.get_phase(phase_index)
            self.logger.info(
                f"Executing phase {phase_index + 1}/{self.execution_plan.total_phases()} with {len(phase_tasks)} tasks"
            )

            if self.enable_parallel and len(phase_tasks) > 1:
                phase_results = self._execute_phase_parallel(
                    phase_tasks, completed_tasks, total_tasks
                )
            else:
                phase_results = self._execute_phase_sequential(
                    phase_tasks, completed_tasks, total_tasks
                )

            results.extend(phase_results)
            completed_tasks += len(
                [r for r in phase_results if r.get("status") == "success"]
            )

            # Check if we should stop on failure
            if self.config.get("stop_on_failure", True):
                failed_tasks = [r for r in phase_results if r.get("status") == "error"]
                if failed_tasks:
                    self.logger.info(
                        f"[STOP] Stopping execution due to {len(failed_tasks)} failed tasks in phase {phase_index + 1}"
                    )
                    break

        self.logger.info("")
        self.logger.info("")
        self.logger.info("[FINISH]" + "=" * 78 + "[FINISH]")
        self.logger.info(f"[BOT] PIPELINE AGENT COMPLETED: {self.name}")
        self.logger.info(f"[SUMMARY] TASKS COMPLETED: {completed_tasks}/{total_tasks}")
        self.logger.info("[FINISH]" + "=" * 78 + "[FINISH]")
        self.logger.info("")
        self.logger.info("")

        return results

    def _execute_phase_sequential(
        self, tasks: List[BaseTask], completed_tasks: int, total_tasks: int
    ) -> List[Dict[str, Any]]:
        """Execute tasks in a phase sequentially.

        Args:
            tasks: Tasks to execute
            completed_tasks: Number of tasks completed so far
            total_tasks: Total number of tasks

        Returns:
            List of execution results
        """
        results: List[Dict[str, Any]] = []

        for task in tasks:
            result = self._execute_single_task(
                task, completed_tasks + len(results) + 1, total_tasks
            )
            results.append(result)

            # Stop if task failed and stop_on_failure is enabled
            if result.get("status") == "error" and self.config.get(
                "stop_on_failure", True
            ):
                break

        return results

    def _execute_phase_parallel(
        self, tasks: List[BaseTask], completed_tasks: int, total_tasks: int
    ) -> List[Dict[str, Any]]:
        """Execute tasks in a phase in parallel.

        Args:
            tasks: Tasks to execute
            completed_tasks: Number of tasks completed so far
            total_tasks: Total number of tasks

        Returns:
            List of execution results
        """
        results: List[Dict[str, Any]] = []
        max_workers = min(len(tasks), self.config.get("max_parallel_tasks", 5))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(
                    self._execute_single_task,
                    task,
                    completed_tasks + i + 1,
                    total_tasks,
                ): task
                for i, task in enumerate(tasks)
            }

            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    error_result = {
                        "task_name": task.name,
                        "status": "error",
                        "error": f"Parallel execution error: {str(e)}",
                    }
                    results.append(error_result)
                    self.logger.error(
                        f"[ERROR] Parallel execution error for task '{task.name}': {str(e)}"
                    )

        # Sort results by original task order
        task_order = {task.name: i for i, task in enumerate(tasks)}
        results.sort(key=lambda r: task_order.get(r["task_name"], 999))

        return results

    def _execute_single_task(
        self, task: BaseTask, task_index: int, total_tasks: int
    ) -> Dict[str, Any]:
        """Execute a single task with dependency resolution.

        Args:
            task: Task to execute
            task_index: Current task index (1-based)
            total_tasks: Total number of tasks

        Returns:
            Task execution result
        """
        self._log_task_start(task.name, task_index, total_tasks)

        try:
            # Record task start in context
            self.context.record_task_started(task.name)

            # Set context for tasks that need it
            task.set_context(self.context)

            # Resolve dependencies and update task config
            resolved_config: Dict[str, Any] = (
                self.dependency_resolver.resolve_dependencies(task)
            )
            task.config.update(resolved_config)

            # Mark task as started
            task.mark_started()

            # Execute task
            task_result: TaskResult = task.run()

            # Handle result based on TaskResult status
            if task_result.is_success():
                # Mark task as successful with the data
                task.mark_success(task_result.data)

                # Store result data in context (maintain backward compatibility)
                self.context.store_result(task.name, task_result.data)
                self.context.record_task_completed(task.name, task_result.data)

                self._log_task_output(task.name, task_result.data)
                self._log_task_end(task.name, task_index, total_tasks, success=True)

                return {
                    "task_name": task.name,
                    "status": "success",
                    "result": task_result.data,
                    "task_result": task_result,  # Include full TaskResult for advanced usage
                    "execution_time": task_result.execution_time,
                    "metadata": task_result.metadata,
                }

            elif task_result.is_partial():
                # Handle partial success - store data but log warning
                task.mark_success(
                    task_result.data
                )  # Mark as success for pipeline continuity

                self.context.store_result(task.name, task_result.data)
                self.context.record_task_completed(task.name, task_result.data)

                self.logger.warning(
                    f"[WARNING]  Task '{task.name}' completed with partial success: {task_result.error}"
                )
                self._log_task_output(task.name, task_result.data)
                self._log_task_end(task.name, task_index, total_tasks, success=True)

                return {
                    "task_name": task.name,
                    "status": "partial",
                    "result": task_result.data,
                    "task_result": task_result,
                    "execution_time": task_result.execution_time,
                    "metadata": task_result.metadata,
                    "warning": task_result.error,
                }

            elif task_result.is_skipped():
                # Handle skipped task
                self.logger.info(
                    f"[SKIP]  Task '{task.name}' was skipped: {task_result.error}"
                )

                # Don't store result but record the skip
                self.context.record_task_failed(
                    task.name, task_result.error or "Task was skipped"
                )

                return {
                    "task_name": task.name,
                    "status": "skipped",
                    "result": None,
                    "task_result": task_result,
                    "execution_time": task_result.execution_time,
                    "metadata": task_result.metadata,
                    "skip_reason": task_result.error,
                }

            else:  # task_result.is_error()
                # Handle error
                error_msg = task_result.error or "Unknown error"

                # Mark task as failed
                task.mark_error(error_msg)

                # Record failure in context
                self.context.record_task_failed(task.name, error_msg)

                self.logger.error(
                    f"[ERROR] ERROR: Task '{task.name}' failed: {error_msg}"
                )
                self._log_task_end(task.name, task_index, total_tasks, success=False)

                return {
                    "task_name": task.name,
                    "status": "error",
                    "error": error_msg,
                    "task_result": task_result,
                    "execution_time": task_result.execution_time,
                    "metadata": task_result.metadata,
                }

        except Exception as e:
            # Handle unexpected exceptions during task execution setup
            error_msg = f"PipelineAgent task execution failed: Task '{task.name}' failed during execution setup: {str(e)}"

            # Mark task as failed
            task.mark_error(error_msg)

            # Record failure in context
            self.context.record_task_failed(task.name, error_msg)

            self.logger.error(f"[ERROR] ERROR: {error_msg}")
            self._log_task_end(task.name, task_index, total_tasks, success=False)

            return {"task_name": task.name, "status": "error", "error": str(e)}

    def get_context(self) -> TaskContext:
        """Get the task context for this agent.

        Returns:
            TaskContext instance
        """
        return self.context

    def get_execution_plan(self) -> Optional[TaskExecutionPlan]:
        """Get the current execution plan.

        Returns:
            TaskExecutionPlan instance or None if not yet created
        """
        return self.execution_plan

    def validate_dependencies(self) -> List[str]:
        """Validate that all task dependencies can be satisfied.

        Returns:
            List of validation error messages (empty if all valid)
        """
        errors: List[str] = []

        for task in self.tasks:
            task_errors: List[str] = self.dependency_resolver.validate_dependencies(
                task
            )
            errors.extend(task_errors)

        return errors

    def get_dependency_info(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get dependency information for all tasks.

        Returns:
            Dictionary mapping task names to their dependency info
        """
        info: Dict[str, List[Dict[str, Any]]] = {}

        for task in self.tasks:
            info[task.name] = self.dependency_resolver.get_dependency_info(task)

        return info

    def reset(self) -> None:
        """Reset the agent and clear context."""
        self.context.clear()
        self.execution_plan = None
        self.logger.info(f"PipelineAgent '{self.name}' reset")

    def display_results(self, sections: Optional[List[str]] = None) -> None:
        """Display results with configurable sections.

        This method provides a simple way to display agent results without writing
        complex display logic in each agent. Users can choose which sections to show.

        Args:
            sections: List of sections to display. Options: "summary", "tasks", "errors".
                     Defaults to ["summary", "tasks", "errors"] (show all sections).
        """
        if sections is None:
            sections = ["summary", "tasks", "errors"]

        if not self.context:
            print("No execution results available")
            return

        print(f"\n{'=' * 60}")
        print(f"[AGENT] {self.name.upper().replace('_', ' ')} RESULTS")
        print(f"{'=' * 60}")

        if "summary" in sections:
            self._display_summary_section()

        if "tasks" in sections:
            self._display_tasks_section()

        if "errors" in sections:
            self._display_errors_section()

        print(f"{'=' * 60}\n")

    def _display_summary_section(self) -> None:
        """Display execution summary section."""
        completed_tasks = self.context.get_completed_tasks()
        failed_tasks = self.context.get_failed_tasks()

        print("\n[SUMMARY] Execution Summary:")
        print(f"  Total Tasks: {len(self.tasks)}")
        print(f"  Completed: {len(completed_tasks)}")
        print(f"  Failed: {len(failed_tasks)}")

        if self.execution_plan:
            print(f"  Execution Phases: {self.execution_plan.total_phases()}")

    def _display_tasks_section(self) -> None:
        """Display completed tasks section with basic result information."""
        completed_tasks = self.context.get_completed_tasks()

        if not completed_tasks:
            return

        print("\n[TASKS] Task Results:")

        for task_name in completed_tasks:
            result = self.context.get_result(task_name)
            if result:
                print(f"\n  --- {task_name.replace('_', ' ').title()} ---")
                print("  Status: ✅ SUCCESS")

                # Handle nested result structures (e.g., from TransformTask)
                data = result
                # Result is always dict[str, Any] from context
                # Check for TransformTask pattern with output_name
                if "output_name" in result:
                    output_name = result.get("output_name")
                    if output_name and output_name in result:
                        data = result[output_name]
                # Check for direct nested data in known keys
                elif "result" in result and isinstance(result["result"], dict):
                    # Task result data structures are dynamic based on task implementations
                    data: dict[str, Any] = result["result"]  # pyright: ignore[reportUnknownVariableType]

                # Show content preview if available
                if "content" in data:
                    # Dynamic task result content can be any type, convert to string for display
                    content = data.get("content", "")
                    if isinstance(content, str) and content.strip():
                        preview = (
                            content[:150] + "..." if len(content) > 150 else content
                        )
                        print(f"  Content: {preview}")

                # Show token usage if available (for LLM tasks)
                if "usage" in data:
                    # LLM response usage data structure varies by provider
                    usage = data["usage"]
                    total_tokens = usage.get("total_tokens", 0)
                    if total_tokens > 0:
                        # Token counts are numeric values from LLM providers
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        print(
                            f"  Tokens: {total_tokens} ({prompt_tokens} prompt + {completion_tokens} completion)"
                        )

                    # Show basic result info for other task types
                    # Show key result metrics based on common patterns
                    if "total_urls" in data:  # URL fetch tasks
                        # URL fetch task metrics are dynamic based on task implementation
                        print(f"  URLs Processed: {data.get('total_urls', 0)}")
                        print(f"  Successful: {data.get('successful_fetches', 0)}")
                    elif "query" in data:  # Search tasks
                        # Search task data structure varies by search provider
                        print(f"  Query: {data.get('query', 'Unknown')}")
                        search_results = data.get("results", [])
                        print(f"  Results Found: {len(search_results)}")
                    elif "file_path" in data:  # File save tasks
                        # File save task results have dynamic file path structure
                        print(f"  Saved to: {data.get('file_path', 'Unknown')}")
                    elif "condition_result" in data:  # Conditional tasks
                        # Conditional task results have boolean condition status
                        condition_met = data.get("condition_result", False)
                        print(
                            f"  Condition Met: {'✅ Yes' if condition_met else '❌ No'}"
                        )

                # For direct result dictionaries (LLM tasks, etc.)
                else:
                    if "total_urls" in result:  # URL fetch tasks
                        print(f"  URLs Processed: {result.get('total_urls', 0)}")
                        print(f"  Successful: {result.get('successful_fetches', 0)}")
                    elif "query" in result:  # Search tasks
                        print(f"  Query: {result.get('query', 'Unknown')}")
                        print(f"  Results Found: {len(result.get('results', []))}")
                    elif "file_path" in result:  # File save tasks
                        print(f"  Saved to: {result.get('file_path', 'Unknown')}")
                    elif "condition_result" in result:  # Conditional tasks
                        condition_met = result.get("condition_result", False)
                        print(
                            f"  Condition Met: {'✅ Yes' if condition_met else '❌ No'}"
                        )

    def _display_errors_section(self) -> None:
        """Display failed tasks section with error details."""
        failed_tasks = self.context.get_failed_tasks()

        if not failed_tasks:
            return

        print("\n[ERRORS] Failed Tasks:")

        for task_name in failed_tasks:
            history = self.context.get_execution_history()
            task_history = next(
                (h for h in history if h["task_name"] == task_name), None
            )
            if task_history and task_history.get("error"):
                print(f"  ❌ {task_name}: {task_history['error']}")

    @override
    def __str__(self) -> str:
        """String representation of the agent."""
        completed = len(self.context.get_completed_tasks())
        failed = len(self.context.get_failed_tasks())
        phases = self.execution_plan.total_phases() if self.execution_plan else 0

        return (
            f"PipelineAgent(name='{self.name}', tasks={len(self.tasks)}, "
            f"completed={completed}, failed={failed}, phases={phases})"
        )
