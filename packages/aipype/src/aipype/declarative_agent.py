"""Declarative pipeline agent with automatic task discovery.

This module provides PipelineAgent, a subclass of BasePipelineAgent that
automatically discovers @task decorated methods and builds the task list.

Example::

    from aipype import PipelineAgent, task, Depends
    from typing import Annotated

    class MyAgent(PipelineAgent):
        @task
        def fetch_data(self) -> dict:
            return {"data": "value"}

        @task
        def process(self, fetch_data: dict) -> str:
            # fetch_data parameter auto-wired from fetch_data task
            return f"Processed: {fetch_data}"

        @task
        def with_explicit_path(
            self,
            content: Annotated[str, Depends("process.output")]
        ) -> str:
            return f"Final: {content}"

    agent = MyAgent("my_agent", {"config_key": "value"})
    result = agent.run()
"""

from typing import Any, Callable, Dict, List, Optional, Set, override

from .base_task import BaseTask
from .dependency_inference import infer_dependencies_from_signature
from .pipeline_agent import BasePipelineAgent
from .task_dependencies import TaskDependency
from .task_wrapper import TaskWrapper
from .utils.common import setup_logger


class PipelineAgent(BasePipelineAgent):
    """Declarative agent that discovers tasks from @task decorated methods.

    PipelineAgent provides a cleaner, more Pythonic way to define
    AI pipelines. Instead of manually creating task objects and wiring
    dependencies, you simply define methods decorated with @task and let
    the framework infer the execution order from parameter names.

    Key Features:
        - Automatic task discovery from decorated methods
        - Dependency inference from function signatures
        - Support for explicit paths via Annotated[T, Depends("path")]
        - Full compatibility with existing execution engine
        - Parallel execution of independent tasks

    How It Works:
        1. On initialization, discovers all @task decorated methods
        2. Analyzes method signatures to infer dependencies
        3. Topologically sorts methods by dependency order
        4. Creates TaskWrapper instances for each method
        5. Returns task list to parent BasePipelineAgent for execution

    The parent BasePipelineAgent handles all execution concerns:
        - Building execution phases
        - Resolving dependencies
        - Parallel execution
        - Error handling
        - Result collection

    Example::

        class ArticleAgent(PipelineAgent):
            @task
            def search(self) -> dict:
                # Access config directly for initial inputs
                return {"results": f"Results for {self.config['topic']}"}

            @task
            def summarize(self, search: dict) -> str:
                # 'search' parameter auto-wired from search task
                return f"Summary of: {search}"

            @task
            def write(self, summarize: str) -> str:
                return f"Article based on: {summarize}"

        agent = ArticleAgent("writer", {"topic": "AI trends"})
        result = agent.run()

    Attributes:
        name: Agent identifier
        config: Configuration dictionary
        tasks: List of TaskWrapper instances (populated by setup_tasks())
        context: TaskContext for inter-task communication
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the pipeline agent.

        Args:
            name: Unique identifier for this agent
            config: Configuration dictionary passed to agent and tasks
        """
        self._declarative_logger = setup_logger(f"declarative_agent_{name}")
        super().__init__(name, config)

    @override
    def setup_tasks(self) -> List[BaseTask]:
        """Auto-discover @task methods and build the task list.

        This method:
        1. Finds all methods decorated with @task
        2. Infers dependencies from method signatures
        3. Sorts methods by dependency order
        4. Creates TaskWrapper instances

        Returns:
            List of BaseTask instances ready for execution
        """
        # Discover all task methods
        task_methods = self._discover_task_methods()

        if not task_methods:
            self._declarative_logger.warning(
                f"No @task methods found in {self.__class__.__name__}"
            )
            return []

        self._declarative_logger.info(f"Discovered {len(task_methods)} task methods")

        # Get all task names for dependency inference
        known_task_names: Set[str] = set()
        for method in task_methods:
            task_name: str = getattr(method, "_task_name", "")
            if task_name:
                known_task_names.add(task_name)

        # Build dependency graph and infer dependencies
        method_deps: Dict[str, List[TaskDependency]] = {}
        for method in task_methods:
            task_name = getattr(method, "_task_name", "")
            deps = infer_dependencies_from_signature(method, known_task_names)
            method_deps[task_name] = deps

            if deps:
                dep_names = [d.name for d in deps]
                self._declarative_logger.debug(
                    f"Task '{task_name}' depends on: {dep_names}"
                )

        # Topologically sort methods
        sorted_methods = self._topological_sort(task_methods, method_deps)

        # Create TaskWrapper for each method
        tasks: List[BaseTask] = []
        for method in sorted_methods:
            task_name = getattr(method, "_task_name", "")
            dependencies = method_deps.get(task_name, [])

            wrapper = TaskWrapper(
                name=task_name,
                method=method,
                agent=self,
                dependencies=dependencies,
            )
            tasks.append(wrapper)

            self._declarative_logger.debug(f"Created TaskWrapper for '{task_name}'")

        self._declarative_logger.info(f"Setup complete: {len(tasks)} tasks ready")

        return tasks

    def _discover_task_methods(self) -> List[Callable[..., Any]]:
        """Find all methods decorated with @task.

        Returns:
            List of bound methods marked as tasks
        """
        methods: List[Callable[..., Any]] = []

        for name in dir(self):
            # Skip private/magic methods
            if name.startswith("_"):
                continue

            try:
                attr = getattr(self, name)
            except AttributeError:
                continue

            # Check if it's a callable marked as a task
            if callable(attr) and getattr(attr, "_is_pipeline_task", False):
                methods.append(attr)

        return methods

    def _topological_sort(
        self,
        methods: List[Callable[..., Any]],
        method_deps: Dict[str, List[TaskDependency]],
    ) -> List[Callable[..., Any]]:
        """Sort methods by dependency order using topological sort.

        Uses Kahn's algorithm to order methods so that dependencies
        come before dependents.

        Args:
            methods: List of task methods
            method_deps: Mapping of task names to their dependencies

        Returns:
            Methods sorted so dependencies come before dependents

        Raises:
            ValueError: If circular dependencies detected
        """
        # Build name -> method mapping
        name_to_method: Dict[str, Callable[..., Any]] = {}
        for method in methods:
            task_name: str = getattr(method, "_task_name", "")
            if task_name:
                name_to_method[task_name] = method

        # Build adjacency list (task -> tasks that depend on it)
        dependents: Dict[str, Set[str]] = {name: set() for name in name_to_method}
        in_degree: Dict[str, int] = {name: 0 for name in name_to_method}

        for task_name, deps in method_deps.items():
            for dep in deps:
                # Extract dependency task name from source_path
                dep_task_name = dep.source_path.split(".")[0]

                if dep_task_name in dependents:
                    dependents[dep_task_name].add(task_name)
                    in_degree[task_name] += 1

        # Kahn's algorithm for topological sort
        queue = [name for name, degree in in_degree.items() if degree == 0]
        sorted_names: List[str] = []

        while queue:
            # Process task with no remaining dependencies
            current = queue.pop(0)
            sorted_names.append(current)

            # Reduce in-degree for dependent tasks
            for dependent in dependents[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # Check for cycles
        if len(sorted_names) != len(methods):
            remaining = set(in_degree.keys()) - set(sorted_names)
            raise ValueError(
                f"Circular dependency detected involving tasks: {remaining}"
            )

        # Return methods in sorted order
        return [name_to_method[name] for name in sorted_names]

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
