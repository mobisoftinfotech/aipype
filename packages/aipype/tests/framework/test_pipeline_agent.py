"""Tests for BasePipelineAgent - automatic task orchestration based on dependencies."""

import pytest
import time
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

from typing import override
from aipype import (
    BasePipelineAgent,
    TaskExecutionPlan,
    BaseTask,
    TaskDependency,
    DependencyType,
    TaskResult,
    AgentRunStatus,
)


class MockTask(BaseTask):
    """Mock task for testing pipeline agent functionality."""

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        dependencies: Optional[List[TaskDependency]] = None,
        run_result: Any = None,
        should_fail: bool = False,
        execution_delay: float = 0,
    ):
        super().__init__(name, config, dependencies)
        self.run_result = run_result or {"task": name, "status": "completed"}
        self.should_fail = should_fail
        self.execution_delay = execution_delay
        self.run_called = False
        self.run_count = 0
        self.execution_time: Optional[float] = None

    @override
    def run(self) -> TaskResult:
        self.run_called = True
        self.run_count += 1
        self.execution_time = time.time()

        if self.execution_delay > 0:
            time.sleep(self.execution_delay)

        if self.should_fail:
            return TaskResult.failure(f"Mock task {self.name} intentionally failed")

        return TaskResult.success(self.run_result)


def create_test_pipeline_agent(
    name: str, tasks: List[BaseTask], config: Optional[Dict[str, Any]] = None
) -> BasePipelineAgent:
    """Helper function to create a concrete BasePipelineAgent for testing."""

    class TestPipelineAgent(BasePipelineAgent):
        @override
        def setup_tasks(self) -> List[BaseTask]:
            return tasks

    return TestPipelineAgent(name, config or {})


class TestBasePipelineAgent:
    """Test suite for BasePipelineAgent functionality."""

    def test_agent_executes_tasks_automatically(self) -> None:
        """BasePipelineAgent runs tasks without custom run() method."""
        # Create simple tasks without dependencies
        task1 = MockTask("task1", {"value": 1})
        task2 = MockTask("task2", {"value": 2})
        task3 = MockTask("task3", {"value": 3})

        # Create pipeline agent
        agent = create_test_pipeline_agent(
            "test_agent", [task1, task2, task3], {"stop_on_failure": True}
        )

        # Execute pipeline
        result = agent.run()

        # Verify all tasks were executed
        assert task1.run_called
        assert task2.run_called
        assert task3.run_called

        # Verify result structure
        assert result.status == AgentRunStatus.SUCCESS
        assert result.agent_name == "test_agent"
        assert result.total_tasks == 3
        assert result.completed_tasks == 3

    def test_agent_resolves_dependency_order(self) -> None:
        """Tasks execute in correct order based on dependencies."""
        # Create tasks with dependencies that require specific order
        # Task order: search -> fetch -> outline -> article -> save

        search_task = MockTask("search", {"query": "test"})

        fetch_task = MockTask(
            "fetch",
            {},
            [TaskDependency("urls", "search.results[].url", DependencyType.REQUIRED)],
        )

        outline_task = MockTask(
            "outline",
            {},
            [TaskDependency("content", "fetch.articles", DependencyType.REQUIRED)],
        )

        article_task = MockTask(
            "article",
            {},
            [TaskDependency("outline", "outline.content", DependencyType.REQUIRED)],
        )

        save_task = MockTask(
            "save",
            {},
            [
                TaskDependency(
                    "article_content", "article.content", DependencyType.REQUIRED
                )
            ],
        )

        # Add tasks in random order to agent
        agent = create_test_pipeline_agent(
            "test_agent",
            [save_task, outline_task, search_task, article_task, fetch_task],
        )

        # Setup expected results for dependency resolution
        with patch(
            "aipype.task_dependencies.DependencyResolver.resolve_dependencies"
        ) as mock_resolve:
            # Mock successful dependency resolution for all tasks
            mock_resolve.return_value = {}

            # Execute pipeline
            result = agent.run()

        # Verify execution order by checking execution times
        assert (
            search_task.execution_time is not None
            and fetch_task.execution_time is not None
        )
        assert (
            outline_task.execution_time is not None
            and article_task.execution_time is not None
        )
        assert save_task.execution_time is not None
        assert search_task.execution_time < fetch_task.execution_time
        assert fetch_task.execution_time < outline_task.execution_time
        assert outline_task.execution_time < article_task.execution_time
        assert article_task.execution_time < save_task.execution_time

        # Verify all tasks completed
        assert result.completed_tasks == 5

    def test_agent_handles_task_failures_stop_on_failure(self) -> None:
        """Agent stops when tasks fail and stop_on_failure=True."""
        task1 = MockTask("task1", {"value": 1})
        task2 = MockTask(
            "task2",
            {"value": 2},
            [TaskDependency("data1", "task1.task", DependencyType.REQUIRED)],
            should_fail=True,
        )  # This task will fail
        task3 = MockTask(
            "task3",
            {"value": 3},
            [TaskDependency("data2", "task2.task", DependencyType.REQUIRED)],
        )

        # Create agent with stop_on_failure=True
        agent = create_test_pipeline_agent(
            "test_agent", [task1, task2, task3], {"stop_on_failure": True}
        )

        # Mock dependency resolution to handle dependencies
        with patch(
            "aipype.task_dependencies.DependencyResolver"
        ) as mock_resolver_class:
            mock_resolver = Mock()
            mock_resolver_class.return_value = mock_resolver

            def resolve_side_effect(task: BaseTask) -> Dict[str, Any]:
                if task.name == "task1":
                    return {"value": 1}
                elif task.name == "task2":
                    return {"value": 2, "data1": "task1"}
                elif task.name == "task3":
                    return {"value": 3, "data2": "task2"}
                return {}

            mock_resolver.resolve_dependencies.side_effect = resolve_side_effect

            # Execute pipeline
            result = agent.run()

        # Verify execution stopped after failure
        assert task1.run_called  # First task should run
        assert task2.run_called  # Failing task should run
        assert not task3.run_called  # Third task should not run due to failure

        # Verify result reflects partial completion
        assert (
            result.status == AgentRunStatus.PARTIAL
        )  # Agent completes with partial success
        assert result.completed_tasks == 1  # Only first task succeeded
        assert result.total_tasks == 3

    def test_agent_handles_task_failures_continue_on_failure(self) -> None:
        """Agent continues when tasks fail and stop_on_failure=False."""
        task1 = MockTask("task1", {"value": 1})
        task2 = MockTask("task2", {"value": 2}, should_fail=True)  # This task will fail
        task3 = MockTask("task3", {"value": 3})

        # Create agent with stop_on_failure=False
        agent = create_test_pipeline_agent(
            "test_agent", [task1, task2, task3], {"stop_on_failure": False}
        )

        # Execute pipeline
        result = agent.run()

        # Verify all tasks were attempted
        assert task1.run_called
        assert task2.run_called
        assert task3.run_called

        # Verify result reflects mixed success/failure
        assert result.completed_tasks == 2  # Tasks 1 and 3 succeeded

    def test_agent_supports_parallel_execution(self) -> None:
        """Independent tasks can run in parallel."""
        # Create tasks with intentional delays
        task1 = MockTask("independent1", {"value": 1}, execution_delay=0.1)
        task2 = MockTask("independent2", {"value": 2}, execution_delay=0.1)
        task3 = MockTask("independent3", {"value": 3}, execution_delay=0.1)

        # All tasks are independent (no dependencies)
        agent = create_test_pipeline_agent(
            "test_agent", [task1, task2, task3], {"enable_parallel": True}
        )

        # Measure execution time
        start_time = time.time()
        result = agent.run()
        end_time = time.time()

        # Verify all tasks completed
        assert result.completed_tasks == 3

        # Verify parallel execution (should be faster than sequential)
        # Sequential would take ~0.3s, parallel should take ~0.1s
        execution_time = end_time - start_time
        assert (
            execution_time < 0.25
        )  # Allow some overhead, but much faster than sequential

    def test_agent_populates_task_configs_from_context(self) -> None:
        """Task configs get populated with context values before execution."""
        # Create task that provides data
        search_task = MockTask(
            "search",
            {"query": "test"},
            run_result={
                "query": "test query",
                "results": [{"url": "http://example.com"}],
            },
        )

        # Create task that depends on search data
        fetch_deps = [
            TaskDependency("urls", "search.results[].url", DependencyType.REQUIRED),
            TaskDependency(
                "timeout", "config.timeout", DependencyType.OPTIONAL, default_value=30
            ),
        ]
        fetch_task = MockTask("fetch", {"base_config": "value"}, fetch_deps)

        # Setup initial context data
        agent = create_test_pipeline_agent("test_agent", [search_task, fetch_task])

        # Mock the dependency resolver instance after agent is created
        mock_resolver = Mock()
        agent.dependency_resolver = mock_resolver

        # Mock resolved configuration for both tasks
        def resolve_side_effect(task: BaseTask) -> Dict[str, Any]:
            if task.name == "search":
                return {"query": "test"}  # No dependencies, return original config
            elif task.name == "fetch":
                return {
                    "base_config": "value",  # Original config preserved
                    "urls": ["http://example.com"],  # From dependency
                    "timeout": 30,  # Default value
                }
            return {}

        mock_resolver.resolve_dependencies.side_effect = resolve_side_effect

        # Execute pipeline
        agent.run()

        # Verify dependency resolver was called for both tasks
        assert mock_resolver.resolve_dependencies.call_count == 2

        # Verify fetch task was called with dependencies resolved
        fetch_call_found = False
        for call in mock_resolver.resolve_dependencies.call_args_list:
            call_args = call[0]
            if call_args[0].name == "fetch":
                fetch_call_found = True
                break
        assert fetch_call_found, "Dependency resolver should be called for fetch task"

    def test_agent_builds_execution_plan(self) -> None:
        """Agent builds proper execution plan based on task dependencies."""
        # Create complex dependency graph
        task_a = MockTask("task_a", {})  # No dependencies
        task_b = MockTask(
            "task_b",
            {},
            [TaskDependency("a_data", "task_a.result", DependencyType.REQUIRED)],
        )
        task_c = MockTask("task_c", {})  # No dependencies
        task_d = MockTask(
            "task_d",
            {},
            [
                TaskDependency("b_data", "task_b.result", DependencyType.REQUIRED),
                TaskDependency("c_data", "task_c.result", DependencyType.REQUIRED),
            ],
        )

        agent = create_test_pipeline_agent(
            "test_agent", [task_d, task_b, task_a, task_c]
        )  # Add in random order

        # Get execution plan
        agent.setup_tasks()
        execution_plan = agent.get_execution_plan() or agent._build_execution_plan()  # pyright: ignore

        # Verify plan structure
        assert len(execution_plan.phases) > 0

        # Phase 1 should contain tasks with no dependencies (A and C)
        phase_1_names = {task.name for task in execution_plan.phases[0]}
        assert "task_a" in phase_1_names
        assert "task_c" in phase_1_names

        # Task B should come after A
        task_b_phase = next(
            i
            for i, phase in enumerate(execution_plan.phases)
            if any(task.name == "task_b" for task in phase)
        )
        task_a_phase = next(
            i
            for i, phase in enumerate(execution_plan.phases)
            if any(task.name == "task_a" for task in phase)
        )
        assert task_b_phase > task_a_phase

        # Task D should come after both B and C
        task_d_phase = next(
            i
            for i, phase in enumerate(execution_plan.phases)
            if any(task.name == "task_d" for task in phase)
        )
        assert task_d_phase > task_b_phase

    def test_agent_detects_circular_dependencies(self) -> None:
        """Agent detects and handles circular dependencies."""
        # Create circular dependency: A -> B -> C -> A
        task_a = MockTask(
            "task_a",
            {},
            [TaskDependency("c_data", "task_c.result", DependencyType.REQUIRED)],
        )
        task_b = MockTask(
            "task_b",
            {},
            [TaskDependency("a_data", "task_a.result", DependencyType.REQUIRED)],
        )
        task_c = MockTask(
            "task_c",
            {},
            [TaskDependency("b_data", "task_b.result", DependencyType.REQUIRED)],
        )

        agent = create_test_pipeline_agent("test_agent", [task_a, task_b, task_c])

        # Should detect circular dependency and raise error
        with pytest.raises(ValueError, match="Circular dependency detected"):
            agent.run()

    def test_agent_handles_empty_task_list(self) -> None:
        """Agent handles case with no tasks gracefully."""
        agent = create_test_pipeline_agent("test_agent", [])

        # Run with no tasks
        result = agent.run()

        assert result.status == AgentRunStatus.SUCCESS
        assert result.total_tasks == 0
        assert result.completed_tasks == 0

    def test_agent_maintains_task_context(self) -> None:
        """Agent maintains and updates task context throughout execution."""
        task1 = MockTask("task1", {}, run_result={"data": "result1"})
        task2 = MockTask("task2", {}, run_result={"data": "result2"})

        agent = create_test_pipeline_agent("test_agent", [task1, task2])

        # Execute and check context
        agent.run()

        # Verify context contains task results
        context = agent.get_context()
        assert context.has_result("task1")
        assert context.has_result("task2")
        task1_result = context.get_result("task1")
        task2_result = context.get_result("task2")
        assert task1_result is not None and task1_result["data"] == "result1"
        assert task2_result is not None and task2_result["data"] == "result2"

        # Verify execution history
        history = context.get_execution_history()
        assert len(history) == 2
        assert all(entry["status"] == "completed" for entry in history)


class TestTaskExecutionPlan:
    """Test suite for TaskExecutionPlan class."""

    def test_execution_plan_phases(self) -> None:
        """TaskExecutionPlan correctly organizes tasks into execution phases."""
        task1 = MockTask("independent1", {})
        task2 = MockTask("independent2", {})
        task3 = MockTask(
            "dependent",
            {},
            [
                TaskDependency("data1", "independent1.result", DependencyType.REQUIRED),
                TaskDependency("data2", "independent2.result", DependencyType.REQUIRED),
            ],
        )

        plan = TaskExecutionPlan([task1, task2, task3])

        # Should have 2 phases: [independent1, independent2] then [dependent]
        assert len(plan.phases) == 2
        assert len(plan.phases[0]) == 2  # Phase 1: independent tasks
        assert len(plan.phases[1]) == 1  # Phase 2: dependent task

        # Verify phase contents
        phase_1_names = {task.name for task in plan.phases[0]}
        assert phase_1_names == {"independent1", "independent2"}

        phase_2_names = {task.name for task in plan.phases[1]}
        assert phase_2_names == {"dependent"}

    def test_execution_plan_validates_dependencies(self) -> None:
        """TaskExecutionPlan validates that all dependencies can be satisfied."""
        # Task with unsatisfiable dependency
        task_with_missing_dep = MockTask(
            "task1",
            {},
            [
                TaskDependency(
                    "missing", "nonexistent_task.data", DependencyType.REQUIRED
                )
            ],
        )

        # Should raise error for unsatisfiable dependency
        with pytest.raises(ValueError, match="Cannot satisfy dependency"):
            TaskExecutionPlan([task_with_missing_dep])

    def test_execution_plan_optimization(self) -> None:
        """TaskExecutionPlan optimizes for maximum parallelism."""
        # Create a more complex dependency graph
        # A and B are independent
        # C depends on A
        # D depends on B
        # E depends on C and D

        task_a = MockTask("a", {})
        task_b = MockTask("b", {})
        task_c = MockTask(
            "c", {}, [TaskDependency("a_data", "a.result", DependencyType.REQUIRED)]
        )
        task_d = MockTask(
            "d", {}, [TaskDependency("b_data", "b.result", DependencyType.REQUIRED)]
        )
        task_e = MockTask(
            "e",
            {},
            [
                TaskDependency("c_data", "c.result", DependencyType.REQUIRED),
                TaskDependency("d_data", "d.result", DependencyType.REQUIRED),
            ],
        )

        plan = TaskExecutionPlan([task_a, task_b, task_c, task_d, task_e])

        # Should have 3 phases for optimal parallelism
        # Phase 1: [A, B] (independent)
        # Phase 2: [C, D] (depend on Phase 1, but independent of each other)
        # Phase 3: [E] (depends on Phase 2)

        assert len(plan.phases) == 3

        # Verify phase 1
        phase_1_names = {task.name for task in plan.phases[0]}
        assert phase_1_names == {"a", "b"}

        # Verify phase 2
        phase_2_names = {task.name for task in plan.phases[1]}
        assert phase_2_names == {"c", "d"}

        # Verify phase 3
        phase_3_names = {task.name for task in plan.phases[2]}
        assert phase_3_names == {"e"}

    def test_execution_plan_total_phases(self) -> None:
        """TaskExecutionPlan provides correct total phase count."""
        # Simple linear dependency: A -> B -> C
        task_a = MockTask("a", {})
        task_b = MockTask(
            "b", {}, [TaskDependency("a_data", "a.result", DependencyType.REQUIRED)]
        )
        task_c = MockTask(
            "c", {}, [TaskDependency("b_data", "b.result", DependencyType.REQUIRED)]
        )

        plan = TaskExecutionPlan([task_a, task_b, task_c])

        # Should have 3 phases (linear execution)
        assert plan.total_phases() == 3
        assert len(plan.phases) == 3

        # Each phase should have exactly one task
        for phase in plan.phases:
            assert len(phase) == 1
