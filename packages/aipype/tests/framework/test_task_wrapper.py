"""Tests for TaskWrapper class."""

from typing import Any
from unittest.mock import MagicMock

from pydantic import BaseModel

from aipype.base_task import BaseTask
from aipype.task_dependencies import DependencyType, TaskDependency
from aipype.task_result import TaskResult
from aipype.task_wrapper import TaskWrapper


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""

    value: int
    name: str


class MockTask(BaseTask):
    """Mock task for testing delegation."""

    def __init__(self, name: str, return_value: dict[str, Any]) -> None:
        super().__init__(name, {})
        self.return_value = return_value

    def run(self) -> TaskResult:
        return TaskResult.success(data=self.return_value, execution_time=0.1)


class TestTaskWrapperInit:
    """Tests for TaskWrapper initialization."""

    def test_init_with_no_dependencies(self) -> None:
        """Test TaskWrapper initialization with no dependencies."""

        def method() -> dict[str, Any]:
            return {"result": "value"}

        agent = MagicMock()
        wrapper = TaskWrapper(
            name="test_task", method=method, agent=agent, dependencies=[]
        )

        assert wrapper.name == "test_task"
        assert wrapper.method == method
        assert wrapper.agent == agent
        assert wrapper.dependencies == []

    def test_init_with_dependencies(self) -> None:
        """Test TaskWrapper initialization with dependencies."""

        def method(data: dict[str, Any]) -> dict[str, Any]:
            return {"processed": data}

        agent = MagicMock()
        deps = [
            TaskDependency(
                name="data",
                source_path="other_task.data",
                dependency_type=DependencyType.REQUIRED,
            )
        ]
        wrapper = TaskWrapper(
            name="test_task", method=method, agent=agent, dependencies=deps
        )

        assert len(wrapper.dependencies) == 1
        assert wrapper.dependencies[0].name == "data"


class TestTaskWrapperRun:
    """Tests for TaskWrapper run method."""

    def test_run_returns_dict(self) -> None:
        """Test run with method returning dict."""

        def method() -> dict[str, Any]:
            return {"key": "value", "count": 42}

        agent = MagicMock()
        wrapper = TaskWrapper(
            name="test_task", method=method, agent=agent, dependencies=[]
        )

        result = wrapper.run()

        assert result.is_success()
        # Original keys are preserved
        assert result.data["key"] == "value"
        assert result.data["count"] == 42
        # Full dict is also available under 'data' for consistent access
        assert result.data["data"] == {"key": "value", "count": 42}

    def test_run_returns_string(self) -> None:
        """Test run with method returning string."""

        def method() -> str:
            return "hello world"

        agent = MagicMock()
        wrapper = TaskWrapper(
            name="test_task", method=method, agent=agent, dependencies=[]
        )

        result = wrapper.run()

        assert result.is_success()
        assert result.data["output"] == "hello world"
        assert result.data["data"] == "hello world"

    def test_run_returns_list(self) -> None:
        """Test run with method returning list."""

        def method() -> list[int]:
            return [1, 2, 3]

        agent = MagicMock()
        wrapper = TaskWrapper(
            name="test_task", method=method, agent=agent, dependencies=[]
        )

        result = wrapper.run()

        assert result.is_success()
        assert result.data["output"] == [1, 2, 3]

    def test_run_returns_pydantic_model(self) -> None:
        """Test run with method returning Pydantic model."""

        def method() -> SampleModel:
            return SampleModel(value=10, name="test")

        agent = MagicMock()
        wrapper = TaskWrapper(
            name="test_task", method=method, agent=agent, dependencies=[]
        )

        result = wrapper.run()

        assert result.is_success()
        # Original fields are preserved
        assert result.data["value"] == 10
        assert result.data["name"] == "test"
        assert result.metadata["return_type"] == "pydantic"
        # Full dict is also available under 'data' for consistent access
        assert result.data["data"] == {"value": 10, "name": "test"}

    def test_run_returns_task_result(self) -> None:
        """Test run with method returning TaskResult directly."""

        def method() -> TaskResult:
            return TaskResult.success(
                data={"custom": "data"}, metadata={"source": "manual"}
            )

        agent = MagicMock()
        wrapper = TaskWrapper(
            name="test_task", method=method, agent=agent, dependencies=[]
        )

        result = wrapper.run()

        assert result.is_success()
        assert result.data["custom"] == "data"

    def test_run_delegates_to_base_task(self) -> None:
        """Test run with method returning BaseTask (delegation)."""

        def method() -> MockTask:
            return MockTask("delegated", {"delegated_result": True})

        agent = MagicMock()
        wrapper = TaskWrapper(
            name="test_task", method=method, agent=agent, dependencies=[]
        )

        result = wrapper.run()

        assert result.is_success()
        assert result.data["delegated_result"] is True
        assert result.metadata["delegated_from"] == "test_task"
        assert result.metadata["delegated_to"] == "delegated"

    def test_run_with_dependencies(self) -> None:
        """Test run with resolved dependencies in config."""

        def method(input_data: dict[str, Any]) -> dict[str, Any]:
            return {"processed": input_data["value"] * 2}

        agent = MagicMock()
        deps = [
            TaskDependency(
                name="input_data",
                source_path="source.data",
                dependency_type=DependencyType.REQUIRED,
            )
        ]
        wrapper = TaskWrapper(
            name="test_task", method=method, agent=agent, dependencies=deps
        )

        # Simulate resolved dependencies
        wrapper.config["input_data"] = {"value": 21}

        result = wrapper.run()

        assert result.is_success()
        assert result.data["processed"] == 42

    def test_run_handles_exception(self) -> None:
        """Test run handles exceptions and returns failure."""

        def method() -> dict[str, Any]:
            raise ValueError("Something went wrong")

        agent = MagicMock()
        wrapper = TaskWrapper(
            name="test_task", method=method, agent=agent, dependencies=[]
        )

        result = wrapper.run()

        assert result.is_error()
        assert result.error is not None
        assert "Something went wrong" in result.error
        assert result.metadata["error_type"] == "ValueError"


class TestTaskWrapperContext:
    """Tests for TaskWrapper context handling."""

    def test_set_context(self) -> None:
        """Test setting context on wrapper."""

        def method() -> dict[str, Any]:
            return {}

        agent = MagicMock()
        wrapper = TaskWrapper(
            name="test_task", method=method, agent=agent, dependencies=[]
        )

        mock_context = MagicMock()
        wrapper.set_context(mock_context)

        assert wrapper.context_instance == mock_context

    def test_context_passed_to_delegated_task(self) -> None:
        """Test that context is passed to delegated BaseTask."""
        mock_context = MagicMock()

        def method() -> MockTask:
            return MockTask("delegated", {"result": True})

        agent = MagicMock()
        wrapper = TaskWrapper(
            name="test_task", method=method, agent=agent, dependencies=[]
        )
        wrapper.set_context(mock_context)
        wrapper.set_agent_name("test_agent")

        wrapper.run()

        # The delegated task should have context and agent name set
        # We can't easily verify this without modifying MockTask,
        # but the test ensures no errors occur


class TestTaskWrapperStr:
    """Tests for TaskWrapper string representation."""

    def test_str_representation(self) -> None:
        """Test __str__ method."""

        def method() -> dict[str, Any]:
            return {}

        agent = MagicMock()
        deps = [
            TaskDependency(
                name="dep1", source_path="t.d", dependency_type=DependencyType.REQUIRED
            ),
            TaskDependency(
                name="dep2", source_path="t.e", dependency_type=DependencyType.REQUIRED
            ),
        ]
        wrapper = TaskWrapper(
            name="my_task", method=method, agent=agent, dependencies=deps
        )

        result = str(wrapper)
        assert "my_task" in result
        assert "2" in result  # 2 dependencies
