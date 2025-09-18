"""Tests for BaseTask class."""

import pytest
import threading
import time
import unittest.mock
from datetime import datetime
from typing import Any, List
from unittest.mock import patch, MagicMock

from typing import override
from aipype import (
    BaseTask,
    TaskStatus,
    TaskResult,
    TaskDependency,
    DependencyType,
    TaskContext,
)


class MockTaskImplementation(BaseTask):
    """Concrete implementation of BaseTask for testing."""

    @override
    def run(self) -> TaskResult:
        return_value = self.config.get("return_value", "default_result")
        return TaskResult.success(data=return_value)


class TestBaseTaskInitialization:
    """Test BaseTask initialization."""

    def test_init_with_name_only(self) -> None:
        task = MockTaskImplementation("test_task")

        assert task.name == "test_task"
        assert task.config == {}
        assert task.get_status() == TaskStatus.NOT_STARTED
        assert task.get_result() is None
        assert task.get_error() is None
        assert isinstance(task.status_changed_at, datetime)

    def test_init_with_config(self) -> None:
        config = {"key": "value", "number": 42}
        task = MockTaskImplementation("test_task", config)

        assert task.name == "test_task"
        assert task.config == config
        assert task.get_status() == TaskStatus.NOT_STARTED

    @patch("aipype.base_task.setup_logger")
    def test_logger_setup(self, mock_setup_logger: MagicMock) -> None:
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        task = MockTaskImplementation("test_task")

        mock_setup_logger.assert_called_once_with("task.test_task")
        assert task.logger == mock_logger


class TestBaseTaskStatusManagement:
    """Test BaseTask status management methods."""

    def test_initial_status(self) -> None:
        task = MockTaskImplementation("test_task")

        assert task.get_status() == TaskStatus.NOT_STARTED
        assert not task.is_completed()
        assert not task.has_error()
        assert task.get_result() is None
        assert task.get_error() is None

    def test_mark_started(self) -> None:
        task = MockTaskImplementation("test_task")
        initial_time = task.status_changed_at

        task.mark_started()

        assert task.get_status() == TaskStatus.STARTED
        assert task.status_changed_at > initial_time
        assert not task.is_completed()
        assert not task.has_error()

    def test_mark_success(self) -> None:
        task = MockTaskImplementation("test_task")
        result = "test_result"

        task.mark_success(result)

        assert task.get_status() == TaskStatus.SUCCESS
        assert task.is_completed()
        assert not task.has_error()
        assert task.get_result() == result
        assert task.get_error() is None

    def test_mark_success_without_result(self) -> None:
        task = MockTaskImplementation("test_task")

        task.mark_success()

        assert task.get_status() == TaskStatus.SUCCESS
        assert task.is_completed()
        assert task.get_result() is None

    def test_mark_error(self) -> None:
        task = MockTaskImplementation("test_task")
        error_msg = "Something went wrong"

        task.mark_error(error_msg)

        assert task.get_status() == TaskStatus.ERROR
        assert not task.is_completed()
        assert task.has_error()
        assert task.get_error() == error_msg
        assert task.get_result() is None

    def test_status_transitions(self) -> None:
        task = MockTaskImplementation("test_task")

        # NOT_STARTED -> STARTED
        task.mark_started()
        assert task.get_status() == TaskStatus.STARTED

        # STARTED -> SUCCESS
        task.mark_success("result")
        assert task.get_status() == TaskStatus.SUCCESS
        assert task.get_result() == "result"

    def test_status_transition_to_error(self) -> None:
        task = MockTaskImplementation("test_task")

        # NOT_STARTED -> STARTED -> ERROR
        task.mark_started()
        task.mark_error("failed")

        assert task.get_status() == TaskStatus.ERROR
        assert task.get_error() == "failed"
        assert task.get_result() is None

    def test_status_change_updates_timestamp(self) -> None:
        task = MockTaskImplementation("test_task")
        initial_time = task.status_changed_at

        # Small delay to ensure timestamp difference
        import time

        time.sleep(0.01)

        task.mark_started()
        assert task.status_changed_at > initial_time

        started_time = task.status_changed_at
        time.sleep(0.01)

        task.mark_success("result")
        assert task.status_changed_at > started_time


class TestBaseTaskReset:
    """Test BaseTask reset functionality."""

    def test_reset_from_success(self) -> None:
        task = MockTaskImplementation("test_task")

        # Set up task in SUCCESS state
        task.mark_started()
        task.mark_success("result")

        # Reset task
        task.reset()

        assert task.get_status() == TaskStatus.NOT_STARTED
        assert task.get_result() is None
        assert task.get_error() is None
        assert not task.is_completed()
        assert not task.has_error()

    def test_reset_from_error(self) -> None:
        task = MockTaskImplementation("test_task")

        # Set up task in ERROR state
        task.mark_started()
        task.mark_error("error message")

        # Reset task
        task.reset()

        assert task.get_status() == TaskStatus.NOT_STARTED
        assert task.get_result() is None
        assert task.get_error() is None
        assert not task.is_completed()
        assert not task.has_error()

    def test_reset_updates_timestamp(self) -> None:
        task = MockTaskImplementation("test_task")
        task.mark_success("result")
        success_time = task.status_changed_at

        import time

        time.sleep(0.01)

        task.reset()
        assert task.status_changed_at > success_time


class TestBaseTaskStringRepresentation:
    """Test BaseTask string representation."""

    def test_str_not_started(self) -> None:
        task = MockTaskImplementation("test_task")
        str_repr = str(task)

        assert "test_task" in str_repr
        assert "not_started" in str_repr
        assert "changed_at" in str_repr

    def test_str_success(self) -> None:
        task = MockTaskImplementation("test_task")
        task.mark_success("result")
        str_repr = str(task)

        assert "test_task" in str_repr
        assert "success" in str_repr
        assert "changed_at" in str_repr

    def test_str_error(self) -> None:
        task = MockTaskImplementation("test_task")
        task.mark_error("error")
        str_repr = str(task)

        assert "test_task" in str_repr
        assert "error" in str_repr
        assert "changed_at" in str_repr


class TestBaseTaskEdgeCases:
    """Test BaseTask edge cases and error conditions."""

    def test_get_result_only_returns_for_success(self) -> None:
        task = MockTaskImplementation("test_task")

        # Not started
        assert task.get_result() is None

        # Started but not completed
        task.mark_started()
        assert task.get_result() is None

        # Error state
        task.mark_error("error")
        assert task.get_result() is None

        # Reset and mark success
        task.reset()
        task.mark_success("result")
        assert task.get_result() == "result"

    def test_get_error_only_returns_for_error(self) -> None:
        task = MockTaskImplementation("test_task")

        # Not started
        assert task.get_error() is None

        # Started but not completed
        task.mark_started()
        assert task.get_error() is None

        # Success state
        task.mark_success("result")
        assert task.get_error() is None

        # Reset and mark error
        task.reset()
        task.mark_error("error message")
        assert task.get_error() == "error message"

    def test_multiple_status_changes_clear_previous_state(self) -> None:
        task = MockTaskImplementation("test_task")

        # Mark success, then error
        task.mark_success("result")
        assert task.get_result() == "result"
        assert task.get_error() is None

        task.mark_error("error")
        assert task.get_result() is None
        assert task.get_error() == "error"

        # Mark success again
        task.mark_success("new_result")
        assert task.get_result() == "new_result"
        assert task.get_error() is None


class TestBaseTaskHelperMethods:
    """Test BaseTask helper methods."""

    def test_get_dependencies_with_dependencies(self) -> None:
        """Test get_dependencies returns correct dependency list."""
        dependencies = [
            TaskDependency("dep1", "source1.field", DependencyType.REQUIRED),
            TaskDependency("dep2", "source2.data", DependencyType.OPTIONAL),
        ]
        task = MockTaskImplementation("test_task", {}, dependencies)

        result = task.get_dependencies()
        assert result == dependencies
        assert len(result) == 2
        assert result[0].name == "dep1"
        assert result[1].name == "dep2"

    def test_get_dependencies_empty_when_none(self) -> None:
        """Test get_dependencies returns empty list when no dependencies."""
        task = MockTaskImplementation("test_task")

        result = task.get_dependencies()
        assert result == []
        assert len(result) == 0

    def test_get_dependencies_empty_when_none_explicit(self) -> None:
        """Test get_dependencies with explicitly passed None dependencies."""
        task = MockTaskImplementation("test_task", {}, None)

        result = task.get_dependencies()
        assert result == []

    def test_set_agent_name_stores_correctly(self) -> None:
        """Test set_agent_name stores agent name correctly."""
        task = MockTaskImplementation("test_task")

        task.set_agent_name("test_agent")

        assert task.agent_name == "test_agent"

    def test_set_agent_name_initial_none(self) -> None:
        """Test agent_name is initially None."""
        task = MockTaskImplementation("test_task")

        assert task.agent_name is None

    def test_set_agent_name_can_be_changed(self) -> None:
        """Test agent_name can be changed multiple times."""
        task = MockTaskImplementation("test_task")

        task.set_agent_name("first_agent")
        assert task.agent_name == "first_agent"

        task.set_agent_name("second_agent")
        assert task.agent_name == "second_agent"

    def test_set_context_default_implementation(self) -> None:
        """Test set_context default implementation does nothing."""
        task = MockTaskImplementation("test_task")
        context = TaskContext()

        # Should not raise any exception
        task.set_context(context)
        # No state should change since base implementation is no-op
        assert task.get_status() == TaskStatus.NOT_STARTED


class TestBaseTaskLogging:
    """Test BaseTask logging behavior."""

    @patch("aipype.base_task.setup_logger")
    def test_logger_setup_with_task_name(self, mock_setup_logger: MagicMock) -> None:
        """Test logger is set up with correct task name."""
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        task = MockTaskImplementation("my_task")

        mock_setup_logger.assert_called_once_with("task.my_task")
        assert task.logger == mock_logger

    @patch("aipype.base_task.setup_logger")
    def test_status_change_logging_started(self, mock_setup_logger: MagicMock) -> None:
        """Test logging when task status changes to STARTED."""
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        task = MockTaskImplementation("test_task")
        task.mark_started()

        mock_logger.info.assert_called_with(
            "Task 'test_task' status changed from not_started to started"
        )

    @patch("aipype.base_task.setup_logger")
    def test_status_change_logging_success(self, mock_setup_logger: MagicMock) -> None:
        """Test logging when task status changes to SUCCESS."""
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        task = MockTaskImplementation("test_task")
        task.mark_success("result")

        mock_logger.info.assert_called_with(
            "Task 'test_task' status changed from not_started to success"
        )

    @patch("aipype.base_task.setup_logger")
    def test_status_change_logging_error(self, mock_setup_logger: MagicMock) -> None:
        """Test logging when task status changes to ERROR."""
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        task = MockTaskImplementation("test_task")
        task.mark_error("error message")

        mock_logger.info.assert_called_with(
            "Task 'test_task' status changed from not_started to error"
        )

    @patch("aipype.base_task.setup_logger")
    def test_status_change_logging_reset(self, mock_setup_logger: MagicMock) -> None:
        """Test logging when task is reset."""
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        task = MockTaskImplementation("test_task")
        task.mark_success("result")
        mock_logger.reset_mock()  # Clear previous calls

        task.reset()

        # Should log both status change and reset message
        expected_calls = [
            unittest.mock.call(
                "Task 'test_task' status changed from success to not_started"
            ),
            unittest.mock.call("Task 'test_task' reset"),
        ]
        mock_logger.info.assert_has_calls(expected_calls, any_order=False)

    @patch("aipype.base_task.setup_logger")
    def test_set_agent_name_logging(self, mock_setup_logger: MagicMock) -> None:
        """Test logging when agent name is set."""
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        task = MockTaskImplementation("test_task")
        task.set_agent_name("my_agent")

        mock_logger.debug.assert_called_with(
            "Task 'test_task' assigned to agent 'my_agent'"
        )

    @patch("aipype.base_task.setup_logger")
    def test_multiple_status_changes_logging(
        self, mock_setup_logger: MagicMock
    ) -> None:
        """Test logging for multiple status changes."""
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        task = MockTaskImplementation("test_task")

        task.mark_started()
        task.mark_success("result")
        task.mark_error("error")

        expected_calls = [
            unittest.mock.call(
                "Task 'test_task' status changed from not_started to started"
            ),
            unittest.mock.call(
                "Task 'test_task' status changed from started to success"
            ),
            unittest.mock.call("Task 'test_task' status changed from success to error"),
        ]
        mock_logger.info.assert_has_calls(expected_calls, any_order=False)


class TestBaseTaskEdgeCasesExtended:
    """Test BaseTask edge cases and error scenarios."""

    def test_initialization_with_complex_dependencies(self) -> None:
        """Test initialization with complex dependency structures."""

        # Complex dependencies with transform functions
        def transform_func(x: Any) -> str:
            return str(x).upper()

        dependencies = [
            TaskDependency("dep1", "source1.field", DependencyType.REQUIRED),
            TaskDependency(
                "dep2",
                "source2.data",
                DependencyType.OPTIONAL,
                transform_func=transform_func,
            ),
            TaskDependency("dep3", "source3.nested.value", DependencyType.REQUIRED),
        ]

        task = MockTaskImplementation("complex_task", {"param": "value"}, dependencies)

        assert len(task.get_dependencies()) == 3
        assert task.get_dependencies()[1].transform_func == transform_func
        assert task.config["param"] == "value"

    def test_status_change_with_none_values(self) -> None:
        """Test status changes with None values for result/error."""
        task = MockTaskImplementation("test_task")

        # Success with None result
        task.mark_success(None)
        assert task.get_status() == TaskStatus.SUCCESS
        assert task.get_result() is None
        assert task.get_error() is None

        # Error with empty string
        task.mark_error("")
        assert task.get_status() == TaskStatus.ERROR
        assert task.get_error() == ""
        assert task.get_result() is None

    def test_status_change_with_complex_result_data(self) -> None:
        """Test status changes with complex result data structures."""
        task = MockTaskImplementation("test_task")

        complex_result = {
            "data": [1, 2, 3],
            "metadata": {"count": 3, "source": "test"},
            "nested": {"deep": {"value": "found"}},
        }

        task.mark_success(complex_result)

        result = task.get_result()
        assert result == complex_result
        assert result is not None
        assert result["data"] == [1, 2, 3]
        assert result["nested"]["deep"]["value"] == "found"

    def test_timestamp_precision_and_ordering(self) -> None:
        """Test timestamp precision and ordering of status changes."""
        task = MockTaskImplementation("test_task")

        initial_time = task.status_changed_at

        # Small delay to ensure timestamp differences
        time.sleep(0.001)

        task.mark_started()
        start_time = task.status_changed_at
        assert start_time > initial_time

        time.sleep(0.001)
        task.mark_success("result")
        success_time = task.status_changed_at
        assert success_time > start_time

        time.sleep(0.001)
        task.reset()
        reset_time = task.status_changed_at
        assert reset_time > success_time

    def test_string_representation_with_special_characters(self) -> None:
        """Test string representation with special characters in task name."""
        special_names = [
            "task-with-dashes",
            "task_with_underscores",
            "task.with.dots",
            "task with spaces",
            "task/with/slashes",
        ]

        for name in special_names:
            task = MockTaskImplementation(name)
            str_repr = str(task)
            assert name in str_repr
            assert "not_started" in str_repr
            assert "changed_at" in str_repr

    def test_concurrent_status_access_safety(self) -> None:
        """Test thread safety of status access (basic check)."""
        task = MockTaskImplementation("concurrent_task")
        results: List[tuple[int, TaskStatus, Any]] = []
        errors: List[str] = []

        def worker(worker_id: int) -> None:
            try:
                for i in range(10):
                    if worker_id % 2 == 0:
                        task.mark_started()
                        time.sleep(0.001)
                        task.mark_success(f"result_{worker_id}_{i}")
                    else:
                        status = task.get_status()
                        result = task.get_result()
                        results.append((worker_id, status, result))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"Worker {worker_id}: {str(e)}")

        threads: List[threading.Thread] = []
        for i in range(4):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should complete without errors
        assert len(errors) == 0, f"Concurrent access errors: {errors}"

    def test_rapid_status_transitions_handling(self) -> None:
        """Test handling of rapid status transitions."""
        task = MockTaskImplementation("rapid_task")

        # Rapid status changes should all be recorded
        task.mark_started()
        task.mark_success("first")
        task.mark_error("error")
        task.mark_success("second")
        task.reset()
        task.mark_error("final_error")

        # Final state should be ERROR with the last error message
        assert task.get_status() == TaskStatus.ERROR
        assert task.get_error() == "final_error"
        assert task.get_result() is None

    def test_status_change_parameter_combinations(self) -> None:
        """Test various parameter combinations through public methods."""
        task = MockTaskImplementation("test_task")

        # Test success with None result
        task.mark_success(None)
        assert task.get_result() is None
        assert task.get_error() is None

        # Test success with complex result
        complex_result = {"nested": {"data": [1, 2, 3]}}
        task.mark_success(complex_result)
        assert task.get_result() == complex_result

        # Test error clears previous result
        task.mark_error("test error")
        assert task.get_result() is None
        assert task.get_error() == "test error"

        # Test reset clears everything
        task.reset()
        assert task.get_result() is None
        assert task.get_error() is None


class TestBaseTaskAbstractContract:
    """Test BaseTask abstract method contract."""

    def test_cannot_instantiate_base_task_directly(self) -> None:
        """Test that BaseTask cannot be instantiated directly."""
        with pytest.raises(
            TypeError, match="Can't instantiate abstract class BaseTask"
        ):
            BaseTask("test_task")  # type: ignore

    def test_subclass_must_implement_run_method(self) -> None:
        """Test that subclasses must implement the run method."""

        class IncompleteTask(BaseTask):
            # Missing run() method implementation
            pass

        with pytest.raises(
            TypeError, match="Can't instantiate abstract class IncompleteTask"
        ):
            IncompleteTask("incomplete_task")  # type: ignore

    def test_run_method_can_return_any_type(self) -> None:
        """Test that run method can return various types."""

        class FlexibleTask(BaseTask):
            def __init__(self, name: str, return_value: Any):
                super().__init__(name)
                self.return_value = return_value

            @override
            def run(self) -> TaskResult:
                return TaskResult.success(data=self.return_value)

        # Test different return types
        string_task = FlexibleTask("string_task", "string_result")
        result = string_task.run()
        assert result.is_success()
        assert result.data == "string_result"

        dict_task = FlexibleTask("dict_task", {"key": "value"})
        dict_result = dict_task.run()
        assert dict_result.is_success()
        assert dict_result.data == {"key": "value"}

        none_task = FlexibleTask("none_task", None)
        none_result = none_task.run()
        assert none_result.is_success()
        assert none_result.data is None

        list_task = FlexibleTask("list_task", [1, 2, 3])
        list_result = list_task.run()
        assert list_result.is_success()
        assert list_result.data == [1, 2, 3]
