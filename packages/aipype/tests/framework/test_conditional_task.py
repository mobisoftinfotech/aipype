"""Comprehensive tests for ConditionalTask - Task that executes based on context conditions."""

import pytest
from typing import Any, Dict
from unittest.mock import Mock, patch
from aipype import (
    ConditionalTask,
    threshold_condition,
    contains_condition,
    list_size_condition,
    success_rate_condition,
    quality_gate_condition,
    log_action,
    increment_counter_action,
    set_flag_action,
    TaskContext,
    TaskDependency,
    DependencyType,
)


class TestConditionalTaskCore:
    """Test core ConditionalTask functionality."""

    def test_conditional_task_initialization(self) -> None:
        """ConditionalTask initializes correctly with config and dependencies."""
        dependencies = [
            TaskDependency("test_data", "source.data", DependencyType.REQUIRED)
        ]

        def condition_func(x: int) -> bool:
            return x > 5

        def action_func() -> Dict[str, str]:
            return {"result": "executed"}

        config: Dict[str, Any] = {
            "condition_function": condition_func,
            "action_function": action_func,
        }

        task = ConditionalTask("test_conditional", config, dependencies)

        assert task.name == "test_conditional"
        assert task.config == config
        assert task.get_dependencies() == dependencies
        assert task.context_instance is None

    def test_set_context_stores_context_instance(self) -> None:
        """set_context method stores TaskContext instance."""
        task = ConditionalTask("test", {})
        context = TaskContext()

        task.set_context(context)

        assert task.context_instance is context

    def test_run_requires_condition_function(self) -> None:
        """ConditionalTask run() requires condition_function in config."""
        task = ConditionalTask("test", {})

        result = task.run()
        assert result.is_error()
        assert result.error is not None
        assert "Missing required fields: ['condition_function']" in result.error

    def test_run_validates_condition_function_callable(self) -> None:
        """ConditionalTask validates condition_function is callable."""
        task = ConditionalTask("test", {"condition_function": "not_callable"})

        result = task.run()
        assert result.is_error()
        assert result.error is not None
        assert "condition_function failed custom validation" in result.error

    def test_condition_true_executes_action(self) -> None:
        """ConditionalTask executes action when condition is True."""

        def condition() -> bool:
            return True

        def action() -> Dict[str, Any]:
            return {"message": "action executed"}

        task = ConditionalTask(
            "test",
            {
                "condition_function": condition,
                "action_function": action,
            },
        )

        result = task.run()

        assert result.is_success()
        assert result.data["condition_result"] is True
        assert result.data["executed"] is True
        assert result.data["action_result"]["message"] == "action executed"
        assert result.data["else_result"] is None
        assert result.data["skip_reason"] is None

    def test_condition_false_skips_action_without_else(self) -> None:
        """ConditionalTask skips execution when condition is False and no else action."""

        def condition() -> bool:
            return False

        task = ConditionalTask(
            "test",
            {
                "condition_function": condition,
                "action_function": lambda: {"executed": True},
            },
        )

        result = task.run()

        assert result.is_success()
        assert result.data["condition_result"] is False
        assert result.data["executed"] is False
        assert result.data["action_result"] is None
        assert result.data["else_result"] is None
        assert result.data["skip_reason"] == "Condition evaluated to False"

    def test_condition_false_executes_else_action(self) -> None:
        """ConditionalTask executes else action when condition is False."""

        def condition() -> bool:
            return False

        def else_action() -> Dict[str, Any]:
            return {"message": "else executed"}

        task = ConditionalTask(
            "test",
            {
                "condition_function": condition,
                "else_function": else_action,
            },
        )

        result = task.run()

        assert result.is_success()
        assert result.data["condition_result"] is False
        assert result.data["executed"] is True  # else action counts as execution
        assert result.data["action_result"] is None
        assert result.data["else_result"]["message"] == "else executed"
        assert result.data["skip_reason"] is None

    def test_custom_skip_reason(self) -> None:
        """ConditionalTask uses custom skip reason when provided."""
        task = ConditionalTask(
            "test",
            {
                "condition_function": lambda: False,
                "skip_reason": "Custom skip message",
            },
        )

        result = task.run()

        assert result.is_success()
        assert result.data["skip_reason"] == "Custom skip message"

    def test_action_function_validation(self) -> None:
        """ConditionalTask validates action_function is callable."""
        task = ConditionalTask(
            "test",
            {
                "condition_function": lambda: True,
                "action_function": "not_callable",
            },
        )

        result = task.run()
        assert result.is_error()
        assert result.error is not None
        assert "action_function failed custom validation" in result.error

    def test_else_function_validation(self) -> None:
        """ConditionalTask validates else_function is callable."""
        task = ConditionalTask(
            "test",
            {
                "condition_function": lambda: False,
                "else_function": "not_callable",
            },
        )

        result = task.run()
        assert result.is_error()
        assert result.error is not None
        assert "else_function failed custom validation" in result.error


class TestConditionalTaskInputHandling:
    """Test ConditionalTask input parameter handling."""

    def test_condition_with_no_inputs(self) -> None:
        """ConditionalTask handles condition functions with no inputs."""
        task = ConditionalTask(
            "test",
            {
                "condition_function": lambda: True,
                "action_function": lambda: {"result": "success"},
            },
        )

        result = task.run()

        assert result.data["condition_result"] is True
        assert result.data["executed"] is True

    def test_condition_with_single_input(self) -> None:
        """ConditionalTask handles condition functions with single input."""

        def condition(value: int) -> bool:
            return value > 10

        task = ConditionalTask(
            "test",
            {
                "condition_function": condition,
                "condition_inputs": ["test_value"],
                "test_value": 15,
                "action_function": lambda: {"result": "success"},
            },
        )

        result = task.run()

        assert result.data["condition_result"] is True
        assert result.data["executed"] is True

    def test_condition_with_multiple_inputs(self) -> None:
        """ConditionalTask handles condition functions with multiple inputs."""

        def condition(min_val: int, max_val: int, test_val: int) -> bool:
            return min_val <= test_val <= max_val

        task = ConditionalTask(
            "test",
            {
                "condition_function": condition,
                "condition_inputs": ["min_val", "max_val", "test_val"],
                "min_val": 5,
                "max_val": 15,
                "test_val": 10,
                "action_function": lambda: {"result": "in_range"},
            },
        )

        result = task.run()

        assert result.data["condition_result"] is True
        assert result.data["action_result"]["result"] == "in_range"

    def test_action_with_single_input(self) -> None:
        """ConditionalTask handles action functions with single input."""

        def action(message: str) -> Dict[str, Any]:
            return {"processed": message.upper()}

        task = ConditionalTask(
            "test",
            {
                "condition_function": lambda: True,
                "action_function": action,
                "action_inputs": ["message"],
                "message": "hello world",
            },
        )

        result = task.run()

        assert result.data["action_result"]["processed"] == "HELLO WORLD"

    def test_action_with_multiple_inputs(self) -> None:
        """ConditionalTask handles action functions with multiple inputs."""

        def action(name: str, age: int) -> Dict[str, Any]:
            return {"greeting": f"Hello {name}, you are {age} years old"}

        task = ConditionalTask(
            "test",
            {
                "condition_function": lambda: True,
                "action_function": action,
                "action_inputs": ["name", "age"],
                "name": "Alice",
                "age": 25,
            },
        )

        result = task.run()

        assert (
            "Hello Alice, you are 25 years old"
            in result.data["action_result"]["greeting"]
        )

    def test_missing_condition_input_error(self) -> None:
        """ConditionalTask returns error when condition input is missing."""

        def condition_func(x: Any) -> bool:
            return int(x) > 5

        task = ConditionalTask(
            "test",
            {
                "condition_function": condition_func,
                "condition_inputs": ["missing_value"],
            },
        )

        result = task.run()
        assert result.is_error()
        assert result.error is not None
        assert "Condition evaluation failed" in result.error

    def test_missing_action_input_error(self) -> None:
        """ConditionalTask returns error when action input is missing."""

        def action_func(x: Any) -> Dict[str, str]:
            return {"value": str(x)}

        task = ConditionalTask(
            "test",
            {
                "condition_function": lambda: True,
                "action_function": action_func,
                "action_inputs": ["missing_value"],
            },
        )

        result = task.run()
        assert result.is_error()
        assert result.error is not None
        assert "Action execution failed" in result.error

    def test_condition_exception_handling(self) -> None:
        """ConditionalTask handles exceptions in condition evaluation."""

        def failing_condition() -> bool:
            raise ValueError("Condition failed")

        task = ConditionalTask(
            "test",
            {
                "condition_function": failing_condition,
            },
        )

        result = task.run()
        assert result.is_error()
        assert result.error is not None
        assert "Condition evaluation failed" in result.error

    def test_action_exception_handling(self) -> None:
        """ConditionalTask handles exceptions in action execution."""

        def failing_action() -> Dict[str, Any]:
            raise ValueError("Action failed")

        task = ConditionalTask(
            "test",
            {
                "condition_function": lambda: True,
                "action_function": failing_action,
            },
        )

        result = task.run()
        assert result.is_error()
        assert result.error is not None
        assert "Action execution failed" in result.error


class TestConditionalTaskPreview:
    """Test ConditionalTask preview functionality."""

    def test_preview_condition_success(self) -> None:
        """preview_condition returns correct information for valid condition."""

        def condition(value: int) -> bool:
            return value > 5

        task = ConditionalTask(
            "test",
            {
                "condition_function": condition,
                "condition_inputs": ["test_value"],
                "test_value": 10,
                "action_function": lambda: {"result": "success"},
            },
        )

        preview = task.preview_condition()

        assert preview["task_name"] == "test"
        assert preview["condition_result"] is True
        assert preview["condition_inputs"] == ["test_value"]
        assert preview["would_execute"] == "action"
        assert preview["skip_reason"] is None
        assert "input_values" in preview
        assert preview["input_values"]["test_value"]["value"] == "10"

    def test_preview_condition_false_with_else(self) -> None:
        """preview_condition shows else execution when condition is false."""
        task = ConditionalTask(
            "test",
            {
                "condition_function": lambda: False,
                "else_function": lambda: {"else": "executed"},
            },
        )

        preview = task.preview_condition()

        assert preview["condition_result"] is False
        assert preview["would_execute"] == "else"
        # When there's an else function, skip_reason should still be set but ignored
        assert "skip_reason" in preview

    def test_preview_condition_false_with_skip(self) -> None:
        """preview_condition shows skip when condition is false and no else."""
        task = ConditionalTask(
            "test",
            {
                "condition_function": lambda: False,
                "skip_reason": "Not ready to execute",
            },
        )

        preview = task.preview_condition()

        assert preview["condition_result"] is False
        assert preview["would_execute"] == "skip"
        assert preview["skip_reason"] == "Not ready to execute"

    def test_preview_condition_truncates_long_values(self) -> None:
        """preview_condition truncates long input values for display."""
        long_value = "x" * 150  # Longer than 100 characters

        def condition_func(text: Any) -> bool:
            return len(str(text)) > 100

        task = ConditionalTask(
            "test",
            {
                "condition_function": condition_func,
                "condition_inputs": ["long_text"],
                "long_text": long_value,
            },
        )

        preview = task.preview_condition()

        displayed_value = preview["input_values"]["long_text"]["value"]
        assert len(displayed_value) <= 103  # 100 chars + "..."
        assert displayed_value.endswith("...")

    def test_preview_condition_no_function_error(self) -> None:
        """preview_condition returns error when no condition function."""
        task = ConditionalTask("test", {})

        preview = task.preview_condition()

        assert "error" in preview
        assert "No condition_function specified" in preview["error"]

    def test_preview_condition_exception_handling(self) -> None:
        """preview_condition handles exceptions gracefully."""

        def failing_condition() -> bool:
            raise ValueError("Preview failed")

        task = ConditionalTask(
            "test",
            {
                "condition_function": failing_condition,
            },
        )

        preview = task.preview_condition()

        assert "error" in preview
        assert "Condition preview failed" in preview["error"]


class TestConditionalTaskStringRepresentation:
    """Test ConditionalTask string representation."""

    def test_string_representation_full(self) -> None:
        """String representation shows all components when present."""
        dependencies = [TaskDependency("data", "source.data", DependencyType.REQUIRED)]

        task = ConditionalTask(
            "test_task",
            {
                "condition_function": lambda: True,
                "action_function": lambda: {"result": "action"},
                "else_function": lambda: {"result": "else"},
            },
            dependencies,
        )

        str_repr = str(task)

        assert "ConditionalTask" in str_repr
        assert "name='test_task'" in str_repr
        assert "dependencies=1" in str_repr
        assert "condition=True" in str_repr
        assert "action=True" in str_repr
        assert "else=True" in str_repr

    def test_string_representation_minimal(self) -> None:
        """String representation shows correct values when components missing."""
        task = ConditionalTask("minimal", {})

        str_repr = str(task)

        assert "name='minimal'" in str_repr
        assert "dependencies=0" in str_repr
        assert "condition=False" in str_repr
        assert "action=False" in str_repr
        assert "else=False" in str_repr


class TestUtilityConditionFunctions:
    """Test utility condition functions provided by ConditionalTask module."""

    def test_threshold_condition_greater_equal(self) -> None:
        """threshold_condition works with >= operator."""
        condition = threshold_condition(10.0, ">=")

        assert condition(10.0) is True
        assert condition(15.5) is True
        assert condition(9.9) is False

    def test_threshold_condition_all_operators(self) -> None:
        """threshold_condition works with all comparison operators."""
        operators_and_values = [
            (">", 10.0, [(11.0, True), (10.0, False), (9.0, False)]),
            ("<=", 10.0, [(9.0, True), (10.0, True), (11.0, False)]),
            ("<", 10.0, [(9.0, True), (10.0, False), (11.0, False)]),
            ("==", 10.0, [(10.0, True), (10.1, False), (9.9, False)]),
            ("!=", 10.0, [(10.1, True), (9.9, True), (10.0, False)]),
        ]

        for operator, threshold, test_cases in operators_and_values:
            condition = threshold_condition(threshold, operator)
            for value, expected in test_cases:
                assert condition(value) == expected, (
                    f"threshold_condition({threshold}, '{operator}') failed for {value}"
                )

    def test_threshold_condition_invalid_operator(self) -> None:
        """threshold_condition raises error for invalid operator."""
        condition = threshold_condition(10.0, "invalid")

        with pytest.raises(ValueError, match="Unknown operator"):
            condition(5.0)

    def test_threshold_condition_invalid_value_type(self) -> None:
        """threshold_condition raises error for non-numeric values."""
        condition = threshold_condition(10.0)

        with pytest.raises(ValueError, match="Value must be numeric"):
            condition("not_numeric")  # pyright: ignore

    def test_contains_condition_case_sensitive(self) -> None:
        """contains_condition works with case sensitivity."""
        condition = contains_condition("Python", case_sensitive=True)

        assert condition("I love Python programming") is True
        assert condition("I love python programming") is False
        assert condition("Java is great") is False

    def test_contains_condition_case_insensitive(self) -> None:
        """contains_condition works case insensitive by default."""
        condition = contains_condition("Python", case_sensitive=False)

        assert condition("I love Python programming") is True
        assert condition("I love python programming") is True
        assert condition("I LOVE PYTHON PROGRAMMING") is True
        assert condition("Java is great") is False

    def test_contains_condition_invalid_input_type(self) -> None:
        """contains_condition raises error for non-string input."""
        condition = contains_condition("test")

        with pytest.raises(ValueError, match="Input must be a string"):
            condition(123)  # pyright: ignore

    def test_list_size_condition_min_only(self) -> None:
        """list_size_condition works with minimum size only."""
        condition = list_size_condition(min_size=3)

        assert condition([1, 2, 3]) is True
        assert condition([1, 2, 3, 4, 5]) is True
        assert condition([1, 2]) is False

    def test_list_size_condition_max_only(self) -> None:
        """list_size_condition works with maximum size only."""
        condition = list_size_condition(max_size=3)

        assert condition([]) is True
        assert condition([1, 2, 3]) is True
        assert condition([1, 2, 3, 4]) is False

    def test_list_size_condition_min_and_max(self) -> None:
        """list_size_condition works with both min and max size."""
        condition = list_size_condition(min_size=2, max_size=4)

        assert condition([1]) is False
        assert condition([1, 2]) is True
        assert condition([1, 2, 3, 4]) is True
        assert condition([1, 2, 3, 4, 5]) is False

    def test_list_size_condition_invalid_input_type(self) -> None:
        """list_size_condition raises error for non-list input."""
        condition = list_size_condition(min_size=1)

        with pytest.raises(ValueError, match="Input must be a list"):
            condition("not_a_list")  # pyright: ignore

    def test_success_rate_condition_basic_fields(self) -> None:
        """success_rate_condition works with basic success/failed fields."""
        condition = success_rate_condition(0.8)  # 80% success rate

        # 4/5 = 80% success rate
        assert condition({"successful_fetches": 4, "failed_fetches": 1}) is True
        # 3/5 = 60% success rate
        assert condition({"successful_fetches": 3, "failed_fetches": 2}) is False

    def test_success_rate_condition_alternative_fields(self) -> None:
        """success_rate_condition works with alternative field names."""
        condition = success_rate_condition(0.7)

        # Test with 'successful'/'failed' fields
        assert condition({"successful": 7, "failed": 3}) is True  # 70%
        assert condition({"successful": 6, "failed": 4}) is False  # 60%

    def test_success_rate_condition_with_total_field(self) -> None:
        """success_rate_condition works when total is explicitly provided."""
        condition = success_rate_condition(0.75)

        assert condition({"successful_fetches": 6, "total_urls": 8}) is True  # 75%
        assert condition({"successful_fetches": 5, "total_urls": 8}) is False  # 62.5%

    def test_success_rate_condition_zero_total(self) -> None:
        """success_rate_condition returns False when total is zero."""
        condition = success_rate_condition(0.5)

        assert condition({"successful_fetches": 0, "failed_fetches": 0}) is False

    def test_success_rate_condition_invalid_input_type(self) -> None:
        """success_rate_condition raises error for non-dict input."""
        condition = success_rate_condition(0.5)

        with pytest.raises(ValueError, match="Input must be a dictionary"):
            condition("not_a_dict")  # pyright: ignore

    def test_quality_gate_condition_default_score_field(self) -> None:
        """quality_gate_condition works with default 'score' field."""
        condition = quality_gate_condition(7.5)

        assert condition({"score": 8.0}) is True
        assert condition({"score": 7.5}) is True
        assert condition({"score": 7.0}) is False

    def test_quality_gate_condition_custom_score_field(self) -> None:
        """quality_gate_condition works with custom score field."""
        condition = quality_gate_condition(85.0, score_field="quality_score")

        assert condition({"quality_score": 90.0}) is True
        assert condition({"quality_score": 80.0}) is False

    def test_quality_gate_condition_missing_score_field(self) -> None:
        """quality_gate_condition raises error when score field missing."""
        condition = quality_gate_condition(7.0, score_field="missing_score")

        with pytest.raises(ValueError, match="Score field 'missing_score' not found"):
            condition({"other_field": 8.0})

    def test_quality_gate_condition_invalid_score_type(self) -> None:
        """quality_gate_condition raises error for non-numeric score."""
        condition = quality_gate_condition(7.0)

        with pytest.raises(ValueError, match="Score must be numeric"):
            condition({"score": "not_numeric"})

    def test_quality_gate_condition_invalid_input_type(self) -> None:
        """quality_gate_condition raises error for non-dict input."""
        condition = quality_gate_condition(7.0)

        with pytest.raises(ValueError, match="Input must be a dictionary"):
            condition("not_a_dict")  # pyright: ignore


class TestUtilityActionFunctions:
    """Test utility action functions provided by ConditionalTask module."""

    @patch("logging.getLogger")
    def test_log_action_info_level(self, mock_get_logger: Mock) -> None:
        """log_action creates action that logs at info level."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        action = log_action("Test message", "info")
        result = action()

        mock_logger.info.assert_called_once_with("Test message")
        assert result == {"action": "log", "message": "Test message", "level": "info"}

    @patch("logging.getLogger")
    def test_log_action_warning_level(self, mock_get_logger: Mock) -> None:
        """log_action creates action that logs at warning level."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        action = log_action("Warning message", "warning")
        result = action()

        mock_logger.warning.assert_called_once_with("Warning message")
        assert result["level"] == "warning"

    @patch("logging.getLogger")
    def test_log_action_error_level(self, mock_get_logger: Mock) -> None:
        """log_action creates action that logs at error level."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        action = log_action("Error message", "error")
        result = action()

        mock_logger.error.assert_called_once_with("Error message")
        assert result["level"] == "error"

    def test_increment_counter_action_new_counter(self) -> None:
        """increment_counter_action creates new counter starting at 1."""
        context = TaskContext()
        action = increment_counter_action("test_counter")

        result = action(context)

        assert result["action"] == "increment"
        assert result["counter"] == "test_counter"
        assert result["new_value"] == 1

        # Verify counter was stored in context
        stored_result = context.get_result("test_counter")
        assert stored_result is not None
        assert stored_result["value"] == 1

    def test_increment_counter_action_existing_counter(self) -> None:
        """increment_counter_action increments existing counter."""
        context = TaskContext()
        context.store_result("test_counter", {"value": 5})

        action = increment_counter_action("test_counter")
        result = action(context)

        assert result["new_value"] == 6

        # Verify counter was updated in context
        stored_result = context.get_result("test_counter")
        assert stored_result is not None
        assert stored_result["value"] == 6

    def test_increment_counter_action_missing_counter(self) -> None:
        """increment_counter_action handles missing counter gracefully."""
        context = TaskContext()
        action = increment_counter_action("missing_counter")

        result = action(context)

        assert result["new_value"] == 1  # Starts from 0 + 1

    def test_set_flag_action_default_value(self) -> None:
        """set_flag_action sets flag to True by default."""
        action = set_flag_action("test_flag")
        result = action()

        assert result["action"] == "set_flag"
        assert result["flag"] == "test_flag"
        assert result["value"] is True

    def test_set_flag_action_custom_value(self) -> None:
        """set_flag_action sets flag to custom value."""
        action = set_flag_action("custom_flag", "custom_value")
        result = action()

        assert result["flag"] == "custom_flag"
        assert result["value"] == "custom_value"

    def test_set_flag_action_complex_value(self) -> None:
        """set_flag_action works with complex values."""
        complex_value = {"status": "ready", "data": [1, 2, 3]}
        action = set_flag_action("complex_flag", complex_value)
        result = action()

        assert result["value"] == complex_value


class TestConditionalTaskIntegration:
    """Test ConditionalTask integration with framework components."""

    def test_conditional_task_with_dependencies_and_context(self) -> None:
        """ConditionalTask works correctly with dependencies and context."""
        context = TaskContext()
        context.store_result(
            "validation",
            {
                "data": {
                    "success_rate": 0.85,
                    "total_requests": 100,
                    "successful_requests": 85,
                }
            },
        )

        # Create condition using success rate
        def quality_check(validation_data: Dict[str, Any]) -> bool:
            return bool(validation_data["success_rate"] >= 0.8)

        # Create action using validation data
        def proceed_action(validation_data: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "decision": "proceed",
                "total_requests": validation_data["total_requests"],
            }

        dependencies = [
            TaskDependency(
                "validation_data", "validation.data", DependencyType.REQUIRED
            )
        ]

        task = ConditionalTask(
            "quality_gate",
            {
                "condition_function": quality_check,
                "condition_inputs": ["validation_data"],
                "action_function": proceed_action,
                "action_inputs": ["validation_data"],
            },
            dependencies,
        )

        # Simulate dependency resolution that PipelineAgent would do
        from aipype import DependencyResolver

        resolver = DependencyResolver(context)
        resolved_config = resolver.resolve_dependencies(task)
        task.config.update(resolved_config)

        task.set_context(context)
        result = task.run()

        assert result.data["condition_result"] is True
        assert result.data["executed"] is True
        assert result.data["action_result"]["decision"] == "proceed"
        assert result.data["action_result"]["total_requests"] == 100

    def test_conditional_task_logging_integration(self) -> None:
        """ConditionalTask logging works correctly."""
        with patch("aipype.base_task.setup_logger") as mock_setup_logger:
            mock_logger = Mock()
            mock_setup_logger.return_value = mock_logger

            task = ConditionalTask(
                "test_logging",
                {
                    "condition_function": lambda: True,
                    "action_function": lambda: {"result": "success"},
                },
            )

            task.run()

            # Verify logger was set up for the task
            mock_setup_logger.assert_called_once_with("task.test_logging")

            # Verify info log was called for successful condition
            mock_logger.info.assert_called_with(
                "Condition true for task 'test_logging' - action executed"
            )

    def test_conditional_task_with_utility_functions_integration(self) -> None:
        """ConditionalTask works with utility condition and action functions."""
        context = TaskContext()

        # Use utility functions for complete integration test
        condition = success_rate_condition(0.75)
        action = log_action("Quality threshold met", "info")

        task = ConditionalTask(
            "utility_integration",
            {
                "condition_function": condition,
                "condition_inputs": ["results"],
                "action_function": action,
                "results": {
                    "successful_fetches": 8,
                    "failed_fetches": 2,
                },  # 80% success
            },
        )

        task.set_context(context)

        with patch("logging.getLogger"):
            result = task.run()

        assert result.data["condition_result"] is True
        assert result.data["executed"] is True
        assert result.data["action_result"]["message"] == "Quality threshold met"
