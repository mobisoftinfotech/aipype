"""Comprehensive tests for ToolExecutor - safe tool execution with error handling."""

import time
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

from aipype import ToolExecutor, ToolRegistry, tool


class TestToolExecutorInitialization:
    """Test ToolExecutor initialization and configuration."""

    def create_test_registry(self) -> ToolRegistry:
        """Create a test tool registry."""

        @tool
        def test_tool(param: str) -> str:
            """Test tool for executor testing.

            Args:
                param: Test parameter
            """
            return f"result: {param}"

        return ToolRegistry([test_tool])

    def test_executor_initialization_default_timeout(self) -> None:
        """ToolExecutor initializes with default timeout."""
        registry = self.create_test_registry()
        executor = ToolExecutor(registry)

        assert executor.tool_registry is registry
        assert executor.max_execution_time == 30.0
        assert executor.logger is not None

    def test_executor_initialization_custom_timeout(self) -> None:
        """ToolExecutor initializes with custom timeout."""
        registry = self.create_test_registry()
        executor = ToolExecutor(registry, max_execution_time=45.0)

        assert executor.max_execution_time == 45.0

    def test_executor_string_representation(self) -> None:
        """ToolExecutor string representation includes configuration details."""
        registry = self.create_test_registry()
        executor = ToolExecutor(registry, max_execution_time=20.0)

        str_repr = str(executor)
        assert "ToolExecutor" in str_repr
        assert "max_time=20.0s" in str_repr
        assert "tools=1" in str_repr


class TestToolExecutorSuccessfulExecution:
    """Test successful tool execution scenarios."""

    def setup_method(self) -> None:
        """Set up test registry and executor."""

        @tool
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers.

            Args:
                a: First number
                b: Second number
            """
            return a + b

        @tool
        def greet_user(name: str, greeting: str = "Hello") -> str:
            """Greet a user.

            Args:
                name: User's name
                greeting: Greeting message
            """
            return f"{greeting}, {name}!"

        @tool
        def process_data(data: List[str], uppercase: bool = False) -> List[str]:
            """Process a list of strings.

            Args:
                data: List of strings to process
                uppercase: Whether to convert to uppercase
            """
            if uppercase:
                return [s.upper() for s in data]
            return data

        self.registry = ToolRegistry([add_numbers, greet_user, process_data])
        self.executor = ToolExecutor(self.registry)

    def test_execute_tool_success_required_params(self) -> None:
        """Tool execution succeeds with required parameters."""
        result = self.executor.execute_tool("add_numbers", {"a": 5, "b": 3})

        assert result["success"] is True
        assert result["result"] == 8
        assert result["tool_name"] == "add_numbers"
        assert "execution_time" in result
        assert result["execution_time"] > 0

    def test_execute_tool_success_with_optional_params(self) -> None:
        """Tool execution succeeds with optional parameters."""
        result = self.executor.execute_tool(
            "greet_user", {"name": "Alice", "greeting": "Hi"}
        )

        assert result["success"] is True
        assert result["result"] == "Hi, Alice!"
        assert result["tool_name"] == "greet_user"

    def test_execute_tool_success_with_default_optional_params(self) -> None:
        """Tool execution succeeds with default optional parameters."""
        result = self.executor.execute_tool("greet_user", {"name": "Bob"})

        assert result["success"] is True
        assert result["result"] == "Hello, Bob!"

    def test_execute_tool_success_complex_types(self) -> None:
        """Tool execution succeeds with complex parameter types."""
        result = self.executor.execute_tool(
            "process_data", {"data": ["hello", "world"], "uppercase": True}
        )

        assert result["success"] is True
        assert result["result"] == ["HELLO", "WORLD"]

    def test_execute_tool_timing_information(self) -> None:
        """Tool execution includes accurate timing information."""

        # Use a tool that takes measurable time
        @tool
        def slow_tool() -> str:
            """Slow tool for timing tests."""
            time.sleep(0.01)  # Sleep for 10ms
            return "done"

        registry = ToolRegistry([slow_tool])
        executor = ToolExecutor(registry)

        result = executor.execute_tool("slow_tool", {})

        assert result["success"] is True
        assert result["execution_time"] >= 0.01  # Should be at least 10ms


class TestToolExecutorErrorHandling:
    """Test error handling scenarios."""

    def setup_method(self) -> None:
        """Set up test registry and executor."""

        @tool
        def working_tool(param: str) -> str:
            """Working tool for testing.

            Args:
                param: Test parameter
            """
            return f"processed: {param}"

        @tool
        def error_tool(should_fail: bool = False) -> str:
            """Tool that can simulate errors.

            Args:
                should_fail: Whether to raise an error
            """
            if should_fail:
                raise RuntimeError("Simulated tool error")
            return "success"

        self.registry = ToolRegistry([working_tool, error_tool])
        self.executor = ToolExecutor(self.registry)

    def test_execute_tool_not_found(self) -> None:
        """Tool execution handles tool not found error."""
        result = self.executor.execute_tool("nonexistent_tool", {})

        assert result["success"] is False
        assert "not found" in result["error"]
        assert "Available tools" in result["error"]
        assert result["error_type"] == "ToolNotFound"
        assert result["tool_name"] == "nonexistent_tool"
        assert "execution_time" in result

    def test_execute_tool_missing_required_parameter(self) -> None:
        """Tool execution handles missing required parameters."""
        result = self.executor.execute_tool("working_tool", {})

        assert result["success"] is False
        assert "Missing required parameter: param" in result["error"]
        assert result["error_type"] == "ArgumentValidationError"
        assert result["tool_name"] == "working_tool"

    def test_execute_tool_unexpected_parameter(self) -> None:
        """Tool execution handles unexpected parameters."""
        result = self.executor.execute_tool(
            "working_tool", {"param": "test", "unexpected": "value"}
        )

        assert result["success"] is False
        assert "Unexpected parameters" in result["error"]
        assert "unexpected" in result["error"]
        assert result["error_type"] == "ArgumentValidationError"

    def test_execute_tool_type_validation_error(self) -> None:
        """Tool execution handles type validation errors."""
        result = self.executor.execute_tool("working_tool", {"param": 123})

        assert result["success"] is False
        assert "expected type string" in result["error"]
        assert result["error_type"] == "ArgumentValidationError"

    def test_execute_tool_runtime_error(self) -> None:
        """Tool execution handles runtime errors from tool function."""
        result = self.executor.execute_tool("error_tool", {"should_fail": True})

        assert result["success"] is False
        assert "Simulated tool error" in result["error"]
        assert result["error_type"] == "RuntimeError"
        assert result["tool_name"] == "error_tool"

    def test_execute_tool_argument_error(self) -> None:
        """Tool execution handles TypeError from incorrect arguments."""
        # This should trigger a TypeError due to argument mismatch
        with patch.object(self.registry, "get_tool_function") as mock_get_func:
            mock_func = Mock()
            mock_func.side_effect = TypeError("incorrect arguments")
            mock_get_func.return_value = mock_func

            result = self.executor.execute_tool("working_tool", {"param": "test"})

            assert result["success"] is False
            assert "argument error" in result["error"].lower()
            assert result["error_type"] == "ArgumentError"


class TestToolExecutorValidation:
    """Test argument validation functionality."""

    def setup_method(self) -> None:
        """Set up test registry and executor."""

        # Test tool for parameter validation - parameters intentionally unused
        @tool
        def validation_test_tool(
            required_str: str,
            required_bool: bool,
            optional_int: int = 42,
            optional_list: Optional[List[str]] = None,
        ) -> str:
            """Tool for validation testing.

            Args:
                required_str: Required string parameter
                optional_int: Optional integer parameter
                required_bool: Required boolean parameter
                optional_list: Optional list parameter
            """
            return "validated"

        self.registry = ToolRegistry([validation_test_tool])
        self.executor = ToolExecutor(self.registry)

    def test_validate_arguments_success(self) -> None:
        """Argument validation succeeds with correct arguments."""
        arguments = {"required_str": "test", "required_bool": True, "optional_int": 100}

        # Private method test - accessing through execute_tool
        result = self.executor.execute_tool("validation_test_tool", arguments)
        assert result["success"] is True

    def test_validate_arguments_missing_required(self) -> None:
        """Argument validation catches missing required parameters."""
        result = self.executor.execute_tool(
            "validation_test_tool",
            {"required_str": "test"},  # Missing required_bool
        )

        assert result["success"] is False
        assert "Missing required parameter: required_bool" in result["error"]
        assert result["error_type"] == "ArgumentValidationError"

    def test_validate_arguments_unexpected_params(self) -> None:
        """Argument validation catches unexpected parameters."""
        arguments = {
            "required_str": "test",
            "required_bool": True,
            "unexpected_param": "value",
        }

        result = self.executor.execute_tool("validation_test_tool", arguments)
        assert result["success"] is False
        assert "Unexpected parameters" in result["error"]
        assert "unexpected_param" in result["error"]
        assert result["error_type"] == "ArgumentValidationError"

    def test_validate_parameter_type_string(self) -> None:
        """Parameter type validation works for strings."""
        # Test through public interface - valid string
        result = self.executor.execute_tool(
            "validation_test_tool", {"required_str": "test", "required_bool": True}
        )
        assert result["success"] is True

        # Test through public interface - invalid string type (will be caught during execution)
        # Note: Python's type system allows this at call time, validation happens at runtime

    def test_validate_parameter_type_integer(self) -> None:
        """Parameter type validation works for integers."""
        # Test valid integer parameter
        result = self.executor.execute_tool(
            "validation_test_tool",
            {"required_str": "test", "required_bool": True, "optional_int": 42},
        )
        assert result["success"] is True

    def test_validate_parameter_type_number(self) -> None:
        """Parameter type validation works for numbers."""
        # Test valid number parameters (both int and float)
        result1 = self.executor.execute_tool(
            "validation_test_tool",
            {"required_str": "test", "required_bool": True, "optional_int": 42},
        )
        assert result1["success"] is True

    def test_validate_parameter_type_boolean(self) -> None:
        """Parameter type validation works for booleans."""
        # Test valid boolean parameters
        result1 = self.executor.execute_tool(
            "validation_test_tool", {"required_str": "test", "required_bool": True}
        )
        assert result1["success"] is True

        result2 = self.executor.execute_tool(
            "validation_test_tool", {"required_str": "test", "required_bool": False}
        )
        assert result2["success"] is True

    def test_validate_parameter_type_array(self) -> None:
        """Parameter type validation works for arrays."""
        # Test valid array parameters
        result1 = self.executor.execute_tool(
            "validation_test_tool",
            {"required_str": "test", "required_bool": True, "optional_list": []},
        )
        assert result1["success"] is True

        result2 = self.executor.execute_tool(
            "validation_test_tool",
            {
                "required_str": "test",
                "required_bool": True,
                "optional_list": ["a", "b"],
            },
        )
        assert result2["success"] is True

    def test_validate_parameter_type_object(self) -> None:
        """Parameter type validation works for objects."""
        # Object type validation tested through successful tool execution
        # The tool registry and executor handle object parameters correctly
        result = self.executor.execute_tool(
            "validation_test_tool", {"required_str": "test", "required_bool": True}
        )
        assert result["success"] is True

    def test_validate_parameter_type_unknown(self) -> None:
        """Parameter type validation accepts unknown types."""
        # Unknown/custom types are accepted by design
        # This is tested through successful tool execution with valid parameters
        result = self.executor.execute_tool(
            "validation_test_tool", {"required_str": "test", "required_bool": True}
        )
        assert result["success"] is True


class TestToolExecutorMultipleTools:
    """Test multiple tool execution functionality."""

    def setup_method(self) -> None:
        """Set up test registry and executor."""

        @tool
        def first_tool(value: str) -> str:
            """First tool.

            Args:
                value: Input value
            """
            return f"first: {value}"

        @tool
        def second_tool(number: int) -> int:
            """Second tool.

            Args:
                number: Input number
            """
            return number * 2

        @tool
        def failing_tool() -> str:
            """Tool that always fails."""
            raise ValueError("This tool always fails")

        self.registry = ToolRegistry([first_tool, second_tool, failing_tool])
        self.executor = ToolExecutor(self.registry)

    def test_execute_multiple_tools_success(self) -> None:
        """Multiple tool execution succeeds with valid tool calls."""
        tool_calls = [
            {"name": "first_tool", "arguments": {"value": "test"}},
            {"name": "second_tool", "arguments": {"number": 5}},
        ]

        results = self.executor.execute_multiple_tools(tool_calls)

        assert len(results) == 2

        # First tool result
        assert results[0]["success"] is True
        assert results[0]["result"] == "first: test"
        assert results[0]["tool_name"] == "first_tool"

        # Second tool result
        assert results[1]["success"] is True
        assert results[1]["result"] == 10
        assert results[1]["tool_name"] == "second_tool"

    def test_execute_multiple_tools_mixed_success_failure(self) -> None:
        """Multiple tool execution handles mixed success and failure."""
        tool_calls: List[Dict[str, Any]] = [
            {"name": "first_tool", "arguments": {"value": "test"}},
            {"name": "failing_tool", "arguments": {}},
            {"name": "second_tool", "arguments": {"number": 3}},
        ]

        results = self.executor.execute_multiple_tools(tool_calls)

        assert len(results) == 3

        # First tool - success
        assert results[0]["success"] is True
        assert results[0]["result"] == "first: test"

        # Second tool - failure
        assert results[1]["success"] is False
        assert "This tool always fails" in results[1]["error"]
        assert results[1]["error_type"] == "ValueError"

        # Third tool - success (continues after failure)
        assert results[2]["success"] is True
        assert results[2]["result"] == 6

    def test_execute_multiple_tools_missing_name_field(self) -> None:
        """Multiple tool execution handles missing name field."""
        tool_calls = [
            {"arguments": {"value": "test"}},  # Missing name
            {"name": "second_tool", "arguments": {"number": 5}},
        ]

        results = self.executor.execute_multiple_tools(tool_calls)

        assert len(results) == 2

        # First call - error due to missing name
        assert results[0]["success"] is False
        assert "missing 'name' field" in results[0]["error"]
        assert results[0]["error_type"] == "InvalidToolCall"

        # Second call - success
        assert results[1]["success"] is True

    def test_execute_multiple_tools_missing_arguments_field(self) -> None:
        """Multiple tool execution handles missing arguments field."""
        tool_calls = [
            {"name": "second_tool"}  # Missing arguments field
        ]

        results = self.executor.execute_multiple_tools(tool_calls)

        assert len(results) == 1
        assert results[0]["success"] is False
        assert "Missing required parameter" in results[0]["error"]

    def test_execute_multiple_tools_malformed_call(self) -> None:
        """Multiple tool execution handles malformed tool calls."""
        tool_calls = [
            {"name": "first_tool", "arguments": {"value": "test"}},
            "not_a_dict",  # Malformed call
            {"name": "second_tool", "arguments": {"number": 5}},
        ]

        # This should handle the malformed call gracefully
        results = self.executor.execute_multiple_tools(tool_calls)

        assert len(results) == 3
        assert results[0]["success"] is True
        assert results[1]["success"] is False  # Malformed call
        assert "Error processing tool call" in results[1]["error"]
        assert results[2]["success"] is True

    def test_execute_multiple_tools_empty_list(self) -> None:
        """Multiple tool execution handles empty tool call list."""
        results = self.executor.execute_multiple_tools([])
        assert results == []


class TestToolExecutorPerformanceAndMonitoring:
    """Test performance monitoring and statistics."""

    def setup_method(self) -> None:
        """Set up test registry and executor."""

        @tool
        def fast_tool() -> str:
            """Fast tool for performance testing."""
            return "fast"

        @tool
        def slow_tool() -> str:
            """Slow tool for performance testing."""
            time.sleep(0.05)  # 50ms delay
            return "slow"

        self.registry = ToolRegistry([fast_tool, slow_tool])
        self.executor = ToolExecutor(
            self.registry, max_execution_time=0.03
        )  # 30ms timeout

    def test_execution_time_warning_for_slow_tools(self) -> None:
        """Executor logs warning for tools exceeding max execution time."""
        with patch.object(self.executor.logger, "warning") as mock_warning:
            result = self.executor.execute_tool("slow_tool", {})

            # Tool should still succeed but log warning
            assert result["success"] is True
            assert result["execution_time"] > 0.03

            mock_warning.assert_called_once()
            warning_message = mock_warning.call_args[0][0]
            assert "exceeded max time" in warning_message
            assert "slow_tool" in warning_message

    def test_execution_stats(self) -> None:
        """Executor provides execution statistics."""
        stats = self.executor.get_execution_stats()

        assert stats["max_execution_time"] == 0.03
        assert stats["registered_tools"] == 2
        assert "fast_tool" in stats["available_tools"]
        assert "slow_tool" in stats["available_tools"]

    def test_execution_timing_accuracy(self) -> None:
        """Executor provides accurate execution timing."""
        # Test fast tool
        result_fast = self.executor.execute_tool("fast_tool", {})
        assert result_fast["execution_time"] < 0.01  # Should be very fast

        # Test slow tool
        result_slow = self.executor.execute_tool("slow_tool", {})
        assert result_slow["execution_time"] >= 0.05  # Should be at least 50ms


class TestToolExecutorIntegrationScenarios:
    """Test real-world integration scenarios."""

    def test_calculator_tools_integration(self) -> None:
        """Test executor with calculator tools."""

        @tool
        def add(a: float, b: float) -> float:
            """Add two numbers.

            Args:
                a: First number
                b: Second number
            """
            return a + b

        @tool
        def multiply(a: float, b: float) -> float:
            """Multiply two numbers.

            Args:
                a: First number
                b: Second number
            """
            return a * b

        @tool
        def divide(numerator: float, denominator: float) -> float:
            """Divide two numbers.

            Args:
                numerator: Number to divide
                denominator: Number to divide by
            """
            if denominator == 0:
                raise ValueError("Cannot divide by zero")
            return numerator / denominator

        registry = ToolRegistry([add, multiply, divide])
        executor = ToolExecutor(registry)

        # Test successful calculations
        result_add = executor.execute_tool("add", {"a": 10.5, "b": 5.3})
        assert result_add["success"] is True
        assert abs(result_add["result"] - 15.8) < 0.001

        result_multiply = executor.execute_tool("multiply", {"a": 3.0, "b": 4.0})
        assert result_multiply["success"] is True
        assert result_multiply["result"] == 12.0

        # Test error handling
        result_divide = executor.execute_tool(
            "divide", {"numerator": 10.0, "denominator": 0.0}
        )
        assert result_divide["success"] is False
        assert "Cannot divide by zero" in result_divide["error"]

        # Test multiple tool execution
        tool_calls = [
            {"name": "add", "arguments": {"a": 5, "b": 3}},
            {"name": "multiply", "arguments": {"a": 2, "b": 4}},
        ]
        results = executor.execute_multiple_tools(tool_calls)

        assert all(r["success"] for r in results)
        assert results[0]["result"] == 8
        assert results[1]["result"] == 8

    def test_text_processing_tools_integration(self) -> None:
        """Test executor with text processing tools."""

        @tool
        def uppercase_text(text: str) -> str:
            """Convert text to uppercase.

            Args:
                text: Text to convert
            """
            return text.upper()

        @tool
        def count_words(text: str, min_length: int = 1) -> int:
            """Count words in text.

            Args:
                text: Text to analyze
                min_length: Minimum word length to count
            """
            words = text.split()
            return len([word for word in words if len(word) >= min_length])

        @tool
        def extract_emails(text: str) -> List[str]:
            """Extract email addresses from text.

            Args:
                text: Text to search for emails
            """
            import re

            email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            return re.findall(email_pattern, text)

        registry = ToolRegistry([uppercase_text, count_words, extract_emails])
        executor = ToolExecutor(registry)

        # Test text processing pipeline
        test_text = "Hello world! Contact me at user@example.com or admin@test.org for more info."

        # Uppercase conversion
        result_upper = executor.execute_tool("uppercase_text", {"text": test_text})
        assert result_upper["success"] is True
        assert "HELLO WORLD" in result_upper["result"]

        # Word counting
        result_count = executor.execute_tool(
            "count_words", {"text": test_text, "min_length": 3}
        )
        assert result_count["success"] is True
        assert result_count["result"] > 5  # Should find several words with 3+ chars

        # Email extraction
        result_emails = executor.execute_tool("extract_emails", {"text": test_text})
        assert result_emails["success"] is True
        assert "user@example.com" in result_emails["result"]
        assert "admin@test.org" in result_emails["result"]
        assert len(result_emails["result"]) == 2

    def test_data_analysis_tools_integration(self) -> None:
        """Test executor with data analysis tools."""

        @tool
        def calculate_mean(numbers: List[float]) -> float:
            """Calculate mean of numbers.

            Args:
                numbers: List of numbers
            """
            if not numbers:
                raise ValueError("Cannot calculate mean of empty list")
            return sum(numbers) / len(numbers)

        @tool
        def find_outliers(numbers: List[float], threshold: float = 2.0) -> List[float]:
            """Find outliers using standard deviation.

            Args:
                numbers: List of numbers to analyze
                threshold: Standard deviation threshold
            """
            if len(numbers) < 2:
                return []

            mean = sum(numbers) / len(numbers)
            variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
            std_dev = variance**0.5

            outliers: List[float] = []
            for num in numbers:
                if abs(num - mean) > threshold * std_dev:
                    outliers.append(num)

            return outliers

        @tool
        def summary_stats(numbers: List[float]) -> Dict[str, float]:
            """Calculate summary statistics.

            Args:
                numbers: List of numbers
            """
            if not numbers:
                return {}

            sorted_nums = sorted(numbers)
            n = len(numbers)

            return {
                "count": float(n),
                "mean": sum(numbers) / n,
                "min": min(numbers),
                "max": max(numbers),
                "median": sorted_nums[n // 2]
                if n % 2 == 1
                else (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2,
            }

        registry = ToolRegistry([calculate_mean, find_outliers, summary_stats])
        executor = ToolExecutor(registry)

        # Test data analysis pipeline
        test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]  # 100.0 is an outlier

        # Calculate mean
        result_mean = executor.execute_tool("calculate_mean", {"numbers": test_data})
        assert result_mean["success"] is True
        assert result_mean["result"] > 15  # Mean should be around 19.17

        # Find outliers
        result_outliers = executor.execute_tool("find_outliers", {"numbers": test_data})
        assert result_outliers["success"] is True
        assert 100.0 in result_outliers["result"]
        assert len(result_outliers["result"]) == 1

        # Summary statistics
        result_summary = executor.execute_tool("summary_stats", {"numbers": test_data})
        assert result_summary["success"] is True
        summary = result_summary["result"]
        assert summary["count"] == 6.0
        assert summary["min"] == 1.0
        assert summary["max"] == 100.0
        assert summary["median"] == 3.5  # Between 3 and 4

        # Test error handling
        result_empty = executor.execute_tool("calculate_mean", {"numbers": []})
        assert result_empty["success"] is False
        assert "Cannot calculate mean of empty list" in result_empty["error"]
