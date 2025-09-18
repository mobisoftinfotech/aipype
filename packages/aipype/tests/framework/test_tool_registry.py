"""Comprehensive tests for ToolRegistry - tool management and schema generation."""

import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

# Tool decorator creates functions with dynamic type signatures for LLM tool calling
from aipype import ToolRegistry, tool, ToolMetadata, ToolSchemaGenerator


class TestToolRegistryInitialization:
    """Test ToolRegistry initialization and setup."""

    def create_test_tools(self) -> List[Any]:
        """Create a set of test tools for registry testing."""

        @tool
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers together.

            Args:
                a: First number
                b: Second number
            """
            return a + b

        @tool
        def greet_user(name: str, greeting: str = "Hello") -> str:
            """Greet a user with a custom message.

            Args:
                name: User's name
                greeting: Greeting message to use
            """
            return f"{greeting}, {name}!"

        @tool
        def calculate_stats(
            data: List[float], include_median: bool = False
        ) -> Dict[str, float]:
            """Calculate statistics for a list of numbers.

            Args:
                data: List of numerical values
                include_median: Whether to include median in results
            """
            if not data:
                return {}
            stats = {"mean": sum(data) / len(data), "count": len(data)}
            if include_median:
                sorted_data = sorted(data)
                n = len(sorted_data)
                stats["median"] = (
                    sorted_data[n // 2]
                    if n % 2 == 1
                    else (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
                )
            return stats

        return [add_numbers, greet_user, calculate_stats]

    def test_registry_initialization_success(self) -> None:
        """ToolRegistry initializes successfully with valid decorated functions."""
        tools = self.create_test_tools()
        registry = ToolRegistry(tools)

        assert len(registry.tools) == 3
        assert len(registry.schemas) == 3
        assert registry.get_tool_count() == 3

    def test_registry_initialization_stores_tool_metadata(self) -> None:
        """ToolRegistry stores tool metadata correctly during initialization."""
        tools = self.create_test_tools()
        registry = ToolRegistry(tools)

        assert "add_numbers" in registry.tools
        assert "greet_user" in registry.tools
        assert "calculate_stats" in registry.tools

        # Check metadata is properly stored
        add_metadata = registry.tools["add_numbers"]
        assert isinstance(add_metadata, ToolMetadata)
        assert add_metadata.name == "add_numbers"
        assert "add two numbers" in add_metadata.description.lower()

    def test_registry_initialization_generates_schemas(self) -> None:
        """ToolRegistry generates OpenAI schemas during initialization."""
        tools = self.create_test_tools()
        registry = ToolRegistry(tools)

        schemas = registry.get_tool_schemas()
        assert len(schemas) == 3

        # Check schema structure
        for schema in schemas:
            assert schema["type"] == "function"
            assert "function" in schema
            assert "name" in schema["function"]
            assert "description" in schema["function"]
            assert "parameters" in schema["function"]

    def test_registry_initialization_with_empty_list(self) -> None:
        """ToolRegistry handles empty tool list."""
        registry = ToolRegistry([])

        assert len(registry.tools) == 0
        assert len(registry.schemas) == 0
        assert registry.get_tool_count() == 0

    def test_registry_initialization_rejects_undecorated_function(self) -> None:
        """ToolRegistry rejects functions without @tool decorator."""

        def undecorated_function() -> None:
            """Regular function without @tool decorator."""
            pass

        with pytest.raises(ValueError, match="must be decorated with @tool"):
            ToolRegistry([undecorated_function])

    def test_registry_initialization_rejects_non_callable(self) -> None:
        """ToolRegistry rejects non-callable objects."""
        not_callable = "not a function"

        with pytest.raises(ValueError, match="is not callable"):
            ToolRegistry([not_callable])  # pyright: ignore[reportArgumentType]

    def test_registry_initialization_rejects_duplicate_names(self) -> None:
        """ToolRegistry rejects tools with duplicate names."""

        @tool
        def first_function() -> None:
            """First function with this name."""
            pass

        @tool
        def second_function() -> None:
            """Second function with same name."""
            pass

        # Tool decorator creates dynamic metadata attributes for function calling
        first_function._tool_metadata.name = "duplicate_name"  # pyright: ignore[reportFunctionMemberAccess]
        second_function._tool_metadata.name = "duplicate_name"  # pyright: ignore[reportFunctionMemberAccess]

        with pytest.raises(ValueError, match="Duplicate tool name"):
            ToolRegistry([first_function, second_function])

    def test_registry_initialization_missing_metadata_raises_error(self) -> None:
        """ToolRegistry raises error for functions missing tool metadata."""

        def fake_tool() -> None:
            """Fake tool function."""
            pass

        # Tool decorator dynamically adds type checking attributes
        fake_tool._is_tool = True  # pyright: ignore[reportFunctionMemberAccess]

        with pytest.raises(ValueError, match="missing tool metadata"):
            ToolRegistry([fake_tool])


class TestToolRegistryRetrieval:
    """Test tool retrieval functionality."""

    def setup_method(self) -> None:
        """Set up test registry with sample tools."""

        @tool
        def sample_tool(param: str) -> str:
            """Sample tool for testing.

            Args:
                param: Test parameter
            """
            return f"processed: {param}"

        @tool
        def another_tool(x: int, y: int = 10) -> int:
            """Another tool for testing.

            Args:
                x: First number
                y: Second number with default
            """
            return x + y

        self.registry = ToolRegistry([sample_tool, another_tool])

    def test_get_tool_function_success(self) -> None:
        """get_tool_function returns correct function."""
        # Tool registry returns dynamically typed tool functions
        tool_func = self.registry.get_tool_function("sample_tool")

        # Tool function execution has dynamic return type
        result = tool_func("test")
        assert result == "processed: test"

    def test_get_tool_function_not_found(self) -> None:
        """get_tool_function raises error for missing tool."""
        with pytest.raises(ValueError, match="Tool 'nonexistent' not found"):
            # Registry method has partially unknown return type for error cases
            self.registry.get_tool_function("nonexistent")

    def test_get_tool_metadata_success(self) -> None:
        """get_tool_metadata returns correct metadata."""
        metadata = self.registry.get_tool_metadata("sample_tool")

        assert isinstance(metadata, ToolMetadata)
        assert metadata.name == "sample_tool"
        assert "sample tool for testing" in metadata.description.lower()
        assert "param" in metadata.parameters

    def test_get_tool_metadata_not_found(self) -> None:
        """get_tool_metadata raises error for missing tool."""
        with pytest.raises(ValueError, match="Tool 'missing' not found"):
            self.registry.get_tool_metadata("missing")

    def test_has_tool_existing(self) -> None:
        """has_tool returns True for existing tools."""
        assert self.registry.has_tool("sample_tool") is True
        assert self.registry.has_tool("another_tool") is True

    def test_has_tool_missing(self) -> None:
        """has_tool returns False for missing tools."""
        assert self.registry.has_tool("nonexistent") is False

    def test_list_tool_names(self) -> None:
        """list_tool_names returns all registered tool names."""
        names = self.registry.list_tool_names()

        assert "sample_tool" in names
        assert "another_tool" in names
        assert len(names) == 2

    def test_get_tool_count(self) -> None:
        """get_tool_count returns correct count."""
        assert self.registry.get_tool_count() == 2

    def test_get_tool_schemas_returns_copy(self) -> None:
        """get_tool_schemas returns a copy of schemas."""
        schemas1 = self.registry.get_tool_schemas()
        schemas2 = self.registry.get_tool_schemas()

        # Should be equal but not the same object
        assert schemas1 == schemas2
        assert schemas1 is not schemas2

    def test_get_tool_schemas_structure(self) -> None:
        """get_tool_schemas returns properly structured schemas."""
        schemas = self.registry.get_tool_schemas()

        assert len(schemas) == 2

        # Find sample_tool schema
        sample_schema = next(
            s for s in schemas if s["function"]["name"] == "sample_tool"
        )
        assert sample_schema["type"] == "function"

        function_def = sample_schema["function"]
        assert function_def["description"] == "Sample tool for testing."

        params = function_def["parameters"]
        assert params["type"] == "object"
        assert "param" in params["properties"]
        assert params["required"] == ["param"]


class TestToolRegistryValidation:
    """Test tool validation functionality."""

    def setup_method(self) -> None:
        """Set up test registry with sample tools."""

        @tool
        def working_tool() -> str:
            """Working tool for validation testing."""
            return "success"

        self.registry = ToolRegistry([working_tool])

    def test_validate_all_tools_success(self) -> None:
        """validate_all_tools returns True for all valid tools."""
        validation_results = self.registry.validate_all_tools()

        assert validation_results["working_tool"] is True
        assert all(validation_results.values())

    def test_validate_all_tools_with_broken_tool(self) -> None:
        """validate_all_tools handles broken tools gracefully."""
        # Test metadata corruption for validation testing
        self.registry.tools["working_tool"].func = "not_callable"  # pyright: ignore[reportAttributeAccessIssue]

        validation_results = self.registry.validate_all_tools()
        assert validation_results["working_tool"] is False

    @patch.object(ToolSchemaGenerator, "generate_schema")
    def test_validate_all_tools_schema_generation_failure(
        self, mock_generate_schema: Mock
    ) -> None:
        """validate_all_tools handles schema generation failures."""
        mock_generate_schema.side_effect = Exception("Schema generation failed")

        validation_results = self.registry.validate_all_tools()
        assert validation_results["working_tool"] is False

    def test_validate_all_tools_logs_failures(self) -> None:
        """validate_all_tools logs validation failures."""
        # Test metadata corruption for validation logging
        self.registry.tools["working_tool"].func = None  # pyright: ignore[reportAttributeAccessIssue]

        with patch.object(self.registry.logger, "warning") as mock_warning:
            validation_results = self.registry.validate_all_tools()

            assert validation_results["working_tool"] is False
            mock_warning.assert_called_once()
            assert "failed validation" in mock_warning.call_args[0][0]


class TestToolRegistryContextGeneration:
    """Test tool context generation for system prompts."""

    def create_diverse_tools(self) -> List[Any]:
        """Create tools with different parameter patterns for context testing."""

        @tool
        def simple_tool() -> str:
            """Simple tool with no parameters."""
            return "simple"

        @tool
        def param_tool(required: str, optional: int = 42) -> str:
            """Tool with required and optional parameters.

            Args:
                required: A required string parameter
                optional: An optional integer parameter
            """
            return f"{required}-{optional}"

        @tool
        def complex_tool(
            data: List[Dict[str, Any]],
            filter_key: Optional[str] = None,
            sort_ascending: bool = True,
        ) -> Dict[str, Any]:
            """Tool with complex parameter types.

            Args:
                data: List of data objects to process
                filter_key: Optional key to filter by
                sort_ascending: Whether to sort in ascending order
            """
            return {"processed": len(data)}

        return [simple_tool, param_tool, complex_tool]

    def test_generate_tool_context_complete_structure(self) -> None:
        """generate_tool_context creates complete context structure."""
        tools = self.create_diverse_tools()
        registry = ToolRegistry(tools)

        context = registry.generate_tool_context()

        assert "# Available Tools" in context
        assert "You have access to the following tools:" in context

        # Check each tool is documented
        assert "## simple_tool" in context
        assert "## param_tool" in context
        assert "## complex_tool" in context

    def test_generate_tool_context_parameter_documentation(self) -> None:
        """generate_tool_context documents parameters correctly."""
        tools = self.create_diverse_tools()
        registry = ToolRegistry(tools)

        context = registry.generate_tool_context()

        # Check parameter documentation for param_tool
        param_tool_section = context[
            context.find("## param_tool") : context.find("## complex_tool")
        ]

        assert "Parameters:" in param_tool_section
        assert "required (string, required)" in param_tool_section
        assert "optional (integer, optional)" in param_tool_section
        assert "default: 42" in param_tool_section

    def test_generate_tool_context_no_parameters_message(self) -> None:
        """generate_tool_context shows message for tools with no parameters."""

        @tool
        def no_param_tool() -> str:
            """Tool without parameters."""
            return "test"

        registry = ToolRegistry([no_param_tool])
        context = registry.generate_tool_context()

        assert "No parameters required." in context

    def test_generate_tool_context_usage_instructions(self) -> None:
        """generate_tool_context includes usage instructions."""
        tools = self.create_diverse_tools()
        registry = ToolRegistry(tools)

        context = registry.generate_tool_context()

        assert "Use these tools when needed to help the user." in context
        assert "Always explain what you're doing when calling tools." in context
        assert (
            "If a tool call fails, continue helping the user as best you can."
            in context
        )

    def test_generate_tool_context_empty_registry(self) -> None:
        """generate_tool_context handles empty registry."""
        registry = ToolRegistry([])
        context = registry.generate_tool_context()

        assert context == ""

    def test_generate_tool_context_complex_types_formatting(self) -> None:
        """generate_tool_context formats complex types correctly."""
        tools = self.create_diverse_tools()
        registry = ToolRegistry(tools)

        context = registry.generate_tool_context()

        # Find complex_tool section
        complex_section = context[context.find("## complex_tool") :]

        assert "data (array, required)" in complex_section
        assert "filter_key (string, optional)" in complex_section
        assert "sort_ascending (boolean, optional)" in complex_section
        assert "default: True" in complex_section

    def test_generate_tool_context_description_formatting(self) -> None:
        """generate_tool_context formats tool descriptions correctly."""

        @tool
        def descriptive_tool() -> None:
            """This is a detailed tool description that explains what the tool does."""
            pass

        registry = ToolRegistry([descriptive_tool])
        context = registry.generate_tool_context()

        assert "This is a detailed tool description" in context


class TestToolRegistryStringRepresentation:
    """Test string representation methods."""

    def setup_method(self) -> None:
        """Set up test registry."""

        @tool
        def test_tool() -> str:
            """Test tool for string representation."""
            return "test"

        @tool
        def another_test() -> int:
            """Another test tool."""
            return 42

        self.registry = ToolRegistry([test_tool, another_test])

    def test_string_representation(self) -> None:
        """String representation includes tool count and names."""
        str_repr = str(self.registry)

        assert "ToolRegistry" in str_repr
        assert "2 tools" in str_repr
        assert "test_tool" in str_repr
        assert "another_test" in str_repr

    def test_repr_same_as_str(self) -> None:
        """__repr__ returns same as __str__."""
        assert repr(self.registry) == str(self.registry)


class TestToolRegistryErrorHandling:
    """Test error handling scenarios."""

    def test_schema_generation_failure_during_init(self) -> None:
        """Registry handles schema generation failures during initialization."""

        @tool
        def problematic_tool() -> None:
            """Tool that will cause schema generation issues."""
            pass

        # Tool decorator creates dynamic metadata for schema generation
        problematic_tool._tool_metadata.parameters = {"invalid": "structure"}  # pyright: ignore[reportFunctionMemberAccess]

        with pytest.raises(ValueError, match="Failed to generate schema"):
            ToolRegistry([problematic_tool])

    def test_registry_with_corrupted_metadata(self) -> None:
        """Registry handles corrupted tool metadata."""

        def corrupted_tool() -> None:
            """Tool with corrupted metadata."""
            pass

        # Tool decorator attributes are dynamically added for testing
        corrupted_tool._is_tool = True  # pyright: ignore[reportFunctionMemberAccess]
        corrupted_tool._tool_metadata = "not_metadata_object"  # pyright: ignore[reportFunctionMemberAccess]

        with pytest.raises(ValueError, match="invalid tool metadata"):
            ToolRegistry([corrupted_tool])


class TestToolRegistryRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_calculator_and_formatter_tools_registry(self) -> None:
        """Test registry with calculator and formatter tools."""

        @tool
        def calculate(expression: str, precision: int = 2) -> float:
            """Perform mathematical calculations.

            Args:
                expression: Mathematical expression to evaluate
                precision: Number of decimal places
            """
            result = eval(expression)  # Unsafe but OK for tests
            return round(float(result), precision)

        @tool
        def format_number(
            number: float, format_type: str = "decimal", currency: str = "USD"
        ) -> str:
            """Format a number according to specified format.

            Args:
                number: Number to format
                format_type: Format type (decimal, currency, percentage)
                currency: Currency code for currency formatting
            """
            if format_type == "currency":
                return f"{currency} {number:.2f}"
            elif format_type == "percentage":
                return f"{number:.1%}"
            else:
                return str(number)

        registry = ToolRegistry([calculate, format_number])

        # Test registry functionality
        assert registry.get_tool_count() == 2
        assert registry.has_tool("calculate")
        assert registry.has_tool("format_number")

        # Tool registry returns dynamically typed functions for execution
        calc_func = registry.get_tool_function("calculate")
        result = calc_func("10 + 5", 1)
        assert result == 15.0

        format_func = registry.get_tool_function("format_number")
        formatted = format_func(123.45, "currency", "EUR")
        assert formatted == "EUR 123.45"

        # Test schema generation
        schemas = registry.get_tool_schemas()
        assert len(schemas) == 2

        calc_schema = next(s for s in schemas if s["function"]["name"] == "calculate")
        assert calc_schema["function"]["parameters"]["required"] == ["expression"]

    def test_data_processing_tools_registry(self) -> None:
        """Test registry with data processing tools."""

        @tool
        def filter_data(
            data: List[Dict[str, Any]], filter_key: str, filter_value: Any
        ) -> List[Dict[str, Any]]:
            """Filter data records by key-value pair.

            Args:
                data: List of data records
                filter_key: Key to filter on
                filter_value: Value to match
            """
            return [item for item in data if item.get(filter_key) == filter_value]

        @tool
        def aggregate_data(
            data: List[Dict[str, Any]],
            group_by: str,
            agg_field: str,
            operation: str = "sum",
        ) -> Dict[str, float]:
            """Aggregate data by grouping key.

            Args:
                data: List of data records
                group_by: Field to group by
                agg_field: Field to aggregate
                operation: Aggregation operation (sum, avg, count)
            """
            groups: Dict[str, List[float]] = {}
            for item in data:
                key = str(item.get(group_by, "unknown"))
                if key not in groups:
                    groups[key] = []
                groups[key].append(float(item.get(agg_field, 0)))

            result = {}
            for key, values in groups.items():
                if operation == "sum":
                    result[key] = sum(values)
                elif operation == "avg":
                    result[key] = sum(values) / len(values) if values else 0
                elif operation == "count":
                    result[key] = float(len(values))
                else:
                    result[key] = sum(values)  # Default to sum

            # Return type has dynamic structure based on aggregation operation
            return result  # pyright: ignore[reportUnknownVariableType]

        registry = ToolRegistry([filter_data, aggregate_data])

        # Test context generation for complex tools
        context = registry.generate_tool_context()

        assert "filter_data" in context
        assert "aggregate_data" in context
        assert "data (array, required)" in context
        assert "operation (string, optional)" in context
        assert "default: sum" in context

        # Test validation
        validation_results = registry.validate_all_tools()
        assert all(validation_results.values())

        # Test tool execution with complex data
        test_data = [
            {"category": "A", "value": 10},
            {"category": "B", "value": 20},
            {"category": "A", "value": 15},
        ]

        # Tool functions return dynamically typed results for data processing
        filter_func = registry.get_tool_function("filter_data")
        filtered = filter_func(test_data, "category", "A")
        assert len(filtered) == 2

        agg_func = registry.get_tool_function("aggregate_data")
        aggregated = agg_func(test_data, "category", "value")
        assert aggregated["A"] == 25.0
        assert aggregated["B"] == 20.0

    def test_mixed_tool_types_registry(self) -> None:
        """Test registry with various tool types and complexity levels."""

        @tool
        def simple_ping() -> str:
            """Simple ping tool with no parameters."""
            return "pong"

        @tool
        def echo_message(message: str) -> str:
            """Echo a message back to the user.

            Args:
                message: Message to echo
            """
            return f"Echo: {message}"

        @tool
        def process_batch(
            items: List[str],
            operation: str,
            parallel: bool = False,
            max_workers: Optional[int] = None,
        ) -> Dict[str, Any]:
            """Process a batch of items.

            Args:
                items: List of items to process
                operation: Operation to perform
                parallel: Whether to process in parallel
                max_workers: Maximum number of worker threads
            """
            return {
                "processed": len(items),
                "operation": operation,
                "parallel": parallel,
                "workers": max_workers,
            }

        registry = ToolRegistry([simple_ping, echo_message, process_batch])

        # Test mixed complexity handling
        assert registry.get_tool_count() == 3

        # Tool functions have dynamic types for various complexity levels
        ping_func = registry.get_tool_function("simple_ping")
        assert ping_func() == "pong"

        # Medium complexity tools return dynamic typed results
        echo_func = registry.get_tool_function("echo_message")
        assert echo_func("hello") == "Echo: hello"

        # Complex tools process dynamic data structures
        batch_func = registry.get_tool_function("process_batch")
        result = batch_func(["a", "b", "c"], "transform", True, 4)
        assert result["processed"] == 3
        assert result["parallel"] is True
        assert result["workers"] == 4

        # Test context generation handles all complexity levels
        context = registry.generate_tool_context()

        # Simple tool
        assert "simple_ping" in context
        assert "No parameters required." in context

        # Medium complexity
        assert "echo_message" in context
        assert "message (string, required)" in context

        # Complex tool
        assert "process_batch" in context
        assert "parallel (boolean, optional)" in context
        assert "max_workers (integer, optional)" in context
