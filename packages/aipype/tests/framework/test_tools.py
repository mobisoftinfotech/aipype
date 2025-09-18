"""Comprehensive tests for tool decorator and metadata extraction."""

import pytest
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import Mock

# Tool decorator creates functions with dynamic type signatures for LLM tool calling
from aipype import tool, ToolMetadata, ToolSchemaGenerator


class TestToolDecorator:
    """Test the @tool decorator functionality."""

    def test_tool_decorator_preserves_function_name(self) -> None:
        """Tool decorator preserves original function name."""

        @tool
        def test_function(arg: str) -> str:
            """Test function for decorator."""
            return f"processed: {arg}"

        assert test_function.__name__ == "test_function"

    def test_tool_decorator_preserves_docstring(self) -> None:
        """Tool decorator preserves original function docstring."""

        @tool
        def documented_function(x: int) -> int:
            """This is a test function with documentation."""
            return x * 2

        assert (
            documented_function.__doc__ == "This is a test function with documentation."
        )

    def test_tool_decorator_adds_metadata_attribute(self) -> None:
        """Tool decorator adds _tool_metadata attribute."""

        @tool
        def metadata_test() -> None:
            """Test function for metadata."""
            pass

        # Tool decorator dynamically modifies function types
        assert hasattr(metadata_test, "_tool_metadata")
        # Tool decorator dynamically adds metadata attribute
        assert isinstance(metadata_test._tool_metadata, ToolMetadata)  # pyright: ignore[reportFunctionMemberAccess]

    def test_tool_decorator_adds_is_tool_flag(self) -> None:
        """Tool decorator adds _is_tool flag."""

        @tool
        def flag_test() -> None:
            """Test function for flag."""
            pass

        # Tool decorator dynamically modifies function types
        assert hasattr(flag_test, "_is_tool")
        # Tool decorator dynamically adds _is_tool attribute
        assert flag_test._is_tool is True  # pyright: ignore[reportFunctionMemberAccess]

    def test_tool_decorator_function_still_callable(self) -> None:
        """Tool decorator preserves function callability."""

        @tool
        def callable_test(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        # Tool-decorated function remains callable with original functionality
        # Tool decorator preserves function signature but changes type representation
        result: int = callable_test(3, 4)
        assert result == 7

    def test_tool_decorator_with_complex_signature(self) -> None:
        """Tool decorator works with complex function signatures."""

        @tool
        def complex_function(
            required: str,
            optional: Optional[str] = None,
            number: int = 42,
            flag: bool = False,
        ) -> Dict[str, Any]:
            """Function with complex signature."""
            return {
                "required": required,
                "optional": optional,
                "number": number,
                "flag": flag,
            }

        # Tool decorator dynamically modifies function types
        assert hasattr(complex_function, "_tool_metadata")
        # Tool decorator dynamically adds metadata attribute
        metadata = complex_function._tool_metadata  # pyright: ignore[reportFunctionMemberAccess]
        assert "required" in metadata.parameters
        assert "optional" in metadata.parameters
        assert "number" in metadata.parameters
        assert "flag" in metadata.parameters

    def test_tool_decorator_extracts_description_from_docstring(self) -> None:
        """Tool decorator extracts description from docstring first line."""

        @tool
        def described_function() -> None:
            """This is the main description.

            This is additional detail.

            Args:
                none: No arguments
            """
            pass

        # Tool decorator dynamically adds metadata attribute
        # Tool decorator dynamically adds metadata attribute
        metadata = described_function._tool_metadata  # pyright: ignore[reportFunctionMemberAccess]
        assert metadata.description == "This is the main description."

    def test_tool_decorator_falls_back_to_function_name(self) -> None:
        """Tool decorator falls back to function name when no docstring."""

        @tool
        def undocumented_function() -> None:
            pass

        # Tool decorator dynamically adds metadata attribute
        # Tool decorator dynamically adds metadata attribute
        metadata = undocumented_function._tool_metadata  # pyright: ignore[reportFunctionMemberAccess]
        assert "undocumented_function" in metadata.description

    def test_tool_decorator_extends_short_description(self) -> None:
        """Tool decorator extends description when first line is too short."""

        @tool
        def short_desc() -> None:
            """Test.

            This is a more detailed description that should be included.

            Args:
                none: No arguments
            """
            pass

        # Tool decorator dynamically adds metadata attribute
        # Tool decorator dynamically adds metadata attribute
        metadata = short_desc._tool_metadata  # pyright: ignore[reportFunctionMemberAccess]
        assert "detailed description" in metadata.description


class TestToolMetadata:
    """Test ToolMetadata class functionality."""

    def create_test_function(self) -> Any:
        """Create a test function for metadata extraction."""

        def test_func(
            name: str,
            age: int = 25,
            active: bool = True,
            tags: Optional[List[str]] = None,
        ) -> str:
            """Process user information.

            Args:
                name: User's full name
                age: User's age in years
                active: Whether user is active
                tags: List of user tags

            Returns:
                Processed user string
            """
            return f"User: {name}, Age: {age}"

        return test_func

    def test_metadata_initialization(self) -> None:
        """ToolMetadata initializes correctly with function details."""
        test_func = self.create_test_function()
        metadata = ToolMetadata(test_func, "test_func", "Test function")

        # Metadata stores reference to original function
        assert metadata.func == test_func
        assert metadata.name == "test_func"
        assert metadata.description == "Test function"
        assert isinstance(metadata.parameters, dict)

    def test_metadata_extracts_parameters(self) -> None:
        """ToolMetadata extracts parameter information correctly."""
        test_func = self.create_test_function()
        metadata = ToolMetadata(test_func, "test_func", "Test function")

        params = metadata.parameters

        # Check required parameter
        assert "name" in params
        assert params["name"]["optional"] is False
        assert params["name"]["type"] == "string"
        assert "full name" in params["name"]["description"]

        # Check optional parameters with defaults
        assert "age" in params
        assert params["age"]["optional"] is True
        assert params["age"]["type"] == "integer"
        assert params["age"]["default"] == 25

        assert "active" in params
        assert params["active"]["optional"] is True
        assert params["active"]["type"] == "boolean"
        assert params["active"]["default"] is True

    def test_metadata_handles_no_docstring(self) -> None:
        """ToolMetadata handles functions without docstrings."""

        def undocumented_func(x: int) -> int:
            return x * 2

        metadata = ToolMetadata(undocumented_func, "undocumented", "Test")

        params = metadata.parameters
        assert "x" in params
        assert "Parameter x" in params["x"]["description"]

    def test_metadata_parses_docstring_args_section(self) -> None:
        """ToolMetadata correctly parses Args section from docstring."""

        def documented_func(param1: str, param2: int) -> None:
            """Function with documented parameters.

            Args:
                param1: First parameter description
                param2: Second parameter with more
                    detailed description spanning
                    multiple lines
            """
            pass

        metadata = ToolMetadata(documented_func, "test", "Test")
        params = metadata.parameters

        assert params["param1"]["description"] == "First parameter description"
        assert "detailed description" in params["param2"]["description"]
        assert "multiple lines" in params["param2"]["description"]

    def test_metadata_handles_complex_args_section(self) -> None:
        """ToolMetadata handles complex Args sections with various formats."""

        def complex_args_func(simple: str, detailed: int, edge_case: bool) -> None:
            """Function with complex documentation.

            Args:
                simple: Simple description
                detailed: This is a detailed description that
                    spans multiple lines and includes various
                    formatting and edge cases
                edge_case: Description with: colons and other punctuation!

            Returns:
                Nothing special
            """
            pass

        metadata = ToolMetadata(complex_args_func, "test", "Test")
        params = metadata.parameters

        assert params["simple"]["description"] == "Simple description"
        assert "multiple lines" in params["detailed"]["description"]
        assert "colons and other punctuation" in params["edge_case"]["description"]

    def test_metadata_type_conversion_basic_types(self) -> None:
        """ToolMetadata correctly converts basic Python types."""

        def typed_func(text: str, number: int, decimal: float, flag: bool) -> None:
            """Function with basic types."""
            pass

        metadata = ToolMetadata(typed_func, "test", "Test")
        params = metadata.parameters

        assert params["text"]["type"] == "string"
        assert params["number"]["type"] == "integer"
        assert params["decimal"]["type"] == "number"
        assert params["flag"]["type"] == "boolean"

    def test_metadata_type_conversion_generic_types(self) -> None:
        """ToolMetadata correctly converts generic types."""

        def generic_func(
            items: List[str], mapping: Dict[str, int], optional: Optional[str] = None
        ) -> None:
            """Function with generic types."""
            pass

        metadata = ToolMetadata(generic_func, "test", "Test")
        params = metadata.parameters

        assert params["items"]["type"] == "array"
        assert params["mapping"]["type"] == "object"
        assert params["optional"]["type"] == "string"  # Extracts from Optional
        assert params["optional"]["optional"] is True

    def test_metadata_handles_no_type_hints(self) -> None:
        """ToolMetadata handles functions without type hints."""

        def untyped_func(param1: Any, param2: str = "default") -> None:
            """Function without type hints."""
            pass

        metadata = ToolMetadata(untyped_func, "test", "Test")
        params = metadata.parameters

        assert params["param1"]["type"] == "string"  # Default type
        assert params["param2"]["type"] == "string"  # Default type
        assert params["param2"]["default"] == "default"

    def test_metadata_skips_special_parameters(self) -> None:
        """ToolMetadata skips *args and **kwargs parameters."""

        def variadic_func(required: str, *args: Any, **kwargs: Any) -> None:
            """Function with variadic parameters."""
            pass

        metadata = ToolMetadata(variadic_func, "test", "Test")
        params = metadata.parameters

        assert "required" in params
        assert "args" not in params
        assert "kwargs" not in params

    def test_metadata_extracts_return_type(self) -> None:
        """ToolMetadata extracts return type information."""

        def return_typed_func() -> str:
            """Function with return type."""
            return "test"

        metadata = ToolMetadata(return_typed_func, "test", "Test")
        assert metadata.return_type == "string"

    def test_metadata_handles_no_return_type(self) -> None:
        """ToolMetadata handles functions without return type annotation."""

        def untyped_return_func() -> str:
            return "test"

        metadata = ToolMetadata(untyped_return_func, "test", "Test")
        assert metadata.return_type == "string"  # Default


class TestToolSchemaGenerator:
    """Test ToolSchemaGenerator functionality."""

    def create_decorated_function(self) -> Callable[..., float]:
        """Create a properly decorated test function."""

        @tool
        def calculate(expression: str, precision: int = 2) -> float:
            """Perform mathematical calculations.

            Args:
                expression: Mathematical expression to evaluate
                precision: Number of decimal places for result

            Returns:
                Calculated result
            """
            return eval(expression)  # Note: eval is unsafe, for testing only

        # Return the tool-decorated function
        return calculate

    def test_generate_schema_success(self) -> None:
        """ToolSchemaGenerator generates valid OpenAI schema."""
        calc_func = self.create_decorated_function()
        # Generate OpenAI-compatible schema from tool metadata
        schema = ToolSchemaGenerator.generate_schema(calc_func)
        assert schema["type"] == "function"
        assert "function" in schema

        function_def = schema["function"]
        assert function_def["name"] == "calculate"
        assert "mathematical calculations" in function_def["description"].lower()

        assert "parameters" in function_def
        parameters = function_def["parameters"]
        assert parameters["type"] == "object"
        assert "properties" in parameters
        assert "required" in parameters

    def test_generate_schema_properties_structure(self) -> None:
        """ToolSchemaGenerator creates proper properties structure."""
        calc_func = self.create_decorated_function()
        # Generate schema for property structure validation
        schema = ToolSchemaGenerator.generate_schema(calc_func)
        properties = schema["function"]["parameters"]["properties"]

        # Check required parameter
        assert "expression" in properties
        expr_prop = properties["expression"]
        assert expr_prop["type"] == "string"
        assert "expression to evaluate" in expr_prop["description"].lower()

        # Check optional parameter
        assert "precision" in properties
        precision_prop = properties["precision"]
        assert precision_prop["type"] == "integer"
        assert precision_prop["default"] == 2

    def test_generate_schema_required_parameters(self) -> None:
        """ToolSchemaGenerator correctly identifies required parameters."""
        calc_func = self.create_decorated_function()
        # Generate schema for required parameter validation
        schema = ToolSchemaGenerator.generate_schema(calc_func)
        required = schema["function"]["parameters"]["required"]
        assert "expression" in required
        assert "precision" not in required  # Has default value

    def test_generate_schema_no_parameters(self) -> None:
        """ToolSchemaGenerator handles functions with no parameters."""

        @tool
        def no_params() -> str:
            """Function with no parameters."""
            return "hello"

        # Generate schema for parameterless function
        schema = ToolSchemaGenerator.generate_schema(no_params)
        parameters = schema["function"]["parameters"]
        assert parameters["properties"] == {}
        assert parameters["required"] == []

    def test_generate_schema_all_optional_parameters(self) -> None:
        """ToolSchemaGenerator handles functions with all optional parameters."""

        @tool
        def all_optional(name: str = "default", count: int = 1) -> str:
            """Function with all optional parameters."""
            return f"{name} x{count}"

        # Generate schema for function with all optional parameters
        schema = ToolSchemaGenerator.generate_schema(all_optional)
        parameters = schema["function"]["parameters"]
        assert parameters["required"] == []
        assert len(parameters["properties"]) == 2

    def test_generate_schema_invalid_function_raises_error(self) -> None:
        """ToolSchemaGenerator raises error for non-decorated functions."""

        def regular_function() -> None:
            """Regular function without @tool decorator."""
            pass

        with pytest.raises(ValueError, match="not decorated with @tool"):
            # Should raise error for non-decorated function
            ToolSchemaGenerator.generate_schema(regular_function)

    def test_validate_tool_function_valid_tool(self) -> None:
        """ToolSchemaGenerator validates properly decorated functions."""
        calc_func = self.create_decorated_function()
        # Validate properly decorated tool function
        assert ToolSchemaGenerator.validate_tool_function(calc_func) is True

    def test_validate_tool_function_missing_decorator(self) -> None:
        """ToolSchemaGenerator rejects functions missing @tool decorator."""

        def undecorated() -> None:
            pass

        # Undecorated function should fail validation
        assert ToolSchemaGenerator.validate_tool_function(undecorated) is False

    def test_validate_tool_function_missing_metadata(self) -> None:
        """ToolSchemaGenerator rejects functions missing metadata."""

        def fake_tool() -> None:
            pass

        # Manually add _is_tool but not _tool_metadata
        # Manually add _is_tool but not _tool_metadata
        fake_tool._is_tool = True  # pyright: ignore[reportFunctionMemberAccess]
        # Function with incomplete tool attributes should fail validation
        assert ToolSchemaGenerator.validate_tool_function(fake_tool) is False

    def test_validate_tool_function_handles_schema_generation_errors(self) -> None:
        """ToolSchemaGenerator handles errors during schema generation."""

        def broken_tool() -> None:
            pass

        # Create invalid metadata that will cause schema generation to fail
        broken_metadata = Mock()
        broken_metadata.name = "broken"
        broken_metadata.description = "broken"
        broken_metadata.parameters = {"broken": "not_a_dict"}  # Invalid structure

        # Create function with invalid metadata to test error handling
        # Manually assign _is_tool attribute for error testing
        broken_tool._is_tool = True  # pyright: ignore[reportFunctionMemberAccess]
        # Manually assign metadata attribute for error testing
        broken_tool._tool_metadata = broken_metadata  # pyright: ignore[reportFunctionMemberAccess]
        # Function with broken metadata should fail validation
        assert ToolSchemaGenerator.validate_tool_function(broken_tool) is False


class TestTypeConversion:
    """Test type hint to JSON schema conversion."""

    def test_type_to_json_schema_basic_types(self) -> None:
        """Type conversion handles basic Python types correctly."""

        @tool
        def basic_types() -> None:
            pass

        # Tool decorator dynamically adds metadata attribute
        # Tool decorator dynamically adds metadata attribute
        metadata = basic_types._tool_metadata  # pyright: ignore[reportFunctionMemberAccess]
        assert metadata._type_to_json_schema(str) == {"type": "string"}
        assert metadata._type_to_json_schema(int) == {"type": "integer"}
        assert metadata._type_to_json_schema(float) == {"type": "number"}
        assert metadata._type_to_json_schema(bool) == {"type": "boolean"}

    def test_type_to_json_schema_generic_types(self) -> None:
        """Type conversion handles generic types correctly."""

        @tool
        def generic_types() -> None:
            pass

        # Tool decorator dynamically adds metadata attribute
        # Tool decorator dynamically adds metadata attribute
        metadata = generic_types._tool_metadata  # pyright: ignore[reportFunctionMemberAccess]
        assert metadata._type_to_json_schema(List[str]) == {
            "type": "array",
            "items": {"type": "string"},
        }
        assert metadata._type_to_json_schema(Dict[str, int]) == {"type": "object"}
        assert metadata._type_to_json_schema(list) == {
            "type": "array",
            "items": {"type": "string"},
        }
        assert metadata._type_to_json_schema(dict) == {"type": "object"}

    def test_type_to_json_schema_optional_types(self) -> None:
        """Type conversion extracts type from Optional."""

        @tool
        def optional_types() -> None:
            pass

        # Tool decorator dynamically adds metadata attribute
        # Tool decorator dynamically adds metadata attribute
        metadata = optional_types._tool_metadata  # pyright: ignore[reportFunctionMemberAccess]
        # Optional[str] should extract to schema with "string" type
        assert metadata._type_to_json_schema(Optional[str]) == {"type": "string"}
        assert metadata._type_to_json_schema(Optional[int]) == {"type": "integer"}

    def test_type_to_json_schema_unknown_types_default_to_string(self) -> None:
        """Type conversion defaults to string for unknown types."""

        @tool
        def unknown_types() -> None:
            pass

        # Tool decorator dynamically adds metadata attribute
        # Tool decorator dynamically adds metadata attribute
        metadata = unknown_types._tool_metadata  # pyright: ignore[reportFunctionMemberAccess]

        # Custom classes and unknown types should default to "string"
        class CustomClass:
            pass

        assert metadata._type_to_json_schema(CustomClass) == {"type": "string"}
        assert metadata._type_to_json_schema("unknown") == {"type": "string"}


class TestDocstringParsing:
    """Test docstring parsing functionality."""

    def test_parse_docstring_args_basic(self) -> None:
        """Docstring parsing handles basic Args sections."""

        @tool
        def test_func() -> None:
            pass

        # Tool decorator dynamically adds metadata attribute
        # Tool decorator dynamically adds metadata attribute
        metadata = test_func._tool_metadata  # pyright: ignore[reportFunctionMemberAccess]
        docstring = """Function description.

        Args:
            param1: First parameter
            param2: Second parameter
        """

        descriptions = metadata._parse_docstring_args(docstring)
        assert descriptions["param1"] == "First parameter"
        assert descriptions["param2"] == "Second parameter"

    def test_parse_docstring_args_multiline(self) -> None:
        """Docstring parsing handles multiline parameter descriptions."""

        @tool
        def test_func() -> None:
            pass

        # Tool decorator dynamically adds metadata attribute
        # Tool decorator dynamically adds metadata attribute
        metadata = test_func._tool_metadata  # pyright: ignore[reportFunctionMemberAccess]
        docstring = """Function description.

        Args:
            param1: This is a very long description
                that spans multiple lines and includes
                various details about the parameter
            param2: Short description
        """

        descriptions = metadata._parse_docstring_args(docstring)
        expected = "This is a very long description that spans multiple lines and includes various details about the parameter"
        assert descriptions["param1"] == expected
        assert descriptions["param2"] == "Short description"

    def test_parse_docstring_args_with_returns_section(self) -> None:
        """Docstring parsing stops at Returns section."""

        @tool
        def test_func() -> None:
            pass

        # Tool decorator dynamically adds metadata attribute
        # Tool decorator dynamically adds metadata attribute
        metadata = test_func._tool_metadata  # pyright: ignore[reportFunctionMemberAccess]
        docstring = """Function description.

        Args:
            param1: Parameter description

        Returns:
            return_desc: This should not be parsed as a parameter
        """

        descriptions = metadata._parse_docstring_args(docstring)
        assert descriptions["param1"] == "Parameter description"
        assert "return_desc" not in descriptions

    def test_parse_docstring_args_no_args_section(self) -> None:
        """Docstring parsing handles missing Args section."""

        @tool
        def test_func() -> None:
            pass

        # Tool decorator dynamically adds metadata attribute
        # Tool decorator dynamically adds metadata attribute
        metadata = test_func._tool_metadata  # pyright: ignore[reportFunctionMemberAccess]
        docstring = """Function description without Args section."""

        descriptions = metadata._parse_docstring_args(docstring)
        assert descriptions == {}

    def test_parse_docstring_args_case_insensitive(self) -> None:
        """Docstring parsing is case insensitive for Args section."""

        @tool
        def test_func() -> None:
            pass

        # Tool decorator dynamically adds metadata attribute
        # Tool decorator dynamically adds metadata attribute
        metadata = test_func._tool_metadata  # pyright: ignore[reportFunctionMemberAccess]
        docstring = """Function description.

        args:
            param1: Lower case args section
        """

        descriptions = metadata._parse_docstring_args(docstring)
        assert descriptions["param1"] == "Lower case args section"

    def test_parse_docstring_args_complex_parameter_names(self) -> None:
        """Docstring parsing handles complex parameter names."""

        @tool
        def test_func() -> None:
            pass

        # Tool decorator dynamically adds metadata attribute
        # Tool decorator dynamically adds metadata attribute
        metadata = test_func._tool_metadata  # pyright: ignore[reportFunctionMemberAccess]
        docstring = """Function description.

        Args:
            simple_param: Simple parameter
            param_with_underscores: Parameter with underscores
            CamelCaseParam: CamelCase parameter (unusual but valid)
        """

        descriptions = metadata._parse_docstring_args(docstring)
        assert descriptions["simple_param"] == "Simple parameter"
        assert descriptions["param_with_underscores"] == "Parameter with underscores"
        assert (
            descriptions["CamelCaseParam"] == "CamelCase parameter (unusual but valid)"
        )


class TestIntegrationScenarios:
    """Test integration scenarios with real-world tool examples."""

    def test_calculator_tool_complete_workflow(self) -> None:
        """Test complete workflow with calculator tool."""

        @tool
        def calculate(expression: str, precision: int = 2) -> float:
            """Perform mathematical calculations.

            Args:
                expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")
                precision: Number of decimal places for result

            Returns:
                Calculated numerical result
            """
            # Simple evaluation for testing (unsafe in production)
            result = eval(expression)
            return round(float(result), precision)

        # Test decorator application
        # Tool decorator dynamically modifies function types
        assert hasattr(calculate, "_tool_metadata")
        assert hasattr(calculate, "_is_tool")
        # Tool decorator dynamically adds _is_tool attribute
        assert calculate._is_tool is True  # pyright: ignore[reportFunctionMemberAccess]
        # Test function still works
        # Tool-decorated function remains callable
        # Tool decorator preserves function signature but changes type representation
        result: float = calculate("2 + 3", 1)
        assert result == 5.0

        # Test metadata extraction
        # Tool decorator dynamically adds metadata attribute
        # Tool decorator dynamically adds metadata attribute
        metadata = calculate._tool_metadata  # pyright: ignore[reportFunctionMemberAccess]
        assert metadata.name == "calculate"
        assert "mathematical calculations" in metadata.description.lower()

        # Test parameter extraction
        params = metadata.parameters
        assert len(params) == 2
        assert params["expression"]["type"] == "string"
        assert params["expression"]["optional"] is False
        assert params["precision"]["type"] == "integer"
        assert params["precision"]["optional"] is True
        assert params["precision"]["default"] == 2

        # Test schema generation
        # Generate OpenAI-compatible schema from tool metadata
        schema = ToolSchemaGenerator.generate_schema(calculate)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "calculate"
        required_params = schema["function"]["parameters"]["required"]
        assert "expression" in required_params
        assert "precision" not in required_params

    def test_file_reader_tool_with_optional_parameters(self) -> None:
        """Test tool with multiple optional parameters."""

        @tool
        def read_file(
            filepath: str,
            encoding: str = "utf-8",
            max_lines: Optional[int] = None,
            binary_mode: bool = False,
        ) -> str:
            """Read content from a file.

            Args:
                filepath: Path to the file to read
                encoding: Character encoding to use when reading
                max_lines: Maximum number of lines to read (None for all)
                binary_mode: Whether to read in binary mode

            Returns:
                File content as string
            """
            # Mock implementation for testing
            return f"Content from {filepath}"

        # Tool decorator dynamically adds metadata attribute
        # Tool decorator dynamically adds metadata attribute
        metadata = read_file._tool_metadata  # pyright: ignore[reportFunctionMemberAccess]
        params = metadata.parameters

        # Check required parameter
        assert params["filepath"]["optional"] is False
        assert params["filepath"]["type"] == "string"

        # Check optional parameters with defaults
        assert params["encoding"]["optional"] is True
        assert params["encoding"]["default"] == "utf-8"
        assert params["binary_mode"]["optional"] is True
        assert params["binary_mode"]["default"] is False

        # Check optional parameter with None default
        assert params["max_lines"]["optional"] is True
        assert "max_lines" not in params or "default" not in params["max_lines"]

        # Test schema generation
        # Generate schema for function with mixed parameter types
        schema = ToolSchemaGenerator.generate_schema(read_file)
        required = schema["function"]["parameters"]["required"]
        assert required == ["filepath"]  # Only required parameter

    def test_data_processor_tool_with_complex_types(self) -> None:
        """Test tool with complex type annotations."""

        @tool
        def process_data(
            data: List[Dict[str, Any]],
            filters: Optional[Dict[str, str]] = None,
            sort_keys: Optional[List[str]] = None,
            output_format: str = "json",
        ) -> Dict[str, Any]:
            """Process structured data with filtering and sorting.

            Args:
                data: List of data records to process
                filters: Key-value pairs for filtering records
                sort_keys: List of keys to sort by
                output_format: Output format (json, csv, xml)

            Returns:
                Processed data with metadata
            """
            sort_keys_list = sort_keys or []
            return {
                "processed": len(data),
                "format": output_format,
                "sort_keys": len(sort_keys_list),
            }

        # Tool decorator dynamically adds metadata attribute
        # Tool decorator dynamically adds metadata attribute
        metadata = process_data._tool_metadata  # pyright: ignore[reportFunctionMemberAccess]
        params = metadata.parameters

        # Check complex type handling
        assert params["data"]["type"] == "array"
        assert params["filters"]["type"] == "object"
        assert params["sort_keys"]["type"] == "array"
        assert params["output_format"]["type"] == "string"

        # Check optional status
        assert params["data"]["optional"] is False
        assert params["filters"]["optional"] is True
        assert params["sort_keys"]["optional"] is True
        assert params["output_format"]["optional"] is True

        # Test function execution
        test_data = [{"name": "test", "value": 1}]
        # Tool-decorated function remains callable with original behavior
        # Tool decorator preserves function signature but changes type representation
        result: Dict[str, Any] = process_data(test_data)
        assert result["processed"] == 1
        assert result["format"] == "json"

    def test_error_handling_tool(self) -> None:
        """Test tool that demonstrates error handling patterns."""

        @tool
        def divide_numbers(numerator: float, denominator: float) -> float:
            """Divide two numbers with error handling.

            Args:
                numerator: The number to be divided
                denominator: The number to divide by (cannot be zero)

            Returns:
                Result of division

            Raises:
                ValueError: When denominator is zero
            """
            if denominator == 0:
                raise ValueError("Division by zero is not allowed")
            return numerator / denominator

        # Test normal operation
        # Tool-decorated function maintains original functionality
        # Tool decorator preserves function signature but changes type representation
        result: float = divide_numbers(10.0, 2.0)
        assert result == 5.0

        # Test error handling
        with pytest.raises(ValueError, match="Division by zero"):
            divide_numbers(10.0, 0.0)

        # Test metadata
        # Tool decorator dynamically adds metadata attribute
        # Tool decorator dynamically adds metadata attribute
        metadata = divide_numbers._tool_metadata  # pyright: ignore[reportFunctionMemberAccess]
        assert metadata.name == "divide_numbers"
        assert len(metadata.parameters) == 2
        assert all(param["type"] == "number" for param in metadata.parameters.values())

        # Test schema validation
        # Validate tool function and generate schema
        assert ToolSchemaGenerator.validate_tool_function(divide_numbers) is True
        schema = ToolSchemaGenerator.generate_schema(divide_numbers)
        assert len(schema["function"]["parameters"]["required"]) == 2
