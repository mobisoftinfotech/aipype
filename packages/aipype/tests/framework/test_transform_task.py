"""Comprehensive tests for TransformTask - Generic data transformation task."""

import pytest
import time
from typing import Any, Dict, List
from unittest.mock import Mock, patch


from aipype import (
    TransformTask,
    extract_field_from_list,
    combine_text_fields,
    filter_by_condition,
    aggregate_numeric_field,
    format_as_markdown_list,
    TaskContext,
    TaskDependency,
    DependencyType,
)


class TestTransformTaskCore:
    """Test core TransformTask functionality - initialization, configuration, basic operations."""

    def test_transform_task_initialization_minimal(self) -> None:
        """TransformTask initializes with minimal parameters."""
        task = TransformTask("test_transform")

        assert task.name == "test_transform"
        assert task.config == {}
        assert task.get_dependencies() == []
        assert task.context_instance is None

    def test_transform_task_initialization_full_config(self) -> None:
        """TransformTask initializes with full configuration."""

        def dummy_transform(x: Any) -> str:
            return str(x)

        def dummy_validator(x: Any) -> bool:
            return x is not None

        config = {
            "transform_function": dummy_transform,
            "input_field": "data",
            "output_name": "transformed_data",
            "validate_input": True,
            "validate_output": True,
            "input_validator": dummy_validator,
            "output_validator": dummy_validator,
        }

        dependencies = [TaskDependency("data", "source.data", DependencyType.REQUIRED)]

        task = TransformTask("full_transform", config, dependencies)

        assert task.name == "full_transform"
        assert task.config == config
        assert len(task.get_dependencies()) == 1
        assert task.get_dependencies()[0].name == "data"
        assert task.context_instance is None

    def test_set_context_stores_context_instance(self) -> None:
        """set_context method stores TaskContext instance."""
        task = TransformTask("test")
        context = TaskContext()

        task.set_context(context)

        assert task.context_instance is context

    def test_string_representation(self) -> None:
        """String representation shows task details."""
        # Task with no dependencies
        task1 = TransformTask("simple_transform", {"output_name": "result"})
        str_repr1 = str(task1)

        assert "TransformTask" in str_repr1
        assert "name='simple_transform'" in str_repr1
        assert "dependencies=0" in str_repr1
        assert "output='result'" in str_repr1

        # Task with dependencies and custom output name
        dependencies = [
            TaskDependency("data1", "source1.data", DependencyType.REQUIRED),
            TaskDependency("data2", "source2.data", DependencyType.OPTIONAL),
        ]
        task2 = TransformTask(
            "complex_transform", {"output_name": "processed_data"}, dependencies
        )
        str_repr2 = str(task2)

        assert "name='complex_transform'" in str_repr2
        assert "dependencies=2" in str_repr2
        assert "output='processed_data'" in str_repr2

    def test_string_representation_default_output_name(self) -> None:
        """String representation uses default output name when not specified."""
        task = TransformTask("default_output")
        str_repr = str(task)

        assert "output='result'" in str_repr


class TestTransformTaskInputHandling:
    """Test TransformTask input data retrieval and handling."""

    def test_get_input_data_single_field(self) -> None:
        """input_data property retrieves single input field correctly."""
        config = {
            "input_field": "test_data",
            "test_data": {"value": 42, "text": "hello"},
            "other_field": "should_be_ignored",
        }

        task = TransformTask("test", config)
        input_data = task.input_data

        assert input_data == {"test_data": {"value": 42, "text": "hello"}}

    def test_get_input_data_multiple_fields(self) -> None:
        """input_data property retrieves multiple input fields correctly."""
        config = {
            "input_fields": ["data1", "data2", "data3"],
            "data1": [1, 2, 3],
            "data2": {"key": "value"},
            "data3": "string data",
            "ignored_field": "not included",
        }

        task = TransformTask("test", config)
        input_data = task.input_data

        expected = {
            "data1": [1, 2, 3],
            "data2": {"key": "value"},
            "data3": "string data",
        }
        assert input_data == expected

    def test_get_input_data_from_dependencies(self) -> None:
        """input_data property uses dependency data when no explicit fields specified."""
        dependencies = [
            TaskDependency("dep1", "source1.data", DependencyType.REQUIRED),
            TaskDependency("dep2", "source2.info", DependencyType.REQUIRED),
        ]

        config = {
            "dep1": "dependency_data_1",
            "dep2": {"nested": "data"},
            "unrelated": "ignored",
        }

        task = TransformTask("test", config, dependencies)
        input_data = task.input_data

        expected = {"dep1": "dependency_data_1", "dep2": {"nested": "data"}}
        assert input_data == expected

    def test_get_input_data_missing_single_field_error(self) -> None:
        """input_data property raises error when single input field is missing."""
        config = {"input_field": "missing_field", "other_data": "present"}

        task = TransformTask("test", config)

        with pytest.raises(ValueError, match="Input field 'missing_field' not found"):
            task.input_data

    def test_get_input_data_missing_multiple_field_error(self) -> None:
        """input_data property raises error when multiple input field is missing."""
        config = {
            "input_fields": ["field1", "field2", "missing_field"],
            "field1": "data1",
            "field2": "data2",
        }

        task = TransformTask("test", config)

        with pytest.raises(ValueError, match="Input field 'missing_field' not found"):
            task.input_data

    def test_get_input_data_invalid_input_fields_type(self) -> None:
        """input_data property raises error when input_fields is not a list."""
        config = {"input_fields": "not_a_list"}

        task = TransformTask("test", config)

        with pytest.raises(ValueError, match="input_fields must be a list"):
            task.input_data

    def test_get_input_data_no_data_validation_enabled(self) -> None:
        """input_data property raises error when no data available and validation enabled."""
        task = TransformTask("test", {})  # No dependencies, no input fields

        with pytest.raises(
            ValueError, match="No input data available for transformation"
        ):
            task.input_data

    def test_get_input_data_no_data_validation_disabled(self) -> None:
        """input_data property returns None when no data available and validation disabled."""
        task = TransformTask("test", {"validate_input": False})

        input_data = task.input_data
        assert input_data is None


class TestTransformTaskTransformations:
    """Test TransformTask transformation execution with different parameter patterns."""

    def test_transform_execution_single_input(self) -> None:
        """TransformTask executes transformation with single input."""

        def double_value(x: int) -> int:
            return x * 2

        config = {
            "transform_function": double_value,
            "input_field": "number",
            "number": 21,
            "output_name": "doubled",
        }

        task = TransformTask("double", config)
        result = task.run()
        assert result.is_success()
        assert result.data["doubled"] == 42
        assert result.data["transformation"] == "double"
        assert result.data["input_fields"] == ["number"]
        assert result.data["output_type"] == "int"

    def test_transform_execution_multiple_inputs_kwargs(self) -> None:
        """TransformTask executes transformation with multiple inputs as kwargs."""

        def combine_strings(first: str, second: str, separator: str = " ") -> str:
            return f"{first}{separator}{second}"

        config = {
            "transform_function": combine_strings,
            "input_fields": ["first", "second", "separator"],
            "first": "Hello",
            "second": "World",
            "separator": "-",
            "output_name": "combined",
        }

        task = TransformTask("combine", config)
        result = task.run()
        assert result.is_success()
        assert result.data["combined"] == "Hello-World"
        assert result.data["transformation"] == "combine"
        assert result.data["input_fields"] == ["first", "second", "separator"]
        assert result.data["output_type"] == "str"

    def test_transform_execution_direct_input(self) -> None:
        """TransformTask executes transformation with direct input (non-dict)."""

        def list_length(items: List[Any]) -> int:
            return len(items)

        # Simulate direct input by having no input_field or input_fields
        # and only one dependency
        config = {
            "transform_function": list_length,
            "validate_input": False,  # Allow direct input
        }

        task = TransformTask("length", config)
        # Mock _get_input_data to return direct list
        task._get_input_data = lambda: [1, 2, 3, 4, 5]  # type: ignore

        result = task.run()
        assert result.is_success()
        assert result.data["result"] == 5
        assert result.data["transformation"] == "length"
        assert result.data["input_fields"] == ["input"]
        assert result.data["output_type"] == "int"

    def test_transform_execution_default_output_name(self) -> None:
        """TransformTask uses default output name when not specified."""

        def identity(x: Any) -> Any:
            return x

        config = {
            "transform_function": identity,
            "input_field": "data",
            "data": "test_value",
        }

        task = TransformTask("identity", config)
        result = task.run()
        assert result.is_success()
        assert result.data["result"] == "test_value"  # Default output name

    def test_transform_execution_complex_return_type(self) -> None:
        """TransformTask handles complex return types correctly."""

        def create_report(data: List[Dict[str, Any]]) -> Dict[str, Any]:
            return {
                "summary": f"Processed {len(data)} items",
                "details": [item["id"] for item in data],
                "metadata": {"timestamp": "2024-01-01", "version": "1.0"},
            }

        input_data = [
            {"id": "A", "value": 10},
            {"id": "B", "value": 20},
            {"id": "C", "value": 30},
        ]

        config = {
            "transform_function": create_report,
            "input_field": "items",
            "items": input_data,
            "output_name": "report",
        }

        task = TransformTask("reporting", config)
        result = task.run()
        assert result.is_success()
        assert result.data["report"]["summary"] == "Processed 3 items"
        assert result.data["report"]["details"] == ["A", "B", "C"]
        assert result.data["output_type"] == "dict"


class TestTransformTaskValidation:
    """Test TransformTask input and output validation functionality."""

    def test_input_validation_enabled_success(self) -> None:
        """Input validation passes when data is valid."""

        def process_list(items: List[int]) -> int:
            return sum(items)

        config = {
            "transform_function": process_list,
            "input_field": "numbers",
            "numbers": [1, 2, 3, 4, 5],
            "validate_input": True,
        }

        task = TransformTask("sum_numbers", config)
        result = task.run()
        assert result.is_success()
        assert result.data["result"] == 15

    def test_input_validation_none_input_fails(self) -> None:
        """Input validation fails when input is None."""

        def process_data(data: Any) -> str:
            return str(data)

        config = {"transform_function": process_data, "validate_input": True}

        task = TransformTask("process", config)
        # Mock _get_input_data to return None
        task._get_input_data = lambda: None  # type: ignore

        result = task.run()
        assert result.is_error()
        assert result.error is not None and "Input data cannot be None" in result.error

    def test_custom_input_validator_success(self) -> None:
        """Custom input validator passes when condition is met."""

        def is_positive_number(input_dict: Dict[str, Any]) -> bool:
            # Validator receives input_data dict, need to extract the actual value
            if "number" in input_dict:
                x = input_dict["number"]
                return isinstance(x, int) and x > 0
            return False

        def square(x: int) -> int:
            return x * x

        config = {
            "transform_function": square,
            "input_field": "number",
            "number": 5,
            "validate_input": True,
            "input_validator": is_positive_number,
        }

        task = TransformTask("square_positive", config)
        result = task.run()
        assert result.is_success()
        assert result.data["result"] == 25

    def test_custom_input_validator_failure(self) -> None:
        """Custom input validator fails when condition is not met."""

        def is_positive_number(input_dict: Dict[str, Any]) -> bool:
            # Validator receives input_data dict, need to extract the actual value
            if "number" in input_dict:
                x = input_dict["number"]
                return isinstance(x, int) and x > 0
            return False

        def square(x: int) -> int:
            return x * x

        config = {
            "transform_function": square,
            "input_field": "number",
            "number": -5,  # Negative number should fail validation
            "validate_input": True,
            "input_validator": is_positive_number,
        }

        task = TransformTask("square_positive", config)

        result = task.run()
        assert result.is_error()
        assert result.error is not None and "Input validation error" in result.error

    def test_custom_input_validator_exception_handling(self) -> None:
        """Custom input validator exceptions are handled properly."""

        def failing_validator(x: Any) -> bool:
            raise RuntimeError("Validator crashed")

        def identity(x: Any) -> Any:
            return x

        config = {
            "transform_function": identity,
            "input_field": "data",
            "data": "test",
            "validate_input": True,
            "input_validator": failing_validator,
        }

        task = TransformTask("test_validation", config)

        result = task.run()
        assert result.is_error()
        assert (
            result.error is not None
            and "Input validation error: Validator crashed" in result.error
        )

    def test_output_validation_enabled_success(self) -> None:
        """Output validation passes when output is valid."""

        def is_valid_result(result: Dict[str, Any]) -> bool:
            return "value" in result

        def create_result(x: int) -> Dict[str, Any]:
            return {"value": x, "type": "number"}

        config = {
            "transform_function": create_result,
            "input_field": "number",
            "number": 42,
            "validate_output": True,
            "output_validator": is_valid_result,
        }

        task = TransformTask("create_valid", config)
        result = task.run()
        assert result.is_success()
        assert result.data["result"]["value"] == 42

    def test_output_validation_failure(self) -> None:
        """Output validation fails when output is invalid."""

        def is_valid_result(result: Dict[str, Any]) -> bool:
            return "required_field" in result

        def create_invalid_result(x: int) -> Dict[str, Any]:
            return {"value": x}  # Missing required_field

        config = {
            "transform_function": create_invalid_result,
            "input_field": "number",
            "number": 42,
            "validate_output": True,
            "output_validator": is_valid_result,
        }

        task = TransformTask("create_invalid", config)

        result = task.run()
        assert result.is_error()
        assert result.error is not None and "Output validation failed" in result.error

    def test_output_validation_exception_handling(self) -> None:
        """Output validator exceptions are handled properly."""

        def failing_output_validator(result: Any) -> bool:
            raise RuntimeError("Output validator crashed")

        def identity(x: Any) -> Any:
            return x

        config = {
            "transform_function": identity,
            "input_field": "data",
            "data": "test",
            "validate_output": True,
            "output_validator": failing_output_validator,
        }

        task = TransformTask("test_output_validation", config)

        result = task.run()
        assert result.is_error()
        assert (
            result.error is not None
            and "Output validation error: Output validator crashed" in result.error
        )


class TestTransformTaskPreview:
    """Test TransformTask preview functionality."""

    def test_preview_transformation_single_input(self) -> None:
        """preview_transformation shows correct information for single input."""

        def upper_case(text: str) -> str:
            return text.upper()

        config = {
            "transform_function": upper_case,
            "input_field": "text",
            "text": "hello world",
            "output_name": "uppercase_text",
        }

        task = TransformTask("uppercase", config)
        preview = task.preview_transformation()

        assert preview["task_name"] == "uppercase"
        assert preview["input_fields"] == ["text"]
        assert preview["input_types"]["text"] == "str"
        assert preview["output_name"] == "uppercase_text"
        assert "upper_case" in preview["transform_function"]

    def test_preview_transformation_multiple_inputs(self) -> None:
        """preview_transformation shows correct information for multiple inputs."""

        def combine_data(
            numbers: List[int], text: str, metadata: Dict[str, Any]
        ) -> Dict[str, Any]:
            return {"numbers": numbers, "text": text, "metadata": metadata}

        config = {
            "transform_function": combine_data,
            "input_fields": ["numbers", "text", "metadata"],
            "numbers": [1, 2, 3, 4, 5],
            "text": "sample text",
            "metadata": {"author": "test", "version": 1},
        }

        task = TransformTask("combine_preview", config)
        preview = task.preview_transformation()

        assert preview["task_name"] == "combine_preview"
        assert preview["input_fields"] == ["numbers", "text", "metadata"]
        assert preview["input_types"]["numbers"] == "list"
        assert preview["input_types"]["numbers_size"] == 5
        assert preview["input_types"]["text"] == "str"
        assert preview["input_types"]["metadata"] == "dict"
        assert preview["input_types"]["metadata_size"] == 2
        assert preview["output_name"] == "result"

    def test_preview_transformation_direct_input(self) -> None:
        """preview_transformation handles direct input correctly."""

        def count_items(items: List[str]) -> int:
            return len(items)

        config = {"transform_function": count_items}

        task = TransformTask("count", config)
        # Mock _get_input_data to return direct list
        task._get_input_data = lambda: ["a", "b", "c"]  # type: ignore

        preview = task.preview_transformation()

        assert preview["task_name"] == "count"
        assert preview["input_fields"] == ["input"]
        assert preview["input_types"]["input"] == "list"

    def test_preview_transformation_error_handling(self) -> None:
        """preview_transformation handles errors gracefully."""

        def failing_transform() -> str:
            return "test"

        config = {"transform_function": failing_transform}

        task = TransformTask("failing", config)
        # Mock _get_input_data to raise an error
        task._get_input_data = lambda: exec('raise RuntimeError("Mock error")')  # type: ignore

        preview = task.preview_transformation()

        assert preview["task_name"] == "failing"
        assert "error" in preview
        assert "Preview failed" in preview["error"]

    def test_preview_transformation_no_transform_function(self) -> None:
        """preview_transformation handles missing transform function."""
        task = TransformTask("no_func", {})

        preview = task.preview_transformation()

        assert preview["task_name"] == "no_func"
        # When no transform function and no input data, preview should fail with error
        assert "error" in preview
        assert "Preview failed" in preview["error"]


class TestTransformTaskEdgeCases:
    """Test TransformTask error handling, edge cases, and robustness."""

    def test_missing_transform_function_error(self) -> None:
        """TransformTask raises error when transform_function is missing."""
        task = TransformTask("missing_func", {"input_field": "data", "data": "test"})

        result = task.run()
        assert result.is_error()
        assert (
            result.error is not None
            and "Missing required fields: ['transform_function']" in result.error
        )

    def test_non_callable_transform_function_error(self) -> None:
        """TransformTask raises error when transform_function is not callable."""
        config = {
            "transform_function": "not_callable",
            "input_field": "data",
            "data": "test",
        }

        task = TransformTask("invalid_func", config)

        result = task.run()
        assert result.is_error()
        assert (
            result.error is not None
            and "transform_function failed custom validation" in result.error
        )

    def test_transformation_runtime_error_handling(self) -> None:
        """TransformTask handles runtime errors during transformation."""

        def failing_transform(x: Any) -> str:
            raise RuntimeError("Transformation failed during execution")

        config = {
            "transform_function": failing_transform,
            "input_field": "data",
            "data": "test",
        }

        task = TransformTask("failing_transform", config)

        result = task.run()
        assert result.is_error()
        assert result.error is not None and (
            "Transformation failed: Transformation failed during execution"
            in result.error
        )

    def test_transformation_type_error_handling(self) -> None:
        """TransformTask handles type errors during transformation."""

        def type_sensitive_transform(x: int) -> int:
            return x + 10  # Will fail if x is not int

        config = {
            "transform_function": type_sensitive_transform,
            "input_field": "data",
            "data": "not_an_integer",  # Wrong type
        }

        task = TransformTask("type_error", config)

        result = task.run()
        assert result.is_error()
        assert result.error is not None and "Transformation failed" in result.error

    def test_large_data_transformation_performance(self) -> None:
        """TransformTask handles large datasets efficiently."""

        def sum_large_list(numbers: List[int]) -> int:
            return sum(numbers)

        # Create large dataset
        large_dataset = list(range(100000))  # 100K integers

        config = {
            "transform_function": sum_large_list,
            "input_field": "numbers",
            "numbers": large_dataset,
        }

        task = TransformTask("large_sum", config)

        start_time = time.time()
        result = task.run()
        assert result.is_success()
        execution_time = time.time() - start_time

        # Verify correctness
        expected_sum = sum(large_dataset)
        assert result.data["result"] == expected_sum

        # Verify reasonable performance (should complete quickly)
        assert execution_time < 1.0, (
            f"Large data transformation too slow: {execution_time:.3f}s"
        )

    def test_memory_efficiency_with_large_transformations(self) -> None:
        """TransformTask is memory efficient with large transformations."""

        def create_large_output(size: int) -> List[Dict[str, Any]]:
            # Create large output structure
            return [{"id": i, "data": f"item_{i}", "value": i * 2} for i in range(size)]

        config = {
            "transform_function": create_large_output,
            "input_field": "size",
            "size": 10000,  # Create 10K items
            "output_name": "large_result",
        }

        task = TransformTask("memory_test", config)
        result = task.run()
        assert result.is_success()
        # Verify result structure
        assert len(result.data["large_result"]) == 10000
        assert result.data["large_result"][0]["id"] == 0
        assert result.data["large_result"][9999]["id"] == 9999

    def test_nested_data_structure_transformations(self) -> None:
        """TransformTask handles deeply nested data structures correctly."""

        def flatten_nested_data(nested: Dict[str, Any]) -> List[Any]:
            def extract_values(obj: Any) -> List[Any]:
                if isinstance(obj, dict):
                    dict_values: List[Any] = []
                    for value in obj.values():  # pyright: ignore
                        dict_values.extend(extract_values(value))
                    return dict_values
                elif isinstance(obj, list):
                    list_values: List[Any] = []
                    for item in obj:  # pyright: ignore
                        list_values.extend(extract_values(item))
                    return list_values
                else:
                    return [obj]

            return extract_values(nested)

        nested_data = {
            "level1": {
                "level2": {"level3": [1, 2, {"level4": [3, 4, 5]}]},
                "other": [6, 7],
            },
            "top_level": 8,
        }

        config = {
            "transform_function": flatten_nested_data,
            "input_field": "nested",
            "nested": nested_data,
        }

        task = TransformTask("flatten", config)
        result = task.run()
        assert result.is_success()
        # Should flatten all nested values
        flattened = result.data["result"]
        assert sorted(flattened) == [1, 2, 3, 4, 5, 6, 7, 8]


class TestUtilityTransformFunctions:
    """Test all utility transformation functions provided by the module."""

    def test_extract_field_from_list_basic(self) -> None:
        """extract_field_from_list extracts specified field from list items."""
        items = [
            {"url": "http://example1.com", "title": "Article 1"},
            {"url": "http://example2.com", "title": "Article 2"},
            {"url": "http://example3.com", "title": "Article 3"},
        ]

        extract_urls = extract_field_from_list("url")
        result = extract_urls(items)

        expected = ["http://example1.com", "http://example2.com", "http://example3.com"]
        assert result == expected

    def test_extract_field_from_list_missing_field(self) -> None:
        """extract_field_from_list handles missing fields gracefully."""
        items = [
            {"url": "http://example1.com", "title": "Article 1"},
            {"title": "Article 2"},  # Missing url field
            {"url": "http://example3.com", "title": "Article 3"},
        ]

        extract_urls = extract_field_from_list("url")
        result = extract_urls(items)

        expected = ["http://example1.com", "http://example3.com"]
        assert result == expected

    def test_extract_field_from_list_invalid_input(self) -> None:
        """extract_field_from_list raises error for non-list input."""
        extract_urls = extract_field_from_list("url")

        with pytest.raises(ValueError, match="Input must be a list"):
            extract_urls("not_a_list")  # pyright: ignore

    def test_extract_field_from_list_complex_fields(self) -> None:
        """extract_field_from_list works with complex field values."""
        items: List[Dict[str, Dict[str, Any]]] = [
            {"metadata": {"author": "Alice", "tags": ["python", "testing"]}},
            {"metadata": {"author": "Bob", "tags": ["javascript"]}},
            {"metadata": {"author": "Charlie", "tags": []}},
        ]

        extract_metadata = extract_field_from_list("metadata")
        result = extract_metadata(items)

        assert len(result) == 3
        assert result[0]["author"] == "Alice"
        assert result[1]["tags"] == ["javascript"]
        assert result[2]["tags"] == []

    def test_combine_text_fields_default_separator(self) -> None:
        """combine_text_fields uses default separator and field names."""
        items = [
            {"title": "Introduction", "content": "This is the introduction section."},
            {"title": "Main Content", "content": "This is the main content section."},
            {"title": "Conclusion", "content": "This is the conclusion section."},
        ]

        combiner = combine_text_fields()
        result = combiner(items)

        expected_parts = [
            "Title: Introduction\nContent: This is the introduction section.",
            "Title: Main Content\nContent: This is the main content section.",
            "Title: Conclusion\nContent: This is the conclusion section.",
        ]
        expected = "\n\n".join(expected_parts)
        assert result == expected

    def test_combine_text_fields_custom_parameters(self) -> None:
        """combine_text_fields works with custom separator and field names."""
        items = [
            {"name": "Section A", "body": "Content of section A"},
            {"name": "Section B", "body": "Content of section B"},
        ]

        combiner = combine_text_fields(
            separator=" | ", title_field="name", content_field="body"
        )
        result = combiner(items)

        expected = "Title: Section A\nContent: Content of section A | Title: Section B\nContent: Content of section B"
        assert result == expected

    def test_combine_text_fields_missing_fields(self) -> None:
        """combine_text_fields handles missing title/content fields."""
        items = [
            {"title": "Has Title", "content": "Has content"},
            {"content": "Missing title"},  # No title
            {"title": "Missing content"},  # No content - will be skipped
            {},  # Both missing - will be skipped
        ]

        combiner = combine_text_fields()
        result = combiner(items)

        # Only items with content are included
        expected_parts = [
            "Title: Has Title\nContent: Has content",
            "Title: Item 2\nContent: Missing title",  # Item index is 1-based, so item 2
        ]
        expected = "\n\n".join(expected_parts)
        assert result == expected

    def test_combine_text_fields_invalid_input(self) -> None:
        """combine_text_fields raises error for non-list input."""
        combiner = combine_text_fields()

        with pytest.raises(ValueError, match="Input must be a list"):
            combiner("not_a_list")  # pyright: ignore

    def test_filter_by_condition_basic(self) -> None:
        """filter_by_condition filters items based on condition."""
        items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        filter_even = filter_by_condition(lambda x: x % 2 == 0)
        result = filter_even(items)

        assert result == [2, 4, 6, 8, 10]

    def test_filter_by_condition_complex_objects(self) -> None:
        """filter_by_condition works with complex objects."""
        items = [
            {"name": "Alice", "age": 25, "active": True},
            {"name": "Bob", "age": 17, "active": False},
            {"name": "Charlie", "age": 30, "active": True},
            {"name": "David", "age": 16, "active": True},
        ]

        filter_active_adults = filter_by_condition(
            lambda person: person["active"] and person["age"] >= 18
        )
        result = filter_active_adults(items)

        expected = [
            {"name": "Alice", "age": 25, "active": True},
            {"name": "Charlie", "age": 30, "active": True},
        ]
        assert result == expected

    def test_filter_by_condition_empty_result(self) -> None:
        """filter_by_condition handles case where no items match."""
        items = [1, 3, 5, 7, 9]  # All odd numbers

        filter_even = filter_by_condition(lambda x: x % 2 == 0)
        result = filter_even(items)

        assert result == []

    def test_filter_by_condition_invalid_input(self) -> None:
        """filter_by_condition raises error for non-list input."""
        filter_func = filter_by_condition(lambda x: True)

        with pytest.raises(ValueError, match="Input must be a list"):
            filter_func("not_a_list")  # pyright: ignore

    def test_aggregate_numeric_field_sum(self) -> None:
        """aggregate_numeric_field calculates sum correctly."""
        items = [
            {"price": 10.5, "quantity": 2},
            {"price": 25.0, "quantity": 1},
            {"price": 15.75, "quantity": 3},
        ]

        sum_prices = aggregate_numeric_field("price", "sum")
        result = sum_prices(items)

        assert result == 51.25

    def test_aggregate_numeric_field_average(self) -> None:
        """aggregate_numeric_field calculates average correctly."""
        items = [{"score": 85}, {"score": 92}, {"score": 78}, {"score": 90}]

        avg_score = aggregate_numeric_field("score", "avg")
        result = avg_score(items)

        assert result == 86.25

    def test_aggregate_numeric_field_min_max(self) -> None:
        """aggregate_numeric_field calculates min and max correctly."""
        items = [
            {"temperature": 23.5},
            {"temperature": 31.2},
            {"temperature": 18.7},
            {"temperature": 27.1},
        ]

        min_temp = aggregate_numeric_field("temperature", "min")
        max_temp = aggregate_numeric_field("temperature", "max")

        assert min_temp(items) == 18.7
        assert max_temp(items) == 31.2

    def test_aggregate_numeric_field_count(self) -> None:
        """aggregate_numeric_field counts items correctly."""
        items = [
            {"value": 1},
            {"value": 2},
            {"other": "ignored"},  # No value field
            {"value": 3},
        ]

        count_values = aggregate_numeric_field("value", "count")
        result = count_values(items)

        assert result == 3.0  # Only 3 items have the value field

    def test_aggregate_numeric_field_empty_values(self) -> None:
        """aggregate_numeric_field handles empty values correctly."""
        items = [
            {"name": "item1"},
            {"name": "item2"},
        ]  # No items have the numeric field

        sum_missing = aggregate_numeric_field("missing_field", "sum")
        result = sum_missing(items)

        assert result == 0.0

    def test_aggregate_numeric_field_invalid_operation(self) -> None:
        """aggregate_numeric_field raises error for unknown operation."""
        items = [{"value": 1}]

        invalid_op = aggregate_numeric_field("value", "unknown_op")

        with pytest.raises(ValueError, match="Unknown operation: unknown_op"):
            invalid_op(items)

    def test_aggregate_numeric_field_invalid_input(self) -> None:
        """aggregate_numeric_field raises error for non-list input."""
        sum_func = aggregate_numeric_field("value", "sum")

        with pytest.raises(ValueError, match="Input must be a list"):
            sum_func("not_a_list")  # pyright: ignore

    def test_format_as_markdown_list_with_urls(self) -> None:
        """format_as_markdown_list creates proper markdown with URLs."""
        items = [
            {"title": "Article 1", "url": "http://example1.com"},
            {"title": "Article 2", "url": "http://example2.com"},
            {"title": "Article 3", "url": "http://example3.com"},
        ]

        formatter = format_as_markdown_list()
        result = formatter(items)

        expected_lines = [
            "- [Article 1](http://example1.com)",
            "- [Article 2](http://example2.com)",
            "- [Article 3](http://example3.com)",
        ]
        expected = "\n".join(expected_lines)
        assert result == expected

    def test_format_as_markdown_list_without_urls(self) -> None:
        """format_as_markdown_list handles missing URLs."""
        items = [
            {"title": "Article 1", "url": "http://example1.com"},
            {"title": "Article 2"},  # No URL
            {"title": "Article 3", "url": None},  # URL is None
        ]

        formatter = format_as_markdown_list()
        result = formatter(items)

        expected_lines = [
            "- [Article 1](http://example1.com)",
            "- Article 2",
            "- Article 3",
        ]
        expected = "\n".join(expected_lines)
        assert result == expected

    def test_format_as_markdown_list_custom_fields(self) -> None:
        """format_as_markdown_list works with custom field names."""
        items = [
            {"name": "Resource 1", "link": "http://resource1.com"},
            {"name": "Resource 2", "link": "http://resource2.com"},
        ]

        formatter = format_as_markdown_list(title_field="name", url_field="link")
        result = formatter(items)

        expected_lines = [
            "- [Resource 1](http://resource1.com)",
            "- [Resource 2](http://resource2.com)",
        ]
        expected = "\n".join(expected_lines)
        assert result == expected

    def test_format_as_markdown_list_missing_titles(self) -> None:
        """format_as_markdown_list handles missing titles."""
        items = [
            {"title": "Has Title", "url": "http://example1.com"},
            {"url": "http://example2.com"},  # No title
            {},  # Both missing
        ]

        formatter = format_as_markdown_list()
        result = formatter(items)

        expected_lines = [
            "- [Has Title](http://example1.com)",
            "- [Untitled](http://example2.com)",
            "- Untitled",
        ]
        expected = "\n".join(expected_lines)
        assert result == expected

    def test_format_as_markdown_list_invalid_input(self) -> None:
        """format_as_markdown_list raises error for non-list input."""
        formatter = format_as_markdown_list()

        with pytest.raises(ValueError, match="Input must be a list"):
            formatter("not_a_list")  # pyright: ignore


class TestTransformTaskIntegration:
    """Test TransformTask integration with framework components."""

    def test_transform_task_with_dependencies_and_context(self) -> None:
        """TransformTask works correctly with dependencies and context."""
        context = TaskContext()

        # Store source data in context
        context.store_result(
            "search",
            {
                "results": [
                    {"title": "Article 1", "score": 0.95},
                    {"title": "Article 2", "score": 0.87},
                    {"title": "Article 3", "score": 0.92},
                ]
            },
        )

        # Create transformation function
        def extract_high_quality_titles(results: List[Dict[str, Any]]) -> List[str]:
            return [r["title"] for r in results if r["score"] > 0.9]

        # Create task with dependencies
        dependencies = [
            TaskDependency("search_results", "search.results", DependencyType.REQUIRED)
        ]

        task = TransformTask(
            "extract_quality",
            {
                "transform_function": extract_high_quality_titles,
                "input_field": "search_results",
                "output_name": "quality_titles",
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
        assert result.is_success()
        # Verify transformation worked with resolved dependencies
        assert result.data["quality_titles"] == ["Article 1", "Article 3"]
        assert result.data["transformation"] == "extract_quality"

    def test_transform_task_with_utility_functions_integration(self) -> None:
        """TransformTask works with utility transformation functions."""
        context = TaskContext()

        # Store complex data structure
        articles = [
            {
                "title": "Python Tips",
                "content": "Learn Python efficiently",
                "url": "http://py1.com",
            },
            {
                "title": "JavaScript Guide",
                "content": "Master JS fundamentals",
                "url": "http://js1.com",
            },
            {
                "title": "Data Science",
                "content": "Explore data insights",
                "url": "http://ds1.com",
            },
        ]

        context.store_result("articles", {"data": articles})

        # Create task using utility function
        markdown_formatter = format_as_markdown_list(
            title_field="title", url_field="url"
        )

        dependencies = [
            TaskDependency("article_data", "articles.data", DependencyType.REQUIRED)
        ]

        task = TransformTask(
            "format_articles",
            {
                "transform_function": markdown_formatter,
                "input_field": "article_data",
                "output_name": "markdown_list",
            },
            dependencies,
        )

        # Simulate dependency resolution
        from aipype import DependencyResolver

        resolver = DependencyResolver(context)
        resolved_config = resolver.resolve_dependencies(task)
        task.config.update(resolved_config)

        result = task.run()
        assert result.is_success()
        # Verify utility function integration
        expected_lines = [
            "- [Python Tips](http://py1.com)",
            "- [JavaScript Guide](http://js1.com)",
            "- [Data Science](http://ds1.com)",
        ]
        expected = "\n".join(expected_lines)
        assert result.data["markdown_list"] == expected

    def test_transform_task_chaining_multiple_transformations(self) -> None:
        """TransformTask can be chained for multi-step transformations."""
        context = TaskContext()

        # Initial data
        raw_data = [
            {"name": "Item A", "value": 100, "category": "electronics"},
            {"name": "Item B", "value": 50, "category": "books"},
            {"name": "Item C", "value": 150, "category": "electronics"},
            {"name": "Item D", "value": 25, "category": "books"},
        ]

        context.store_result("raw_data", {"items": raw_data})

        # First transformation: filter electronics
        filter_electronics = filter_by_condition(
            lambda item: item["category"] == "electronics"
        )

        task1 = TransformTask(
            "filter_electronics",
            {
                "transform_function": filter_electronics,
                "input_field": "raw_items",
                "output_name": "electronics_items",
            },
            [TaskDependency("raw_items", "raw_data.items", DependencyType.REQUIRED)],
        )

        # Simulate first transformation
        from aipype import DependencyResolver

        resolver = DependencyResolver(context)
        resolved_config1 = resolver.resolve_dependencies(task1)
        task1.config.update(resolved_config1)

        result1 = task1.run()

        # Store intermediate result
        context.store_result("filtered_data", result1.data)

        # Second transformation: extract values and sum
        sum_values = aggregate_numeric_field("value", "sum")

        task2 = TransformTask(
            "sum_values",
            {
                "transform_function": sum_values,
                "input_field": "electronics_list",
                "output_name": "total_value",
            },
            [
                TaskDependency(
                    "electronics_list",
                    "filtered_data.electronics_items",
                    DependencyType.REQUIRED,
                )
            ],
        )

        # Simulate second transformation
        resolved_config2 = resolver.resolve_dependencies(task2)
        task2.config.update(resolved_config2)

        result2 = task2.run()

        # Verify chained transformation
        assert result2.data["total_value"] == 250.0  # 100 + 150

    def test_transform_task_error_propagation_in_pipeline(self) -> None:
        """TransformTask errors propagate correctly in pipeline context."""
        context = TaskContext()

        def failing_transform(data: Any) -> Any:
            raise ValueError("Intentional transformation failure")

        context.store_result("source", {"data": "test"})

        task = TransformTask(
            "failing_task",
            {"transform_function": failing_transform, "input_field": "source_data"},
            [TaskDependency("source_data", "source.data", DependencyType.REQUIRED)],
        )

        # Simulate dependency resolution
        from aipype import DependencyResolver

        resolver = DependencyResolver(context)
        resolved_config = resolver.resolve_dependencies(task)
        task.config.update(resolved_config)

        # Verify error propagation
        result = task.run()
        assert result.is_error()
        assert result.error is not None and (
            "Transformation failed: Intentional transformation failure" in result.error
        )

    def test_transform_task_logging_integration(self) -> None:
        """TransformTask logging works correctly."""
        with patch("aipype.base_task.setup_logger") as mock_setup_logger:
            mock_logger = Mock()
            mock_setup_logger.return_value = mock_logger

            def working_transform(x: str) -> str:
                return x.upper()

            task = TransformTask(
                "logging_test",
                {
                    "transform_function": working_transform,
                    "input_field": "text",
                    "text": "hello",
                },
            )

            task.run()

            # Verify logger was set up for the task
            mock_setup_logger.assert_called_once_with("task.logging_test")

            # Task should complete without logging errors (no error calls)
            assert not mock_logger.error.called
