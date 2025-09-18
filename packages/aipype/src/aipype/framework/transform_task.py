"""TransformTask - Generic data transformation task."""

from typing import Any, Callable, Dict, List, Optional

from typing import override
from .base_task import BaseTask
from .task_result import TaskResult
from .task_dependencies import TaskDependency
from .task_context import TaskContext


class TransformTask(BaseTask):
    """Task that applies transformations to data from context."""

    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[TaskDependency]] = None,
    ):
        """Initialize transform task.

        Args:
            name: Task name
            config: Task configuration
            dependencies: List of task dependencies

        Config parameters:
        - transform_function: Function to apply transformation
        - input_field: Single input field name (for single-input transforms)
        - input_fields: List of input field names (for multi-input transforms)
        - output_name: Name for the output in the result
        - validate_input: Whether to validate input data (default: True)
        - validate_output: Whether to validate output data (default: False)
        """
        super().__init__(name, config, dependencies)
        self.validation_rules = {
            "required": ["transform_function"],
            "defaults": {
                "output_name": "result",
                "validate_input": True,
                "validate_output": False,
            },
            "types": {
                "output_name": str,
                "validate_input": bool,
                "validate_output": bool,
            },
            # Validation lambdas intentionally use dynamic typing for flexibility across field types
            "custom": {"transform_function": lambda x: callable(x)},  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
        }
        self.context_instance: Optional[TaskContext] = None

    @override
    def set_context(self, context: TaskContext) -> None:
        """Set the task context.

        Args:
            context: TaskContext instance
        """
        self.context_instance = context

    @override
    def run(self) -> TaskResult:
        """Execute the transformation."""
        from datetime import datetime

        start_time = datetime.now()

        # Validate configuration using instance validation rules
        validation_failure = self._validate_or_fail(start_time)
        if validation_failure:
            return validation_failure

        try:
            # Validation has already ensured transform_function is callable
            transform_func = self.config.get("transform_function")
            assert callable(transform_func), (
                "transform_function should be callable after validation"
            )

            # Get input data
            input_data = self._get_input_data()

            # Validate input if requested
            if self.config.get("validate_input", True):
                self._validate_input(input_data)

            # Apply transformation
            try:
                if isinstance(input_data, dict) and len(input_data) == 1:  # pyright: ignore
                    # Single input
                    input_values: List[Any] = list(input_data.values())  # pyright: ignore
                    result = transform_func(input_values[0])
                elif isinstance(input_data, dict):
                    # Multiple inputs - pass as keyword arguments
                    result = transform_func(**input_data)
                else:
                    # Direct input
                    result = transform_func(input_data)

            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                error_msg = f"TransformTask transformation operation failed: Transformation failed: {str(e)}"
                self.logger.error(error_msg)
                # Use explicit cast to ensure type compatibility
                metadata_dict: Dict[str, Any] = {
                    "task_name": self.name,
                    "error_type": type(e).__name__,
                    # Runtime type introspection on flexible Any-typed input data
                    "input_type": type(input_data).__name__,  # pyright: ignore[reportUnknownArgumentType]
                }
                return TaskResult.failure(
                    error_message=error_msg,
                    execution_time=execution_time,
                    metadata=metadata_dict,
                )

            # Validate output if requested
            if self.config.get("validate_output", False):
                self._validate_output(result)

            # Format result
            output_name = self.config.get("output_name", "result")
            formatted_result: Dict[str, Any] = {
                output_name: result,
                "transformation": self.name,
                "input_fields": list(input_data.keys())  # pyright: ignore
                if isinstance(input_data, dict)
                else ["input"],
                "output_type": type(result).__name__,
            }

            execution_time = (datetime.now() - start_time).total_seconds()

            # Use explicit cast to ensure type compatibility
            metadata_dict: Dict[str, Any] = {
                "task_name": self.name,
                "transformation": self.name,
                # Runtime type introspection on flexible Any-typed transformation data
                "input_type": type(input_data).__name__,  # pyright: ignore[reportUnknownArgumentType]
                "output_type": type(result).__name__,
                "output_name": output_name,
            }
            return TaskResult.success(
                data=formatted_result,
                execution_time=execution_time,
                metadata=metadata_dict,
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"TransformTask run operation failed: Transform task '{self.name}' failed: {str(e)}"
            self.logger.error(error_msg)

            return TaskResult.failure(
                error_message=error_msg,
                execution_time=execution_time,
                metadata={"task_name": self.name, "error_type": type(e).__name__},
            )

    def _get_input_data(self) -> Any:
        """Get input data from resolved dependencies.

        Returns:
            Input data for transformation
        """
        # Check for single input field
        input_field = self.config.get("input_field")
        if input_field:
            if input_field not in self.config:
                raise ValueError(
                    f"Input field '{input_field}' not found in resolved config"
                )
            return {input_field: self.config[input_field]}

        # Check for multiple input fields
        input_fields = self.config.get("input_fields")
        if input_fields:
            if not isinstance(input_fields, list):
                raise ValueError("input_fields must be a list")

            input_data: Dict[str, Any] = {}
            for field in input_fields:  # pyright: ignore
                field_str: str = str(field)  # pyright: ignore
                if field_str not in self.config:
                    raise ValueError(
                        f"Input field '{field_str}' not found in resolved config"
                    )
                input_data[field_str] = self.config[field_str]

            return input_data

        # If no specific input fields specified, use all resolved dependency data
        resolved_data: Dict[str, Any] = {}
        for dependency in self.dependencies:
            if dependency.name in self.config:
                resolved_data[dependency.name] = self.config[dependency.name]

        if not resolved_data:
            # If no input validation is required, allow None input for static generators
            if not self.config.get("validate_input", True):
                return None
            raise ValueError("No input data available for transformation")

        return resolved_data

    @property
    def input_data(self) -> Any:
        """Public readonly property to access input data for testing purposes.

        Returns:
            Input data for transformation
        """
        return self._get_input_data()

    def _validate_input(self, input_data: Any) -> None:
        """Validate input data.

        Args:
            input_data: Input data to validate
        """
        if input_data is None:
            raise ValueError("Input data cannot be None")

        # Additional validation can be added based on requirements
        input_validator = self.config.get("input_validator")
        if input_validator and callable(input_validator):
            try:
                if not input_validator(input_data):
                    raise ValueError("Input validation failed")
            except Exception as e:
                raise ValueError(f"Input validation error: {str(e)}")

    def _validate_output(self, output_data: Any) -> None:
        """Validate output data.

        Args:
            output_data: Output data to validate
        """
        output_validator = self.config.get("output_validator")
        if output_validator and callable(output_validator):
            try:
                if not output_validator(output_data):
                    raise ValueError("Output validation failed")
            except Exception as e:
                raise ValueError(f"Output validation error: {str(e)}")

    def preview_transformation(self) -> Dict[str, Any]:
        """Preview what the transformation would do with current input.

        Returns:
            Preview information about the transformation
        """
        try:
            input_data = self._get_input_data()

            preview: Dict[str, Any] = {
                "task_name": self.name,
                "input_fields": list(input_data.keys())  # pyright: ignore
                if isinstance(input_data, dict)
                else ["input"],
                "input_types": {},
                "transform_function": str(
                    self.config.get("transform_function", "Not specified")
                ),
                "output_name": self.config.get("output_name", "result"),
            }

            # Add input type information
            if isinstance(input_data, dict):
                for key, value in input_data.items():  # pyright: ignore
                    key_str: str = str(key)  # pyright: ignore
                    preview["input_types"][key_str] = type(value).__name__  # pyright: ignore
                    if isinstance(value, (list, dict)):
                        preview["input_types"][f"{key_str}_size"] = len(value)  # pyright: ignore
            else:
                preview["input_types"]["input"] = type(input_data).__name__

            return preview

        except Exception as e:
            return {"task_name": self.name, "error": f"Preview failed: {str(e)}"}

    @override
    def __str__(self) -> str:
        """String representation of the transform task."""
        dep_count = len(self.dependencies)
        output_name = self.config.get("output_name", "result")

        return f"TransformTask(name='{self.name}', dependencies={dep_count}, output='{output_name}')"


# Common transformation functions


def extract_field_from_list(
    field_name: str,
) -> Callable[[List[Dict[str, Any]]], List[Any]]:
    """Create a transformation function that extracts a field from each item in a list.

    Args:
        field_name: Name of the field to extract

    Returns:
        Transformation function
    """

    def transform(items: List[Dict[str, Any]]) -> List[Any]:
        if not isinstance(items, list):  # pyright: ignore
            raise ValueError("Input must be a list")

        extracted: List[Any] = []
        for item in items:
            if isinstance(item, dict) and field_name in item:  # pyright: ignore
                extracted.append(item[field_name])

        return extracted

    return transform


def combine_text_fields(
    separator: str = "\n\n", title_field: str = "title", content_field: str = "content"
) -> Callable[[List[Dict[str, Any]]], str]:
    """Create a transformation function that combines text from multiple items.

    Args:
        separator: Separator between items
        title_field: Field name for titles
        content_field: Field name for content

    Returns:
        Transformation function
    """

    def transform(items: List[Dict[str, Any]]) -> str:
        if not isinstance(items, list):  # pyright: ignore
            raise ValueError("Input must be a list")

        combined_parts: List[str] = []
        for i, item in enumerate(items):
            if isinstance(item, dict):  # pyright: ignore
                title: str = str(item.get(title_field, f"Item {i + 1}"))
                content: str = str(item.get(content_field, ""))

                if content:
                    combined_parts.append(f"Title: {title}\nContent: {content}")

        return separator.join(combined_parts)

    return transform


def filter_by_condition(
    condition_func: Callable[[Any], bool],
) -> Callable[[List[Any]], List[Any]]:
    """Create a transformation function that filters items by a condition.

    Args:
        condition_func: Function that returns True for items to keep

    Returns:
        Transformation function
    """

    def transform(items: List[Any]) -> List[Any]:
        if not isinstance(items, list):  # pyright: ignore
            raise ValueError("Input must be a list")

        return [item for item in items if condition_func(item)]

    return transform


def aggregate_numeric_field(
    field_name: str, operation: str = "sum"
) -> Callable[[List[Dict[str, Any]]], float]:
    """Create a transformation function that aggregates a numeric field.

    Args:
        field_name: Name of the numeric field
        operation: Aggregation operation (sum, avg, min, max, count)

    Returns:
        Transformation function
    """

    def transform(items: List[Dict[str, Any]]) -> float:
        if not isinstance(items, list):  # pyright: ignore
            raise ValueError("Input must be a list")

        values: List[float] = []
        for item in items:
            if isinstance(item, dict) and field_name in item:  # pyright: ignore
                value: Any = item[field_name]
                if isinstance(value, (int, float)):
                    values.append(float(value))

        if not values:
            return 0.0

        if operation == "sum":
            return sum(values)
        elif operation == "avg":
            return sum(values) / len(values)
        elif operation == "min":
            return float(min(values))
        elif operation == "max":
            return float(max(values))
        elif operation == "count":
            return float(len(values))
        else:
            raise ValueError(f"Unknown operation: {operation}")

    return transform


def format_as_markdown_list(
    title_field: str = "title", url_field: str = "url"
) -> Callable[[List[Dict[str, Any]]], str]:
    """Create a transformation function that formats items as a markdown list.

    Args:
        title_field: Field name for titles
        url_field: Field name for URLs

    Returns:
        Transformation function
    """

    def transform(items: List[Dict[str, Any]]) -> str:
        if not isinstance(items, list):  # pyright: ignore
            raise ValueError("Input must be a list")

        markdown_lines: List[str] = []
        for item in items:
            if isinstance(item, dict):  # pyright: ignore
                title = item.get(title_field, "Untitled")
                url = item.get(url_field)

                if url:
                    markdown_lines.append(f"- [{title}]({url})")
                else:
                    markdown_lines.append(f"- {title}")

        return "\n".join(markdown_lines)

    return transform
