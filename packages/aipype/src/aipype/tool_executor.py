"""Tool execution engine with error handling and logging."""

import logging
import time
from typing import Any, Dict, List, override, cast

from .tool_registry import ToolRegistry

# Tool executor handles dynamic function calling with error handling
# This enables LLMs to call registered tools with proper validation


class ToolExecutor:
    """Handles tool execution with error handling and logging."""

    def __init__(self, tool_registry: ToolRegistry, max_execution_time: float = 30.0):
        """Initialize tool executor.

        Args:
            tool_registry: Registry containing tool implementations
            max_execution_time: Maximum time (seconds) to allow for tool execution
        """
        self.tool_registry = tool_registry
        self.max_execution_time = max_execution_time
        self.logger = logging.getLogger(__name__)

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return result or error.

        Args:
            tool_name: Name of the tool to execute
            arguments: Dictionary of arguments to pass to the tool

        Returns:
            Dictionary with execution result containing:
            - success (bool): Whether execution was successful
            - result (Any): Tool result if successful
            - error (str): Error message if failed
            - execution_time (float): Time taken in seconds
            - tool_name (str): Name of executed tool
            - error_type (str): Type of error if failed
        """
        start_time = time.time()

        # Validate tool exists
        if not self.tool_registry.has_tool(tool_name):
            execution_time = time.time() - start_time
            available_tools = self.tool_registry.list_tool_names()
            error_msg = (
                f"Tool '{tool_name}' not found. Available tools: {available_tools}"
            )
            self.logger.error(error_msg)

            return {
                "success": False,
                "error": error_msg,
                "execution_time": execution_time,
                "tool_name": tool_name,
                "error_type": "ToolNotFound",
            }

        try:
            # Get tool function from registry
            # Tool registry returns decorated functions for execution
            tool_func = self.tool_registry.get_tool_function(tool_name)

            # Log execution start
            self.logger.info(
                f"Executing tool '{tool_name}' with arguments: {arguments}"
            )

            # Validate arguments against tool metadata
            validation_error = self._validate_arguments(tool_name, arguments)
            if validation_error:
                execution_time = time.time() - start_time
                self.logger.error(
                    f"Tool '{tool_name}' argument validation failed: {validation_error}"
                )
                return {
                    "success": False,
                    "error": f"Argument validation failed: {validation_error}",
                    "execution_time": execution_time,
                    "tool_name": tool_name,
                    "error_type": "ArgumentValidationError",
                }

            # Execute the tool function
            # Note: In a production environment, you might want to add timeout handling
            # using signal.alarm() or threading.Timer for more robust execution
            # Tool function execution has dynamic return types
            result = tool_func(**arguments)

            execution_time = time.time() - start_time

            # Check for execution time warning
            if execution_time > self.max_execution_time:
                self.logger.warning(
                    f"Tool '{tool_name}' took {execution_time:.3f}s "
                    f"(exceeded max time of {self.max_execution_time}s)"
                )

            self.logger.info(
                f"Tool '{tool_name}' executed successfully in {execution_time:.3f}s"
            )

            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "tool_name": tool_name,
            }

        except TypeError as e:
            # Handle argument-related errors specifically
            execution_time = time.time() - start_time
            error_msg = f"Tool '{tool_name}' argument error: {str(e)}"
            self.logger.error(error_msg)

            return {
                "success": False,
                "error": error_msg,
                "execution_time": execution_time,
                "tool_name": tool_name,
                "error_type": "ArgumentError",
            }

        except Exception as e:
            # Handle all other execution errors
            execution_time = time.time() - start_time
            error_msg = f"Tool '{tool_name}' execution failed: {str(e)}"
            self.logger.exception(error_msg)  # Use exception() for full traceback

            return {
                "success": False,
                "error": error_msg,
                "execution_time": execution_time,
                "tool_name": tool_name,
                "error_type": type(e).__name__,
            }

    def _validate_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Validate tool arguments against metadata.

        Args:
            tool_name: Name of the tool
            arguments: Arguments to validate

        Returns:
            Empty string if valid, error message if invalid
        """
        try:
            metadata = self.tool_registry.get_tool_metadata(tool_name)

            # Check required parameters
            for param_name, param_info in metadata.parameters.items():
                is_optional = param_info.get("optional", False)

                if not is_optional and param_name not in arguments:
                    return f"Missing required parameter: {param_name}"

            # Check for unexpected parameters
            expected_params = set(metadata.parameters.keys())
            provided_params = set(arguments.keys())
            unexpected_params = provided_params - expected_params

            if unexpected_params:
                return f"Unexpected parameters: {list(unexpected_params)}"

            # Basic type checking
            for param_name, value in arguments.items():
                if param_name in metadata.parameters:
                    param_info = metadata.parameters[param_name]
                    schema = param_info["schema"]
                    expected_type = schema["type"]

                    # Basic type validation
                    if not self._validate_parameter_type(value, expected_type):
                        return f"Parameter '{param_name}' expected type {expected_type}, got {type(value).__name__}"

            return ""  # No validation errors

        except Exception as e:
            return f"Validation error: {str(e)}"

    def _validate_parameter_type(self, value: Any, expected_type: str) -> bool:
        """Basic type validation for parameters.

        Args:
            value: Value to validate
            expected_type: Expected JSON schema type

        Returns:
            True if type matches expectation, False otherwise
        """
        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        elif expected_type == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        elif expected_type == "boolean":
            return isinstance(value, bool)
        elif expected_type == "array":
            return isinstance(value, list)
        elif expected_type == "object":
            return isinstance(value, dict)
        else:
            # For unknown types, accept any value
            return True

    def execute_multiple_tools(self, tool_calls: List[Any]) -> List[Dict[str, Any]]:
        """Execute multiple tools in sequence.

        Args:
            tool_calls: List of tool call objects (expected to be dictionaries with 'name' and 'arguments')

        Returns:
            List of execution results for each tool call
        """
        results: List[Dict[str, Any]] = []

        for i, tool_call in enumerate(tool_calls):
            try:
                # Validate tool call structure first
                # Tool call validation for LLM function calling format
                if not isinstance(tool_call, dict):
                    error_msg = f"Error processing tool call {i}: Expected dictionary, got {type(tool_call).__name__}"
                    # Tool execution result structure for error handling
                    error_result: Dict[str, Any] = {
                        "success": False,
                        "error": error_msg,
                        "execution_time": 0.0,
                        "tool_name": "unknown",
                        "error_type": "InvalidToolCall",
                    }
                    results.append(error_result)
                    continue

                # Extract tool information from call dictionary
                # Type narrowing: tool_call is confirmed to be dict after isinstance check
                dict_call = cast(Dict[str, Any], tool_call)
                tool_name = dict_call.get("name")
                arguments = dict_call.get("arguments", {})

                if not tool_name:
                    # Tool call validation error result
                    name_error_result: Dict[str, Any] = {
                        "success": False,
                        "error": f"Tool call {i} missing 'name' field",
                        "execution_time": 0.0,
                        "tool_name": "unknown",
                        "error_type": "InvalidToolCall",
                    }
                    results.append(name_error_result)
                    continue

                result = self.execute_tool(tool_name, arguments)
                # Tool execution results collected for batch processing
                results.append(result)

            except Exception as e:
                error_msg = f"Error processing tool call {i}: {str(e)}"
                self.logger.error(error_msg)
                # Extract tool name safely for error reporting
                tool_name_for_error = "unknown"
                if isinstance(tool_call, dict):
                    dict_call = cast(Dict[str, Any], tool_call)
                    tool_name_for_error = dict_call.get("name", "unknown") or "unknown"

                # Tool execution exception result
                exception_result: Dict[str, Any] = {
                    "success": False,
                    "error": error_msg,
                    "execution_time": 0.0,
                    "tool_name": tool_name_for_error,
                    "error_type": type(e).__name__,
                }
                results.append(exception_result)

        # Return list of tool execution results
        return results

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics for monitoring.

        Returns:
            Dictionary with execution statistics
        """
        # This is a simple implementation. In a production system,
        # you might want to track more detailed statistics
        return {
            "max_execution_time": self.max_execution_time,
            "registered_tools": self.tool_registry.get_tool_count(),
            "available_tools": self.tool_registry.list_tool_names(),
        }

    @override
    def __str__(self) -> str:
        """String representation of the tool executor."""
        return f"ToolExecutor(max_time={self.max_execution_time}s, tools={self.tool_registry.get_tool_count()})"
