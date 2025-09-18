"""Tool registry for managing tool schemas and implementations."""

import logging
from typing import Any, Callable, Dict, List, override

from .tools import ToolMetadata, ToolSchemaGenerator

# Tool registry manages functions decorated with @tool for LLM function calling
# This allows dynamic schema generation and function execution


class ToolRegistry:
    """Manages tool schemas and implementations with automatic generation."""

    def __init__(self, tool_functions: List[Callable[..., Any]]):
        """Initialize tool registry with list of decorated functions.

        Args:
            tool_functions: List of functions decorated with @tool

        Raises:
            ValueError: If any function is not decorated with @tool
        """
        # Tool registry stores decorated functions for schema generation
        self.tool_functions = tool_functions
        self.tools: Dict[str, ToolMetadata] = {}
        self.schemas: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

        self._process_tools()

    def _process_tools(self) -> None:
        """Process tool functions and generate schemas."""
        for func in self.tool_functions:
            # Tool decorator preserves function metadata for validation
            func_name = getattr(func, "__name__", str(func))

            # Validate that tool function is callable first
            # Tool decorator creates callable functions with metadata
            if not callable(func):
                raise ValueError(f"Tool {func_name} is not callable")

            # Validate tool function
            # Tool decorator adds validation attributes to functions
            if not hasattr(func, "_is_tool"):
                raise ValueError(
                    f"Function {func_name} must be decorated with @tool. "
                    f"Add the @tool decorator above the function definition."
                )

            if not hasattr(func, "_tool_metadata"):
                raise ValueError(
                    f"Function {func_name} is missing tool metadata. "
                    f"Ensure it's properly decorated with @tool."
                )

            # Tool decorator attaches ToolMetadata to decorated functions
            metadata = func._tool_metadata  # pyright: ignore[reportFunctionMemberAccess]

            # Validate metadata is a ToolMetadata object
            if not isinstance(metadata, ToolMetadata):
                raise ValueError(
                    f"Function {func_name} has invalid tool metadata. "
                    f"Expected ToolMetadata object, got {type(metadata).__name__}."
                )

            # Check for name conflicts
            if metadata.name in self.tools:
                raise ValueError(
                    f"Duplicate tool name '{metadata.name}'. "
                    f"Tool names must be unique within a registry."
                )

            # Store metadata
            self.tools[metadata.name] = metadata

            # Generate and store OpenAI schema
            try:
                # Tool decorator enables OpenAI schema generation
                schema = ToolSchemaGenerator.generate_schema(func)
                self.schemas.append(schema)
                self.logger.debug(
                    f"Registered tool '{metadata.name}' with {len(metadata.parameters)} parameters"
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to generate schema for tool '{metadata.name}': {str(e)}"
                ) from e

        self.logger.info(f"Successfully registered {len(self.tools)} tools")

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get all tool schemas for LLM API.

        Returns:
            List of OpenAI-compatible tool schemas
        """
        return self.schemas.copy()

    def get_tool_function(self, name: str) -> Callable[..., Any]:
        """Get tool function by name.

        Args:
            name: Name of the tool to retrieve

        Returns:
            Tool function implementation

        Raises:
            ValueError: If tool is not found
        """
        if name not in self.tools:
            available_tools = list(self.tools.keys())
            raise ValueError(
                f"Tool '{name}' not found. Available tools: {available_tools}"
            )
        # Tool registry returns callable functions with preserved signatures
        return self.tools[name].func

    def get_tool_metadata(self, name: str) -> ToolMetadata:
        """Get tool metadata by name.

        Args:
            name: Name of the tool

        Returns:
            ToolMetadata instance

        Raises:
            ValueError: If tool is not found
        """
        if name not in self.tools:
            available_tools = list(self.tools.keys())
            raise ValueError(
                f"Tool '{name}' not found. Available tools: {available_tools}"
            )
        return self.tools[name]

    def has_tool(self, name: str) -> bool:
        """Check if a tool exists in the registry.

        Args:
            name: Name of the tool to check

        Returns:
            True if tool exists, False otherwise
        """
        return name in self.tools

    def list_tool_names(self) -> List[str]:
        """Get list of all registered tool names.

        Returns:
            List of tool names
        """
        return list(self.tools.keys())

    def get_tool_count(self) -> int:
        """Get the number of registered tools.

        Returns:
            Number of tools in registry
        """
        return len(self.tools)

    def generate_tool_context(self) -> str:
        """Generate context description for system prompt.

        Returns:
            Formatted tool context for inclusion in system prompt
        """
        if not self.tools:
            return ""

        context_lines = [
            "# Available Tools",
            "You have access to the following tools:",
            "",
        ]

        for tool_name, metadata in self.tools.items():
            # Add tool name and description
            context_lines.append(f"## {tool_name}")
            context_lines.append(metadata.description)

            # Add parameter information if any
            if metadata.parameters:
                context_lines.append("Parameters:")
                for param_name, param_info in metadata.parameters.items():
                    required_text = (
                        "required"
                        if not param_info.get("optional", False)
                        else "optional"
                    )
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "No description")

                    # Format parameter line
                    param_line = f"  - {param_name} ({param_type}, {required_text}): {param_desc}"

                    # Add default value if present
                    if "default" in param_info:
                        param_line += f" (default: {param_info['default']})"

                    context_lines.append(param_line)
            else:
                context_lines.append("No parameters required.")

            context_lines.append("")  # Empty line between tools

        context_lines.extend(
            [
                "Use these tools when needed to help the user.",
                "Always explain what you're doing when calling tools.",
                "If a tool call fails, continue helping the user as best you can.",
            ]
        )

        return "\n".join(context_lines)

    def validate_all_tools(self) -> Dict[str, bool]:
        """Validate all registered tools.

        Returns:
            Dictionary mapping tool names to validation status
        """
        validation_results = {}

        for tool_name, metadata in self.tools.items():
            try:
                # Check if function is still callable
                # Tool decorator creates callable functions for execution
                if not callable(metadata.func):
                    self.logger.warning(
                        f"Tool '{tool_name}' failed validation: function is not callable"
                    )
                    validation_results[tool_name] = False
                    continue

                # Check if schema can still be generated
                # Tool decorator supports schema validation
                ToolSchemaGenerator.generate_schema(metadata.func)
                validation_results[tool_name] = True

            except Exception as e:
                self.logger.warning(f"Tool '{tool_name}' failed validation: {str(e)}")
                validation_results[tool_name] = False

        # Return validation status for all registered tools
        return validation_results  # pyright: ignore[reportUnknownVariableType]

    @override
    def __str__(self) -> str:
        """String representation of the tool registry."""
        tool_names = list(self.tools.keys())
        return f"ToolRegistry({len(self.tools)} tools: {tool_names})"

    @override
    def __repr__(self) -> str:
        """Detailed string representation of the tool registry."""
        return self.__str__()
