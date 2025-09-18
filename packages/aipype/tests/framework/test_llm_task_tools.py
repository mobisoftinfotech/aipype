"""Comprehensive tests for LLMTask tool calling integration."""

from typing import Any, Dict, List

from aipype import LLMTask, tool, ToolRegistry, ToolExecutor


class TestLLMTaskToolSetup:
    """Test LLMTask tool setup and configuration validation."""

    def create_test_tools(self) -> List[Any]:
        """Create test tools for LLMTask testing."""

        @tool
        def calculate(expression: str, precision: int = 2) -> float:
            """Perform mathematical calculations.

            Args:
                expression: Mathematical expression to evaluate
                precision: Number of decimal places for result
            """
            result = eval(expression)  # Unsafe but OK for tests
            return round(float(result), precision)

        @tool
        def get_weather(city: str, units: str = "metric") -> Dict[str, Any]:
            """Get weather information for a city.

            Args:
                city: City name
                units: Temperature units (metric, imperial)
            """
            return {
                "city": city,
                "temperature": 22.5 if units == "metric" else 72.5,
                "units": units,
                "condition": "sunny",
            }

        return [calculate, get_weather]

    def test_llm_task_tool_setup_success(self) -> None:
        """LLMTask successfully sets up tool support."""
        tools = self.create_test_tools()
        config = {"llm_provider": "openai", "llm_model": "gpt-4", "tools": tools}

        task = LLMTask("tool_test", config)

        assert task.supports_tools is True
        assert task.tool_registry is not None
        assert task.tool_executor is not None
        assert isinstance(task.tool_registry, ToolRegistry)
        assert isinstance(task.tool_executor, ToolExecutor)

    def test_llm_task_without_tools(self) -> None:
        """LLMTask works without tools configured."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4"}

        task = LLMTask("no_tools_test", config)

        assert task.supports_tools is False
        assert task.tool_registry is None
        assert task.tool_executor is None

    def test_llm_task_tool_validation_success(self) -> None:
        """LLMTask validates tools configuration successfully."""
        tools = self.create_test_tools()
        config = {"llm_provider": "openai", "llm_model": "gpt-4", "tools": tools}

        # Should not raise any errors
        task = LLMTask("validation_test", config)
        assert task.supports_tools is True

    def test_llm_task_tool_validation_failure_non_list(self) -> None:
        """LLMTask rejects non-list tools configuration."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4", "tools": "not_a_list"}

        result = LLMTask("invalid_tools_test", config).run()
        assert result.is_error()
        assert result.error and "tools failed custom validation" in result.error

    def test_llm_task_tool_validation_failure_non_callable(self) -> None:
        """LLMTask rejects non-callable items in tools list."""
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "tools": ["not_callable", 123],
        }

        result = LLMTask("invalid_items_test", config).run()
        assert result.is_error()
        assert result.error and "tools failed custom validation" in result.error

    def test_llm_task_tool_validation_failure_undecorated(self) -> None:
        """LLMTask rejects functions not decorated with @tool."""

        def undecorated_function(x: int) -> int:
            return x * 2

        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "tools": [undecorated_function],
        }

        result = LLMTask("undecorated_test", config).run()
        assert result.is_error()
        assert result.error and "tools failed custom validation" in result.error

    def test_llm_task_tool_setup_initialization_error(self) -> None:
        """LLMTask handles tool setup initialization errors."""

        def broken_tool() -> None:
            pass

        # Dynamically set tool attributes to simulate broken tool metadata for testing
        # This mimics what the @tool decorator does but with invalid metadata structure
        broken_tool._is_tool = True  # pyright: ignore[reportFunctionMemberAccess]
        broken_tool._tool_metadata = "invalid_metadata"  # pyright: ignore[reportFunctionMemberAccess]

        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "tools": [broken_tool],
        }

        result = LLMTask("broken_setup_test", config).run()
        assert result.is_error()
        assert result.error and "tool setup operation failed" in result.error

    def test_llm_task_tool_configuration_parameters(self) -> None:
        """LLMTask correctly configures tool-related parameters."""
        tools = self.create_test_tools()
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "max_tool_execution_time": 45.0,
        }

        task = LLMTask("config_test", config)

        assert task.config["tool_choice"] == "auto"
        assert task.config["parallel_tool_calls"] is True
        assert task.config["max_tool_execution_time"] == 45.0
        assert task.tool_executor is not None
        assert task.tool_executor.max_execution_time == 45.0

    def test_llm_task_tool_configuration_defaults(self) -> None:
        """LLMTask applies default values for tool configuration."""
        tools = self.create_test_tools()
        config = {"llm_provider": "openai", "llm_model": "gpt-4", "tools": tools}

        task = LLMTask("defaults_test", config)

        assert task.config["tool_choice"] == "auto"
        assert task.config["parallel_tool_calls"] is True
        assert task.config["max_tool_execution_time"] == 30.0

    def test_llm_task_tool_choice_validation(self) -> None:
        """LLMTask validates tool_choice parameter."""
        tools = self.create_test_tools()

        # Valid tool_choice values
        for choice in ["auto", "none", "calculate"]:
            config = {
                "llm_provider": "openai",
                "llm_model": "gpt-4",
                "tools": tools,
                "tool_choice": choice,
            }
            task = LLMTask(f"choice_test_{choice}", config)
            result = task.run()
            # Should not fail validation (might fail for other reasons like missing API key)
            if result.is_error():
                assert (
                    not result.error
                    or "tool_choice failed custom validation" not in result.error
                )


# Complex integration test classes have been moved to:
# integration_tests/framework/test_llm_task_tools_integration.py
#
# The following test classes were moved:
# - TestLLMTaskToolExecution
# - TestLLMTaskToolIntegrationWithTemplates
# - TestLLMTaskToolConfiguration (failing test only)
# - TestLLMTaskToolRealWorldScenarios
#
# This file now contains only true unit tests that don't require complex LLM mocking.

# Placeholder class to prevent syntax errors - will be removed
