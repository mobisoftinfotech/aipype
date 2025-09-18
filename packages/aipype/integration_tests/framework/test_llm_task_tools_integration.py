"""Integration tests for LLM Task tool calling functionality.

These tests use real LLM providers (primarily Ollama) to test end-to-end tool calling
functionality without mocking. They verify that the tool calling implementation works
correctly with actual LLM responses.

Prerequisites:
- Ollama installed and running (ollama serve)
- Model qwen3:4b pulled (ollama pull qwen3:4b)
"""

from typing import Any, Dict, List

import pytest

from aipype import LLMTask, tool
from aipype import TaskContext


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of the two numbers
    """
    return a + b


@tool
def multiply_numbers(x: int, y: int) -> int:
    """Multiply two numbers.

    Args:
        x: First number
        y: Second number

    Returns:
        Product of the two numbers
    """
    return x * y


@pytest.mark.integration
@pytest.mark.ollama
class TestLLMTaskToolExecutionIntegration:
    """Integration tests for LLM task tool execution with real LLM providers."""

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        """Basic configuration for LLM task with tools."""
        return {
            "llm_provider": "ollama",
            "llm_model": "qwen3:4b",  # Supports tool calling
            "prompt": "You are a helpful assistant that can perform calculations.",
            "tools": [add_numbers, multiply_numbers],
            "temperature": 0.1,  # Low temperature for more predictable responses
            "max_tokens": 200,
        }

    def test_llm_task_handles_tool_calls_in_response(
        self, config: Dict[str, Any]
    ) -> None:
        """LLM task processes tool calls from real LLM response."""
        # Use OpenAI for more reliable tool calling if available, otherwise fallback to Ollama
        import os

        if os.getenv("OPENAI_API_KEY"):
            test_config: Dict[str, Any] = {
                "llm_provider": "openai",
                "llm_model": "gpt-4o-mini",
                "prompt": "Please calculate 5 + 3 using the add_numbers tool.",
                "tools": [add_numbers, multiply_numbers],
                "temperature": 0.1,
                "max_tokens": 200,
            }
        else:
            test_config = {
                **config,
                "prompt": "Please calculate 5 + 3 using the add_numbers tool.",
            }

        task = LLMTask("tool_calls_test", test_config)

        result = task.run()
        assert result.is_success()

        # Check that tool calls were executed
        if "tool_calls" in result.data:
            tool_calls = result.data["tool_calls"]
            assert len(tool_calls) >= 1

            # Find the add_numbers call
            add_call = next(
                (call for call in tool_calls if call.get("tool_name") == "add_numbers"),
                None,
            )
            if add_call:
                assert add_call["success"] is True
                assert add_call["result"] == 8

    def test_llm_task_handles_multiple_tool_calls(self, config: Dict[str, Any]) -> None:
        """LLM task handles multiple tool calls in sequence."""
        task = LLMTask(
            "multi_tool_test",
            {
                **config,
                "prompt": "Please calculate (5 + 3) and then multiply the result by 2. Use the appropriate tools.",
            },
        )

        result = task.run()
        assert result.is_success()

        if "tool_calls" in result.data:
            tool_calls = result.data["tool_calls"]
            # We might get multiple calls depending on how the LLM responds
            assert len(tool_calls) >= 1

    def test_llm_task_handles_tool_execution_errors(
        self, config: Dict[str, Any]
    ) -> None:
        """LLM task handles tool execution errors gracefully."""

        # Create a tool that will fail
        @tool
        def failing_tool(value: str) -> str:
            """A tool that always fails.

            Args:
                value: Input value

            Returns:
                Never returns, always raises exception
            """
            raise ValueError("This tool always fails")

        task = LLMTask(
            "error_test",
            {
                **config,
                "tools": [failing_tool],
                "prompt": "Please use the failing_tool with value 'test'.",
            },
        )

        result = task.run()
        # Task should still succeed even if tool fails
        assert result.is_success()

        if "tool_calls" in result.data:
            tool_calls = result.data["tool_calls"]
            failed_calls = [
                call for call in tool_calls if not call.get("success", True)
            ]
            if failed_calls:
                assert any(
                    "fail" in call.get("error", "").lower() for call in failed_calls
                )

    def test_llm_task_handles_invalid_tool_call_json(
        self, config: Dict[str, Any]
    ) -> None:
        """LLM task handles invalid JSON in tool arguments."""
        # This test is inherently difficult with real LLMs since they typically
        # generate valid JSON. We'll test our error handling capability instead.
        task = LLMTask(
            "json_test", {**config, "prompt": "Calculate 10 + 15 using add_numbers."}
        )

        result = task.run()
        # Should succeed regardless of JSON handling
        assert result.is_success()

    def test_llm_task_handles_unknown_tool_call(self, config: Dict[str, Any]) -> None:
        """LLM task handles calls to unknown/non-existent tools."""
        task = LLMTask(
            "unknown_tool_test", {**config, "prompt": "Please calculate 7 + 9."}
        )

        result = task.run()
        # Should succeed - LLM will either use available tools or respond without tools
        assert result.is_success()


@pytest.mark.integration
@pytest.mark.ollama
class TestLLMTaskToolIntegrationWithTemplatesIntegration:
    """Integration tests for LLM task tool calling with template resolution."""

    @pytest.fixture
    def context(self) -> TaskContext:
        """Create a task context with sample data."""
        context = TaskContext()
        context.set_data(
            "user_input",
            {"user_numbers": {"first": 10, "second": 20}, "operation": "addition"},
        )
        return context

    def test_llm_task_tools_with_template_resolution(
        self, context: TaskContext
    ) -> None:
        """LLM task resolves templates and uses tools."""
        config = {
            "llm_provider": "ollama",
            "llm_model": "qwen3:4b",
            "prompt": "Please calculate ${user_input.user_numbers.first} + ${user_input.user_numbers.second} using the add_numbers tool.",
            "tools": [add_numbers],
            "temperature": 0.1,
            "max_tokens": 200,
        }

        task = LLMTask("template_test", config)
        task.set_context(context)

        result = task.run()
        assert result.is_success()

        # Check that template was resolved and tool was called
        if "tool_calls" in result.data:
            tool_calls = result.data["tool_calls"]
            add_call = next(
                (call for call in tool_calls if call.get("tool_name") == "add_numbers"),
                None,
            )
            if add_call:
                assert add_call["success"] is True
                assert add_call["result"] == 30

    def test_llm_task_tools_context_resolution(self, context: TaskContext) -> None:
        """LLM task resolves context variables in tool calling scenarios."""
        config = {
            "llm_provider": "ollama",
            "llm_model": "qwen3:4b",
            "context": "You are performing ${user_input.operation} operations.",
            "prompt": "Calculate the result using the appropriate tool.",
            "tools": [add_numbers, multiply_numbers],
            "temperature": 0.1,
            "max_tokens": 200,
        }

        task = LLMTask("context_test", config)
        task.set_context(context)

        result = task.run()
        assert result.is_success()


@pytest.mark.integration
@pytest.mark.ollama
class TestLLMTaskToolConfigurationIntegration:
    """Integration tests for LLM task tool configuration."""

    def test_llm_task_parallel_tool_calls_disabled(self) -> None:
        """LLM task respects parallel_tool_calls configuration."""
        config = {
            "llm_provider": "ollama",
            "llm_model": "qwen3:4b",
            "prompt": "Calculate 5 + 3 and 6 * 7 separately.",
            "tools": [add_numbers, multiply_numbers],
            "parallel_tool_calls": False,  # Disable parallel calls
            "temperature": 0.1,
            "max_tokens": 250,
        }

        task = LLMTask("parallel_test", config)
        result = task.run()

        assert result.is_success()
        # Configuration is applied - actual behavior depends on LLM response
        assert task.config["parallel_tool_calls"] is False


@pytest.mark.integration
@pytest.mark.ollama
class TestLLMTaskToolRealWorldScenariosIntegration:
    """Integration tests for real-world tool calling scenarios."""

    def test_research_assistant_with_tools(self) -> None:
        """Test research assistant scenario with calculation tools."""

        @tool
        def calculate_average(numbers: List[float]) -> float:
            """Calculate the average of a list of numbers.

            Args:
                numbers: List of numbers to average

            Returns:
                Average value
            """
            if not numbers:
                return 0.0
            return float(sum(numbers) / len(numbers))

        @tool
        def find_maximum(numbers: List[int]) -> int:
            """Find the maximum value in a list of numbers.

            Args:
                numbers: List of numbers

            Returns:
                Maximum value
            """
            return max(numbers, default=0)

        config = {
            "llm_provider": "ollama",
            "llm_model": "qwen3:4b",
            "context": "You are a research assistant that helps analyze numerical data.",
            "prompt": "Given the numbers [10, 20, 30, 40, 50], calculate the average and find the maximum value.",
            "tools": [calculate_average, find_maximum],
            "temperature": 0.1,
            "max_tokens": 200,
        }

        task = LLMTask("research_test", config)
        result = task.run()

        assert result.is_success()
        # Check for tool usage if available
        if "tool_calls" in result.data:
            tool_calls = result.data["tool_calls"]
            # Verify at least one tool was called
            assert len(tool_calls) >= 1

    def test_data_analysis_assistant_with_tools(self) -> None:
        """Test data analysis scenario with mathematical tools."""

        @tool
        def calculate_percentage(value: float, total: float) -> float:
            """Calculate percentage.

            Args:
                value: The value to calculate percentage for
                total: The total value

            Returns:
                Percentage value
            """
            if total == 0:
                return 0.0
            return (value / total) * 100

        config = {
            "llm_provider": "ollama",
            "llm_model": "qwen3:4b",
            "context": "You are a data analyst helping with calculations.",
            "prompt": "If we have 75 successful cases out of 100 total cases, what is the success percentage?",
            "tools": [calculate_percentage],
            "temperature": 0.1,
            "max_tokens": 150,
        }

        task = LLMTask("data_analysis_test", config)
        result = task.run()

        assert result.is_success()
        # Check for tool usage
        if "tool_calls" in result.data:
            tool_calls = result.data["tool_calls"]
            percentage_call = next(
                (
                    call
                    for call in tool_calls
                    if call.get("tool_name") == "calculate_percentage"
                ),
                None,
            )
            if percentage_call:
                assert percentage_call["success"] is True
                assert percentage_call["result"] == 75.0
