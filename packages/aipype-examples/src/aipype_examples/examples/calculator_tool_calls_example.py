"""Simple Tool Calls Example - Minimal demonstration of LLM tool calling.

This example shows the basic pattern for using tool calling with llm() helper:
1. Create functions decorated with @tool
2. Pass tools to llm() helper
3. Run the agent - LLM will call your tools automatically

Prerequisites:
- OpenAI API key: export OPENAI_API_KEY=your-key-here
- Or other LLM provider API key

Run with:
    python examples/calculator_tool_calls_example.py
"""

import logging
from typing import List

from aipype import PipelineAgent, task, llm, tool, LLMTask
from aipype import print_header


# Step 1: Define tool functions with @tool decorator
@tool
def add(a: float, b: float) -> float:
    """Add two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of the numbers
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[TOOL] add() called with a={a}, b={b}")
    result = a + b
    logger.info(f"[TOOL] add() returning {result}")
    return result


@tool
def calculate_average(numbers: List[float]) -> float:
    """Calculate average of a list of numbers.

    Args:
        numbers: List of numbers to average

    Returns:
        Average value
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[TOOL] calculate_average() called with numbers={numbers}")
    if not numbers:
        logger.warning("[TOOL] calculate_average() received empty list, returning 0.0")
        return 0.0
    result = sum(numbers) / len(numbers)
    logger.info(f"[TOOL] calculate_average() returning {result}")
    return result


class CalculatorAgent(PipelineAgent):
    """Agent demonstrating tool calling with @task and llm() helper."""

    @task
    def calculator(self) -> LLMTask:
        """Run calculator task with tools.

        Demonstrates passing @tool decorated functions to llm() helper.
        """
        llm_provider = self.config.get("llm_provider", "openai")
        llm_model = self.config.get("llm_model", "gpt-4o-mini")
        prompt = self.config.get(
            "prompt",
            "Please add 15 and 27, then calculate the average of [10, 20, 30, 40, 50]. Use the available tools. It is very important.",
        )
        temperature = self.config.get("temperature", 0.1)
        max_tokens = self.config.get("max_tokens", 300)

        return llm(
            prompt=prompt,
            model=llm_model,
            provider=llm_provider,
            tools=[add, calculate_average],  # Pass tool functions here
            temperature=temperature,
            max_tokens=max_tokens,
        )


def main() -> None:
    """Demonstrate simple tool calling with @task and llm()."""
    # Configure logging to show tool execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print_header("SIMPLE TOOL CALLS EXAMPLE")

    # Create and run agent
    agent = CalculatorAgent("calculator-agent", {})

    print("\n[TASK] Running LLM task with tools...")

    agent.run()

    # Display results
    result = agent.context.get_result("calculator")
    if result:
        print("\n[SUCCESS] Task completed!")
        print(f"Response: {result.get('content', '')}")

        tool_calls = result.get("tool_calls", [])
        if tool_calls:
            print(f"\n[TOOLS] {len(tool_calls)} tool calls made:")
            for i, call in enumerate(tool_calls):
                name = call.get("tool_name", "unknown")
                if call.get("success", False):
                    print(f"  {i + 1}. {name}: {call.get('result', 'N/A')}")
                else:
                    print(f"  {i + 1}. {name}: ERROR - {call.get('error', 'Unknown')}")
    else:
        print("\n[ERROR] Task failed or no results")

    print()
    print_header("EXAMPLE COMPLETE")


if __name__ == "__main__":
    main()
