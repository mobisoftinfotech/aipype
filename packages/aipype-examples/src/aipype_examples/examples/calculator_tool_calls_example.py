"""Simple Tool Calls Example - Minimal demonstration of LLM tool calling.

This example shows the basic pattern for using tool calling with LLMTask:
1. Create functions decorated with @tool
2. Configure LLMTask with tools parameter
3. Run the task - LLM will call your tools automatically

Prerequisites:
- OpenAI API key: export OPENAI_API_KEY=your-key-here
- Or other LLM provider API key

Run with:
    python examples/calculator_tool_calls_example.py
"""

import logging
from typing import List

from aipype import LLMTask, tool
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


def main() -> None:
    """Demonstrate simple tool calling with LLMTask."""
    # Configure logging to show tool execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print_header("SIMPLE TOOL CALLS EXAMPLE")

    # Step 2: Create LLMTask with tools
    task = LLMTask(
        "calculator",
        {
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "prompt": "Please add 15 and 27, then calculate the average of [10, 20, 30, 40, 50]. Use the available tools. It is very important.",
            "tools": [add, calculate_average],  # Pass tool functions here
            "temperature": 0.1,
            "max_tokens": 300,
        },
    )

    print("\n[TASK] Running LLM task with tools...")

    # Step 3: Run the task
    result = task.run()

    # Step 4: Display results
    if result.is_success():
        print("\n[SUCCESS] Task completed!")
        print(f"Response: {result.data['content']}")

        if "tool_calls" in result.data:
            tool_calls = result.data["tool_calls"]
            print(f"\n[TOOLS] {len(tool_calls)} tool calls made:")
            for i, call in enumerate(tool_calls):
                name = call.get("tool_name", "unknown")
                if call.get("success", False):
                    print(f"  {i + 1}. {name}: {call.get('result', 'N/A')}")
                else:
                    print(f"  {i + 1}. {name}: ERROR - {call.get('error', 'Unknown')}")
    else:
        print(f"\n[ERROR] Task failed: {result.error}")

    print()
    print_header("EXAMPLE COMPLETE")


if __name__ == "__main__":
    main()
