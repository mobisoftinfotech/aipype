"""
Simple Display Demo - demonstrating the framework's simple display_results method
==============================================================================

This example shows how the new framework display_results method simplifies
result display from 40-80 lines of complex code to just 4-5 lines.

Usage examples:
- Show all sections (default): agent.display_results()
- Show only summary: agent.display_results(["summary"])
- Show summary and errors: agent.display_results(["summary", "errors"])
- Show only tasks: agent.display_results(["tasks"])
"""

from typing import Annotated, Any, Dict

from aipype import (
    PipelineAgent,
    task,
    llm,
    search,
    Depends,
    LLMTask,
    SearchTask,
    print_header,
)


class SimpleDisplayDemo(PipelineAgent):
    """Demo agent showing simple display_results usage with declarative syntax."""

    @task
    def search_demo(self) -> SearchTask:
        """Search for Python programming tutorials."""
        return search("Python programming tutorial", max_results=3)

    @task
    def analyze_search(
        self,
        search_demo: Annotated[Dict[str, Any], Depends("search_demo.results")],
    ) -> LLMTask:
        """Analyze search results using LLM."""
        return llm(
            prompt=f"Briefly analyze these search results: {search_demo}",
            model="gpt-4o-mini",
            provider="openai",
            system="You are a content analyst.",
            temperature=0.3,
            max_tokens=200,
        )


def demonstrate_display_options() -> None:
    """Demonstrate different display options."""
    print_header("SIMPLE DISPLAY RESULTS DEMONSTRATION")

    # Create and run agent
    agent = SimpleDisplayDemo(
        "simple-display-demo",
        {
            "enable_parallel": False,  # Sequential for clear demo
        },
    )

    agent.run()

    print()
    print_header("1. DEFAULT DISPLAY (all sections)")
    # Show all sections (default) - just 1 line of code!
    agent.display_results()

    print()
    print_header("2. SUMMARY ONLY")
    # Show only summary - just 1 line of code!
    agent.display_results(["summary"])

    print()
    print_header("3. TASKS ONLY")
    # Show only tasks - just 1 line of code!
    agent.display_results(["tasks"])

    print()
    print_header("4. CUSTOM COMBINATION (summary + errors)")
    # Show summary and errors - just 1 line of code!
    agent.display_results(["summary", "errors"])


if __name__ == "__main__":
    demonstrate_display_options()
