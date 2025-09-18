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

from typing import List, override

from aipype import (
    PipelineAgent,
    BaseTask,
    LLMTask,
    SearchTask,
    TaskDependency,
    DependencyType,
    print_header,
)


class SimpleDisplayDemo(PipelineAgent):
    """Demo agent showing simple display_results usage."""

    @override
    def setup_tasks(self) -> List[BaseTask]:
        """Set up a simple pipeline for demonstration."""
        return [
            SearchTask(
                "search_demo",
                {"query": "Python programming tutorial", "max_results": 3},
            ),
            LLMTask(
                "analyze_search",
                {
                    "context": "You are a content analyst.",
                    "prompt_template": "Briefly analyze these search results: ${search_results}",
                    "llm_provider": "openai",
                    "llm_model": "gpt-4o-mini",
                    "temperature": 0.3,
                    "max_tokens": 200,
                },
                [
                    TaskDependency(
                        "search_results", "search_demo.results", DependencyType.REQUIRED
                    )
                ],
            ),
        ]


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
