#!/usr/bin/env python3
"""Example demonstrating LLM tool calling with enhanced search functionality.

This example shows how to use the search_with_content tool with llm() helper,
where the LLM can automatically search the web and get content from results.

Key concepts:
- Using @task decorator with llm() helper
- Passing tools to llm() for tool calling
- LLM automatically decides when to use tools
"""

import os
from typing import Any, Dict, List

from aipype import PipelineAgent, task, llm, search_with_content, LLMTask
from aipype import print_header, print_message_box


class SearchExampleAgent(PipelineAgent):
    """Agent that demonstrates LLM tool calling with search functionality."""

    @task
    def search_research_task(self) -> LLMTask:
        """Execute LLM task with search tool for research."""
        llm_provider = self.config.get("llm_provider", "openai")
        llm_model = self.config.get("llm_model", "gpt-4o-mini")
        prompt = self.config.get(
            "prompt",
            """I need to research the latest developments in renewable energy technology in 2025.

Please search for recent information and provide:
1. Key technological breakthroughs or innovations
2. Major industry developments or announcements
3. Notable trends or statistics
4. Important challenges or opportunities

Use the search_with_content tool to find current information, then analyze the content to give me a comprehensive overview.""",
        )
        temperature = self.config.get("temperature", 0.3)
        max_tokens = self.config.get("max_tokens", 1500)

        return llm(
            prompt=prompt,
            model=llm_model,
            provider=llm_provider,
            tools=[search_with_content],  # LLM can use this tool
            temperature=temperature,
            max_tokens=max_tokens,
        )


def main() -> None:
    """Demonstrate LLM using search tool for research."""
    print_header("LLM SEARCH TOOL CALLING EXAMPLE")

    # Check for required API keys
    if not os.getenv("SERPER_API_KEY"):
        print("Error: SERPER_API_KEY environment variable is required")
        print("Get your API key from: https://serper.dev/api-key")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required")
        print("Get your API key from: https://platform.openai.com/account/api-keys")
        return

    print_message_box(
        "SEARCH TOOL CAPABILITIES",
        [
            "This example shows:",
            "- LLM using search_with_content tool automatically",
            "- Getting both search results and their content",
            "- LLM analyzing the content to provide insights",
        ],
    )

    # Create and run agent
    agent = SearchExampleAgent("search-example", {})

    print("\nRunning LLM task with search tool...")
    print("The LLM will automatically decide when and how to search...")

    try:
        agent.run()

        print()
        print_header("RESULTS")

        # Get results from context
        result = agent.context.get_result("search_research_task")

        if result:
            print("\nTask completed successfully!")

            # Show the LLM's response
            response = result.get("content", "")
            print("\nLLM Analysis:")
            print("-" * 40)
            print(response)

            # Show tool calls made
            tool_calls: List[Dict[str, Any]] = result.get("tool_calls", [])
            if tool_calls:
                print(f"\nTool Calls Made: {len(tool_calls)}")
                print("-" * 40)

                for i, call in enumerate(tool_calls, 1):
                    tool_name = call.get("tool_name", "unknown")
                    success = call.get("success", False)

                    if success:
                        tool_result = call.get("result", {})
                        if isinstance(tool_result, dict):
                            # Tool result structure is dynamic from LLM response
                            query: str = str(tool_result.get("query") or "N/A")  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                            total_results_raw = tool_result.get("total_results") or 0  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
                            total_results: int = int(total_results_raw)  # pyright: ignore[reportUnknownArgumentType]
                            content_stats_raw = tool_result.get("content_stats") or {}  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
                            content_stats: Dict[str, Any] = (  # pyright: ignore[reportUnknownVariableType]
                                content_stats_raw
                                if isinstance(content_stats_raw, dict)
                                else {}
                            )

                            print(f"{i}. {tool_name}")
                            print(f"   Query: {query}")
                            print(f"   Total results found: {total_results:,}")
                            print(
                                f"   Content fetched: {content_stats.get('successful', 0)}/{content_stats.get('attempted', 0)} URLs"
                            )
                        else:
                            print(f"{i}. {tool_name}: {tool_result}")
                    else:
                        error = call.get("error", "Unknown error")
                        print(f"{i}. {tool_name}: ERROR - {error}")
            else:
                print("\nNo tool calls were made")

            # Show token usage
            usage: Dict[str, Any] = result.get("usage", {})
            if usage:
                print("\nToken Usage:")
                print(f"   Input: {usage.get('prompt_tokens', 0)}")
                print(f"   Output: {usage.get('completion_tokens', 0)}")
                print(f"   Total: {usage.get('total_tokens', 0)}")

        else:
            print("\nTask failed or no results found")

        print()
        print_message_box(
            "KEY FEATURES DEMONSTRATED",
            [
                "- LLM automatically decided to use the search tool",
                "- Search tool fetched both results and content",
                "- LLM analyzed the content to provide insights",
                "- Tool calling provides structured data back to LLM",
            ],
        )

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
