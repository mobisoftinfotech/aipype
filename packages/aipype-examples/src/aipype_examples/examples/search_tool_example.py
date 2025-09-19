#!/usr/bin/env python3
"""Example demonstrating LLM tool calling with enhanced search functionality.

This example shows how to use the search_with_content tool with LLMTask,
where the LLM can automatically search the web and get content from results.
"""

import os
from typing import Any, Dict, List, override

from aipype import LLMTask, search_with_content, PipelineAgent, BaseTask
from aipype import print_header, print_message_box


class SearchExampleAgent(PipelineAgent):
    """Agent that demonstrates LLM tool calling with search functionality."""

    @override
    def setup_tasks(self) -> List[BaseTask]:
        """Set up the LLM task with search tool."""
        # Get configuration with defaults
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

        task = LLMTask(
            "search_research_task",
            {
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "prompt": prompt,
                "tools": [search_with_content],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

        return [task]


def main() -> None:
    """Demonstrate LLM using search tool for research."""
    print_header("LLM SEARCH TOOL CALLING EXAMPLE")

    # Check for required API keys
    if not os.getenv("SERPER_API_KEY"):
        print("❌ Error: SERPER_API_KEY environment variable is required")
        print("Get your API key from: https://serper.dev/api-key")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable is required")
        print("Get your API key from: https://platform.openai.com/account/api-keys")
        return

    print_message_box(
        "SEARCH TOOL CAPABILITIES",
        [
            "This example shows:",
            "• LLM using search_with_content tool automatically",
            "• Getting both search results and their content",
            "• LLM analyzing the content to provide insights",
        ],
    )

    # Create LLM task with search tool
    task = LLMTask(
        "search_research_task",
        {
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "prompt": """I need to research the latest developments in renewable energy technology in 2025.

Please search for recent information and provide:
1. Key technological breakthroughs or innovations
2. Major industry developments or announcements
3. Notable trends or statistics
4. Important challenges or opportunities

Use the search_with_content tool to find current information, then analyze the content to give me a comprehensive overview.""",
            "tools": [search_with_content],  # LLM can use this tool
            "temperature": 0.3,
            "max_tokens": 1500,
        },
    )

    print("\n🤖 Running LLM task with search tool...")
    print("⏳ The LLM will automatically decide when and how to search...")

    try:
        # Execute the task
        result = task.run()

        print()
        print_header("RESULTS")

        if result.is_success():
            print("\n✅ Task completed successfully!")

            # Show the LLM's response
            response = result.data.get("content", "")
            print("\n🤖 LLM Analysis:")
            print("-" * 40)
            print(response)

            # Show tool calls made
            tool_calls = result.data.get("tool_calls", [])
            if tool_calls:
                print(f"\n🔧 Tool Calls Made: {len(tool_calls)}")
                print("-" * 40)

                for i, call in enumerate(tool_calls, 1):
                    tool_name = call.get("tool_name", "unknown")
                    success = call.get("success", False)

                    if success:
                        tool_result = call.get("result", {})
                        if isinstance(tool_result, dict):
                            # Type-safe extraction with explicit casting
                            # LLM tool results have dynamic structure that cannot be statically typed
                            query: str = str(tool_result.get("query") or "N/A")  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                            # Tool response structure includes total_results count
                            total_results_raw = tool_result.get("total_results") or 0  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
                            total_results: int = int(total_results_raw)  # pyright: ignore[reportUnknownArgumentType]
                            # Tool response structure is dynamic and safe with type guards
                            # Tool response structure is dynamic and safe with type guards
                            content_stats_raw = tool_result.get("content_stats") or {}  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
                            # Content stats structure varies by tool implementation but is handled safely
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

                            # Show a sample of results with content
                            # Search results structure is dynamic from tool response
                            results_raw = tool_result.get("results") or []  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
                            # Results list type varies but is protected by type guards
                            results: List[Dict[str, Any]] = (  # pyright: ignore[reportUnknownVariableType]
                                results_raw if isinstance(results_raw, list) else []
                            )
                            successful_content: List[Dict[str, Any]] = [
                                r
                                for r in results
                                if hasattr(r, "get")
                                and r.get("content_status") == "success"
                            ]

                            if successful_content:
                                print("   Sample result with content:")
                                sample: Dict[str, Any] = successful_content[0]
                                title: str = str(sample.get("title", "No title"))[:60]
                                url: str = str(sample.get("url", "No URL"))
                                content_text: str = str(sample.get("content", ""))
                                content_preview: str = content_text[:150].replace(
                                    "\n", " "
                                )
                                print(f"     • {title}...")
                                print(f"     • {url}")
                                print(f"     • Content: {content_preview}...")
                        else:
                            print(f"{i}. {tool_name}: {tool_result}")
                    else:
                        error = call.get("error", "Unknown error")
                        print(f"{i}. {tool_name}: ❌ ERROR - {error}")
            else:
                print("\n🔧 No tool calls were made")

            # Show token usage
            usage = result.data.get("usage", {})
            if usage:
                print("\n📊 Token Usage:")
                print(f"   Input: {usage.get('prompt_tokens', 0)}")
                print(f"   Output: {usage.get('completion_tokens', 0)}")
                print(f"   Total: {usage.get('total_tokens', 0)}")

        else:
            print(f"\n❌ Task failed: {result.error}")

        print()
        print_message_box(
            "KEY FEATURES DEMONSTRATED",
            [
                "• LLM automatically decided to use the search tool",
                "• Search tool fetched both results and content",
                "• LLM analyzed the content to provide insights",
                "• Tool calling provides structured data back to LLM",
            ],
        )

    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
