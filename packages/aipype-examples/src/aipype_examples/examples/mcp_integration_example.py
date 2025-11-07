"""Example demonstrating Model Context Protocol (MCP) integration with LLMTask.

This example shows how to:
1. Use MCP servers as external tools with Claude (Anthropic)
2. Mix Python tools with MCP servers
3. Configure MCP tool restrictions with allowed_tools

## Why Claude + NGrok?

This example uses **Claude (Anthropic)** with **NGrok** to expose MCP servers:
- **Claude has native MCP support** built into the API (not via LiteLLM proxy)
- **Anthropic blocks localhost MCP servers** for security, requiring publicly accessible URLs
- **NGrok** provides a secure tunnel to expose local MCP servers over HTTPS
- **Environment variable** (FILE_READER_MCP_URL) allows flexible NGrok URL configuration

## How It Works

MCP (Model Context Protocol) allows LLMs to access external tools through a
standardized protocol. With Anthropic's native MCP support:
1. Claude fetches available tools from the MCP server URL
2. Tools are presented to the model alongside the prompt
3. Model decides when to call tools based on the prompt
4. Tool results are returned and incorporated into the response

## Requirements

- **ANTHROPIC_API_KEY**: Set your Anthropic API key
- **FILE_READER_MCP_URL**: MCP server endpoint (NGrok URL or local for tests)
- **FastMCP file reader server**: Auto-started by integration tests, or:
  ```bash
  python packages/aipype-examples/src/aipype_examples/mcp_servers/file_reader_server.py
  ```
- **NGrok** (for production/internet-accessible MCP):
  ```bash
  ngrok http 8765
  ```

## Running This Example

```bash
# Terminal 1: Start MCP server (binds to 0.0.0.0 for NGrok forwarding)
python packages/aipype-examples/src/aipype_examples/mcp_servers/file_reader_server.py

# Terminal 2: Start NGrok tunnel
ngrok http 8765
# Copy the HTTPS forwarding URL (e.g., https://abc123.ngrok-free.app)

# Terminal 3: Set environment variables and run
export ANTHROPIC_API_KEY=your_api_key_here
export FILE_READER_MCP_URL=https://abc123.ngrok-free.app/sse
python packages/aipype-examples/src/aipype_examples/examples/mcp_integration_example.py
```

## Security Note

The MCP server restricts file access to the `data/` subdirectory only and
prevents path traversal attacks. Only files within the designated data
directory can be read, making it safe to expose over NGrok.
"""

import os
import sys
from typing import List, override

from aipype import BaseTask, LLMTask, PipelineAgent
from aipype.tools import tool


# Example 1: Simple Python tool for comparison
@tool
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2")

    Returns:
        Result of the calculation
    """
    try:
        # Note: eval() is used for simplicity - use a safer parser in production
        result = eval(expression)  # noqa: S307
        return float(result)
    except Exception:
        return float(0)


class MCPOnlyAgent(PipelineAgent):
    """Agent that uses only MCP servers for tool access.

    This agent uses the FileReader MCP server to read file contents
    and process them with Claude.

    Configuration:
        filename: Name of the file to read
        topic: Topic to analyze in the file content

    Environment Variables:
        FILE_READER_MCP_URL: MCP server endpoint (required)
        ANTHROPIC_API_KEY: Anthropic API key (required)

    Note:
        Uses Claude with NGrok-exposed MCP server for internet accessibility.
    """

    @override
    def setup_tasks(self) -> List[BaseTask]:
        # Get configuration values
        filename = self.config.get("filename", "")
        topic = self.config.get("topic", "")

        if not filename:
            raise ValueError("filename is required in config")
        if not topic:
            raise ValueError("topic is required in config")

        # Get MCP server URL from environment (required)
        mcp_url = os.getenv("FILE_READER_MCP_URL")
        if not mcp_url:
            raise ValueError(
                "FILE_READER_MCP_URL environment variable is required. "
                "Set it to your MCP server endpoint (e.g., https://your-ngrok-url.ngrok-free.app/sse)"
            )

        return [
            LLMTask(
                "analyze_file",
                {
                    "prompt": (
                        "First, list the available files using list_files(). "
                        f"Then read the file '{filename}' using read_file(). "
                        f"After reading the file, analyze its contents and provide insights about {topic}."
                    ),
                    "tools": [
                        {
                            "type": "mcp",
                            "server_label": "file_reader",
                            "server_url": mcp_url,
                            "require_approval": "never",
                        }
                    ],
                    "llm_provider": "anthropic",
                    "llm_model": "claude-sonnet-4-20250514",
                    "max_tokens": 2000,
                },
            )
        ]


class MixedToolsAgent(PipelineAgent):
    """Agent that uses both Python tools and MCP servers.

    Combines local Python calculation tool with MCP file reader.

    Configuration:
        filename: Name of the file to read

    Environment Variables:
        FILE_READER_MCP_URL: MCP server endpoint (required)
        ANTHROPIC_API_KEY: Anthropic API key (required)

    Note:
        Demonstrates mixing Python tools with MCP tools using Claude.
    """

    @override
    def setup_tasks(self) -> List[BaseTask]:
        # Get configuration values
        filename = self.config.get("filename", "")

        if not filename:
            raise ValueError("filename is required in config")

        # Get MCP server URL from environment (required)
        mcp_url = os.getenv("FILE_READER_MCP_URL")
        if not mcp_url:
            raise ValueError(
                "FILE_READER_MCP_URL environment variable is required. "
                "Set it to your MCP server endpoint (e.g., https://your-ngrok-url.ngrok-free.app/sse)"
            )

        return [
            LLMTask(
                "analyze_with_calculations",
                {
                    "prompt": (
                        f"Read the file '{filename}' using read_file(). "
                        "Analyze the content and if you find any mathematical expressions or numbers, "
                        "use the calculate() function to verify or compute results. "
                        "Provide a comprehensive analysis with calculations."
                    ),
                    "tools": [
                        # Python tool - executed locally
                        calculate,
                        # MCP server - executed remotely
                        {
                            "type": "mcp",
                            "server_label": "file_reader",
                            "server_url": mcp_url,
                            "require_approval": "never",
                        },
                    ],
                    "llm_provider": "anthropic",
                    "llm_model": "claude-sonnet-4-20250514",
                    "max_tokens": 3000,
                    "temperature": 0.3,
                },
            )
        ]


class RestrictedMCPToolsAgent(PipelineAgent):
    """Agent that demonstrates MCP tool restrictions with allowed_tools.

    Shows how to restrict which MCP tools are available to the LLM
    using the allowed_tools parameter.

    Configuration:
        file1: Name of the first file to read
        file2: Name of the second file to read

    Environment Variables:
        FILE_READER_MCP_URL: MCP server endpoint (required)
        ANTHROPIC_API_KEY: Anthropic API key (required)

    Note:
        Demonstrates the allowed_tools parameter to restrict which
        MCP tools are available to the LLM.
    """

    @override
    def setup_tasks(self) -> List[BaseTask]:
        # Get configuration values
        file1 = self.config.get("file1", "")
        file2 = self.config.get("file2", "")

        if not file1:
            raise ValueError("file1 is required in config")
        if not file2:
            raise ValueError("file2 is required in config")

        # Get MCP server URL from environment (required)
        mcp_url = os.getenv("FILE_READER_MCP_URL")
        if not mcp_url:
            raise ValueError(
                "FILE_READER_MCP_URL environment variable is required. "
                "Set it to your MCP server endpoint (e.g., https://your-ngrok-url.ngrok-free.app/sse)"
            )

        return [
            LLMTask(
                "analyze_multiple_files",
                {
                    "prompt": (
                        "You have access to a file reader. "
                        f"Read both '{file1}' and '{file2}' files. "
                        "Compare and contrast the contents of both files. "
                        "Note: list_files() is not available, only read_file()."
                    ),
                    "tools": [
                        # MCP server with tool restrictions
                        {
                            "type": "mcp",
                            "server_label": "file_reader_restricted",
                            "server_url": mcp_url,
                            "require_approval": "never",
                            "allowed_tools": [
                                "read_file"
                            ],  # Only allow read_file, not list_files
                        },
                    ],
                    "llm_provider": "anthropic",
                    "llm_model": "claude-sonnet-4-20250514",
                    "max_tokens": 4000,
                },
            )
        ]


def example_mcp_only() -> None:
    """Example: Using only MCP servers."""
    print("=" * 80)
    print("Example 1: MCP-Only Agent (File Reader)")
    print("=" * 80)

    agent = MCPOnlyAgent(
        "file_analyzer",
        config={"filename": "sample_article.txt", "topic": "Model Context Protocol"},
    )
    result = agent.run()

    # Exit immediately if agent fails (like tests do)
    if not (result.is_success() or result.is_partial()):
        print(f"\n✗ Analysis failed: {result.error_message}")
        sys.exit(1)

    if result.is_success():
        print("\n✓ File analysis completed successfully")

        # Get task result from context
        task_result = agent.context.get_result("analyze_file")
        if task_result:
            content = task_result.get("content", "")
            if content:
                print(f"\nResponse: {content[:500]}...")
            else:
                print(
                    "\nNote: Task completed but no text content returned (tool_calls only)"
                )

            # Show tool calls if available
            if "tool_calls" in task_result:
                print("\n📊 MCP Tools used:")
                for tool_call in task_result["tool_calls"]:
                    tool_name = tool_call.get("tool_name", "unknown")
                    success = tool_call.get("success", False)
                    status = "✓" if success else "✗"
                    print(f"  {status} {tool_name}")
        else:
            print("\n⚠️  No task result found in context")


def example_mixed_tools() -> None:
    """Example: Using both Python tools and MCP servers."""
    print("\n" + "=" * 80)
    print("Example 2: Mixed Python + MCP Tools")
    print("=" * 80)

    agent = MixedToolsAgent("hybrid_analyzer", config={"filename": "python_tips.txt"})
    result = agent.run()

    # Exit immediately if agent fails (like tests do)
    if not (result.is_success() or result.is_partial()):
        print(f"\n✗ Analysis failed: {result.error_message}")
        sys.exit(1)

    if result.is_success():
        print("\n✓ Analysis with calculations completed successfully")

        # Get task result from context
        task_result = agent.context.get_result("analyze_with_calculations")
        if task_result:
            content = task_result.get("content", "")
            if content:
                print(f"\nResponse: {content[:500]}...")
            else:
                print(
                    "\nNote: Task completed but no text content returned (tool_calls only)"
                )

            # Check which tools were used
            if "tool_calls" in task_result:
                print("\n📊 Tools used:")
                for tool_call in task_result["tool_calls"]:
                    tool_name = tool_call.get("tool_name", "unknown")
                    success = tool_call.get("success", False)
                    status = "✓" if success else "✗"
                    tool_type = "🐍 Python" if tool_name == "calculate" else "🔌 MCP"
                    print(f"  {status} {tool_type}: {tool_name}")
        else:
            print("\n⚠️  No task result found in context")


def example_restricted_mcp_tools() -> None:
    """Example: Using MCP with tool restrictions."""
    print("\n" + "=" * 80)
    print("Example 3: Restricted MCP Tools (allowed_tools)")
    print("=" * 80)

    agent = RestrictedMCPToolsAgent(
        "restricted_tools_agent",
        config={"file1": "sample_article.txt", "file2": "python_tips.txt"},
    )
    result = agent.run()

    # Exit immediately if agent fails (like tests do)
    if not (result.is_success() or result.is_partial()):
        print(f"\n✗ Comparison failed: {result.error_message}")
        sys.exit(1)

    if result.is_success():
        print("\n✓ Multi-file comparison completed successfully")

        # Get task result from context
        task_result = agent.context.get_result("analyze_multiple_files")
        if task_result:
            content = task_result.get("content", "")
            if content:
                print(f"\nResponse: {content[:500]}...")
            else:
                print(
                    "\nNote: Task completed but no text content returned (tool_calls only)"
                )

            # Show different MCP configurations used
            if "tool_calls" in task_result:
                print("\n📊 MCP Tools used (with restrictions):")
                for tool_call in task_result["tool_calls"]:
                    tool_name = tool_call.get("tool_name", "unknown")
                    success = tool_call.get("success", False)
                    status = "✓" if success else "✗"
                    print(f"  {status} {tool_name}")
        else:
            print("\n⚠️  No task result found in context")


def main() -> None:
    """Run all MCP integration examples."""
    print("\n🚀 MCP Integration Examples")
    print("=" * 80)
    print("These examples demonstrate how to integrate Model Context Protocol servers")
    print(
        "with aipype's LLMTask using Claude (Anthropic) with NGrok-exposed MCP servers."
    )
    print("\n⚠️  Prerequisites:")
    print("  1. ANTHROPIC_API_KEY environment variable set")
    print("  2. FILE_READER_MCP_URL environment variable set (NGrok URL)")
    print("  3. MCP server running and accessible via NGrok tunnel")
    print("\n💡 Setup Instructions:")
    print("  Terminal 1 - Start MCP server:")
    print(
        "     python packages/aipype-examples/src/aipype_examples/mcp_servers/file_reader_server.py"
    )
    print("  Terminal 2 - Start NGrok tunnel:")
    print("     ngrok http 8765")
    print("  Terminal 3 - Export environment variables and run:")
    print("     export ANTHROPIC_API_KEY=your_key_here")
    print("     export FILE_READER_MCP_URL=https://your-ngrok-url.ngrok-free.app/sse")
    print(
        "     python packages/aipype-examples/src/aipype_examples/examples/mcp_integration_example.py"
    )
    print("=" * 80)

    # Run examples
    try:
        example_mcp_only()
        example_mixed_tools()
        example_restricted_mcp_tools()

        print("\n" + "=" * 80)
        print("✓ All examples completed!")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Error running examples: {str(e)}")
        print("\nTroubleshooting:")
        print("  1. Is the FILE_READER_MCP_URL environment variable set?")
        print("  2. Is the ANTHROPIC_API_KEY environment variable set?")
        print("  3. Is the File Reader MCP server running?")
        print("  4. Is NGrok tunnel active and forwarding to port 8765?")
        print("  5. Can you access the MCP server at your NGrok URL?")


if __name__ == "__main__":
    main()
