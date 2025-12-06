# MCP Servers for aipype Examples

This directory contains Model Context Protocol (MCP) servers used in aipype examples and demonstrations.

## Available Servers

### File Reader Server

**File:** `file_reader_server.py`

A simple MCP server that provides tools to read and list files from a local data directory.

**Tools:**
- `read_file(file_path: str)` - Read and return the contents of a file
- `list_files()` - List all files available in the data directory

**Security:** The server restricts file access to only files within the `data/` subdirectory for security.

## Running the File Reader Server

### Quick Start

```bash
# From the repository root
python packages/aipype-examples/src/aipype_examples/mcp_servers/file_reader_server.py
```

The server will start at: **http://127.0.0.1:8765/sse**

### Using FastMCP CLI

Alternatively, you can use the FastMCP CLI:

```bash
fastmcp run file_reader_server.py:mcp --transport sse --port 8765
```

### Verifying the Server

Once running, you should see output like:

```
================================================================================
🚀 File Reader MCP Server Starting
================================================================================
Transport: SSE (Server-Sent Events)
Host: 127.0.0.1
Port: 8765
Endpoint: http://127.0.0.1:8765/sse
================================================================================

Available tools:
  - read_file(file_path: str) -> str
  - list_files() -> str

Press Ctrl+C to stop the server
================================================================================
```

## Using the Server with aipype

### Basic Usage

```python
from aipype import LLMTask, BasePipelineAgent

class FileReaderAgent(BasePipelineAgent):
    def setup_tasks(self):
        return [
            LLMTask("analyze", {
                "prompt": "Read the file '${filename}' and analyze it",
                "tools": [
                    {
                        "type": "mcp",
                        "server_label": "file_reader",
                        "server_url": "http://127.0.0.1:8765/sse",
                        "require_approval": "never"
                    }
                ],
                "llm_provider": "anthropic",
                "llm_model": "claude-sonnet-4"
            })
        ]

# Run the agent
agent = FileReaderAgent("my_agent")
result = agent.run({"filename": "sample_article.txt"})
```

### Complete Example

See `examples/mcp_integration_example.py` for a complete working example that demonstrates:
- MCP-only tool usage
- Mixed Python + MCP tools
- Multiple MCP configurations

## Test Data

The `data/` directory contains sample files for testing:
- `sample_article.txt` - Article about Model Context Protocol
- `python_tips.txt` - Python performance optimization tips

You can add your own files to this directory for testing.

## Requirements

- Python >= 3.12
- fastmcp >= 2.0.0
- aipype >= 0.1.0

Install dependencies:

```bash
uv sync
```

## Troubleshooting

### Port Already in Use

If port 8765 is already in use, you can change it:

```python
# In file_reader_server.py, change the port in main():
mcp.run(transport="sse", host="127.0.0.1", port=8766)  # Use different port
```

Then update the `server_url` in your aipype configuration.

### Connection Refused

1. Make sure the MCP server is running before starting your aipype agent
2. Check that the URL in your configuration matches the server's endpoint
3. Verify the server shows as running (check terminal output)

### File Access Errors

The server only allows access to files within the `data/` subdirectory. Make sure:
1. Your files are in `mcp_servers/data/`
2. Use relative paths: `"sample_article.txt"` not full paths
3. The files have read permissions

## Creating Your Own MCP Server

To create a custom MCP server:

1. **Create a new Python file:**

```python
from fastmcp import FastMCP

mcp = FastMCP("MyServer", version="1.0.0")

@mcp.tool()
def my_tool(param: str) -> str:
    """Description of what the tool does.

    Args:
        param: Parameter description

    Returns:
        Result description
    """
    # Your tool implementation
    return f"Result for {param}"

if __name__ == "__main__":
    mcp.run(transport="sse", host="127.0.0.1", port=8765)
```

2. **Use it in aipype:**

```python
{
    "type": "mcp",
    "server_label": "my_server",
    "server_url": "http://127.0.0.1:8765/sse",
    "require_approval": "never"
}
```

## Additional Resources

- [FastMCP Documentation](https://gofastmcp.com/)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [aipype MCP Integration Guide](../../CLAUDE.md)
