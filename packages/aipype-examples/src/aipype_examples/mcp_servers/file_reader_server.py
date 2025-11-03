"""Simple MCP server that reads file contents.

This server provides a tool to read file contents and return them to the LLM.
It's used for testing MCP integration in aipype examples.

Usage:
    # Run with SSE transport
    python file_reader_server.py

    # Or use fastmcp CLI
    fastmcp run file_reader_server.py:mcp --transport sse --port 8765

The server will be available at http://localhost:8765/sse
"""

import logging
from pathlib import Path

from fastmcp import FastMCP

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FileReaderMCP")

# Create FastMCP server instance
mcp = FastMCP("FileReader", version="1.0.0")


@mcp.tool()
def read_file(file_path: str) -> str:
    """Read and return the contents of a file.

    Args:
        file_path: Path to the file to read (absolute or relative to data directory)

    Returns:
        The contents of the file as a string

    Raises:
        FileNotFoundError: If the file doesn't exist
        PermissionError: If the file can't be read
    """
    logger.info(f"🔧 Tool called: read_file(file_path='{file_path}')")

    # Get the data directory (relative to this file)
    data_dir = Path(__file__).parent / "data"

    # If path is relative, make it relative to data directory
    path = Path(file_path)
    if not path.is_absolute():
        path = data_dir / file_path

    # Security check: ensure file is within data directory
    try:
        resolved_path = path.resolve()
        data_dir_resolved = data_dir.resolve()

        if not str(resolved_path).startswith(str(data_dir_resolved)):
            error_msg = f"Error: Access denied. Can only read files from data directory: {data_dir}"
            logger.warning(f"❌ Access denied for: {file_path}")
            return error_msg
    except Exception as e:
        error_msg = f"Error: Path resolution failed: {str(e)}"
        logger.error(f"❌ Path resolution failed: {e}")
        return error_msg

    # Read the file
    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            content = f.read()

        logger.info(f"✅ Successfully read file: {path.name} ({len(content)} chars)")
        return f"File: {path.name}\n\n{content}"

    except FileNotFoundError:
        logger.error(f"❌ File not found: {file_path}")
        return f"Error: File not found: {file_path}"
    except PermissionError:
        logger.error(f"❌ Permission denied: {file_path}")
        return f"Error: Permission denied reading file: {file_path}"
    except Exception as e:
        logger.error(f"❌ Failed to read file: {e}")
        return f"Error: Failed to read file: {str(e)}"


@mcp.tool()
def list_files() -> str:
    """List all files available in the data directory.

    Returns:
        A formatted list of available files
    """
    logger.info("🔧 Tool called: list_files()")

    data_dir = Path(__file__).parent / "data"

    if not data_dir.exists():
        logger.warning("❌ Data directory does not exist")
        return "Data directory does not exist"

    try:
        files = [f.name for f in data_dir.iterdir() if f.is_file()]

        if not files:
            logger.info("⚠️  No files found in data directory")
            return "No files found in data directory"

        logger.info(f"✅ Found {len(files)} files: {', '.join(sorted(files))}")
        return "Available files:\n" + "\n".join(f"  - {f}" for f in sorted(files))

    except Exception as e:
        logger.error(f"❌ Error listing files: {e}")
        return f"Error listing files: {str(e)}"


def main() -> None:
    """Run the MCP server with SSE transport."""
    print("=" * 80)
    print("🚀 File Reader MCP Server Starting")
    print("=" * 80)
    print("Transport: SSE (Server-Sent Events)")
    print("Host: 0.0.0.0 (accessible from any network interface)")
    print("Port: 8765")
    print("Local: http://127.0.0.1:8765/sse")
    print("Network: http://0.0.0.0:8765/sse")
    print("=" * 80)
    print("\nAvailable tools:")
    print("  - read_file(file_path: str) -> str")
    print("  - list_files() -> str")
    print("\nTool calls will be logged below:")
    print("Press Ctrl+C to stop the server")
    print("=" * 80 + "\n")

    logger.info("🚀 MCP Server starting on http://0.0.0.0:8765/sse")

    # Run the server with SSE transport (bind to all interfaces for NGrok)
    mcp.run(transport="sse", host="0.0.0.0", port=8765)


if __name__ == "__main__":
    main()
