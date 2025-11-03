"""pytest configuration for aipype-examples integration tests."""

import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator, List

import pytest
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def anthropic_available() -> bool:
    """Check if Anthropic API key is available."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))


@pytest.fixture(scope="session")
def ollama_available() -> bool:
    """Check if Ollama service is available."""
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


@pytest.fixture(scope="session")
def test_model_available(ollama_available: bool) -> bool:
    """Check if test model is available in Ollama."""
    if not ollama_available:
        return False

    test_model = os.getenv("INTEGRATION_TEST_MODEL", "qwen3:4b")

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return any(test_model in model.get("name", "") for model in models)
        return False
    except (requests.ConnectionError, requests.Timeout):
        return False


@pytest.fixture
def test_output_dir(tmp_path) -> Any:
    """Provide temporary output directory for tests."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def skip_if_anthropic_unavailable(anthropic_available: bool) -> None:
    """Skip test if Anthropic API key is not available."""
    if not anthropic_available:
        pytest.skip("ANTHROPIC_API_KEY environment variable is not set")


@pytest.fixture
def skip_if_ollama_unavailable(ollama_available: bool) -> None:
    """Skip test if Ollama is not available."""
    if not ollama_available:
        pytest.skip("Ollama service is not available")


@pytest.fixture
def skip_if_test_model_unavailable(test_model_available: bool) -> None:
    """Skip test if test model is not available."""
    if not test_model_available:
        test_model = os.getenv("INTEGRATION_TEST_MODEL", "qwen3:4b")
        pytest.skip(f"{test_model} model is not available in Ollama")


def _is_port_available(host: str = "127.0.0.1", port: int = 8765) -> bool:
    """Check if a port is available (not in use)."""
    try:
        with socket.create_connection((host, port), timeout=1):
            return False  # Port is in use
    except (socket.error, socket.timeout):
        return True  # Port is available


def _wait_for_server(
    host: str = "127.0.0.1", port: int = 8765, timeout: float = 10.0
) -> bool:
    """Wait for server to become available on the specified port.

    Args:
        host: Server host
        port: Server port
        timeout: Maximum time to wait in seconds

    Returns:
        True if server became available, False otherwise
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                logger.info(f"MCP server is ready on {host}:{port}")
                return True
        except (socket.error, socket.timeout):
            time.sleep(0.5)

    logger.error(f"MCP server failed to start within {timeout}s")
    return False


@pytest.fixture(scope="session")
def mcp_server() -> Iterator[bool]:
    """Start MCP file reader server for tests.

    Automatically starts the MCP server in a background process and
    ensures it's ready before running tests. Cleans up on exit.

    Note: Does NOT set FILE_READER_MCP_URL. Tests require this to be
    set to a publicly accessible URL (NGrok) because Anthropic blocks localhost.

    Yields:
        True if server started successfully, False otherwise
    """
    # Check if FILE_READER_MCP_URL is set (required for MCP tests)
    mcp_url = os.getenv("FILE_READER_MCP_URL")
    if mcp_url:
        logger.info(f"FILE_READER_MCP_URL is set to: {mcp_url}")
    else:
        logger.warning(
            "FILE_READER_MCP_URL is not set. MCP tests will skip. "
            "Set to NGrok URL for MCP tests to run."
        )

    # Check if server is already running
    if not _is_port_available():
        logger.info("MCP server already running on port 8765")
        yield True
        return

    # Find the MCP server script
    server_script = (
        Path(__file__).parent.parent
        / "src"
        / "aipype_examples"
        / "mcp_servers"
        / "file_reader_server.py"
    )

    if not server_script.exists():
        logger.error(f"MCP server script not found at {server_script}")
        yield False
        return

    # Start the MCP server process
    logger.info(f"Starting MCP server from {server_script}")
    process = None

    try:
        # Start server in background
        process = subprocess.Popen(
            [sys.executable, str(server_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for server to be ready
        if _wait_for_server(timeout=15.0):
            logger.info("MCP server started successfully")
            yield True
        else:
            # Server failed to start - get error output
            if process.poll() is not None:
                _, stderr = process.communicate(timeout=1)
                logger.error(f"MCP server failed to start. Error: {stderr}")
            else:
                logger.error("MCP server started but is not responding")

            yield False

    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        yield False

    finally:
        # Cleanup: terminate the server process
        if process and process.poll() is None:
            logger.info("Shutting down MCP server")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("MCP server did not terminate gracefully, killing it")
                process.kill()
                process.wait()


@pytest.fixture
def skip_if_mcp_unavailable(mcp_server: bool) -> None:
    """Skip test if MCP server is not available or URL is localhost.

    Note: Anthropic blocks localhost MCP servers for security.
    Tests require FILE_READER_MCP_URL to be set to a publicly accessible URL (NGrok).
    """
    if not mcp_server:
        pytest.skip("MCP file reader server failed to start")

    # Check if FILE_READER_MCP_URL is set and not localhost
    mcp_url = os.getenv("FILE_READER_MCP_URL")
    if not mcp_url:
        pytest.skip(
            "FILE_READER_MCP_URL environment variable is not set. "
            "MCP tests require a publicly accessible URL (NGrok). "
            "Set FILE_READER_MCP_URL to your NGrok endpoint."
        )

    # Check if URL is localhost (Anthropic blocks localhost)
    localhost_patterns = ["127.0.0.1", "localhost", "0.0.0.0"]
    if any(pattern in mcp_url.lower() for pattern in localhost_patterns):
        pytest.skip(
            f"FILE_READER_MCP_URL is set to localhost ({mcp_url}). "
            "Anthropic blocks localhost MCP servers for security. "
            "MCP tests require a publicly accessible URL (NGrok). "
            "Start NGrok (ngrok http 8765) and set FILE_READER_MCP_URL to the NGrok HTTPS URL."
        )


def pytest_configure(config: Any) -> None:
    """Configure pytest with integration test markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test requiring external services",
    )
    config.addinivalue_line(
        "markers", "anthropic: mark test as requiring Anthropic API"
    )
    config.addinivalue_line("markers", "ollama: mark test as requiring Ollama service")
    config.addinivalue_line("markers", "openai: mark test as requiring OpenAI API")
    config.addinivalue_line("markers", "gmail: mark test as requiring Gmail API access")
    config.addinivalue_line("markers", "mcp: mark test as requiring MCP server")
    config.addinivalue_line("markers", "slow: mark test as slow running (>10 seconds)")
    config.addinivalue_line(
        "markers",
        "manual: mark test as manual (requires special setup like NGrok, not run by default)",
    )


def pytest_collection_modifyitems(config: Any, items: List[Any]) -> None:
    """Modify test collection to add markers for integration tests."""
    for item in items:
        # Add integration marker to all tests in integration_tests directory
        if "integration_tests" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
