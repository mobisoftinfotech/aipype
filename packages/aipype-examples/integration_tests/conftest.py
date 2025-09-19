"""pytest configuration for aipype-examples integration tests."""

import os
import pytest
import requests
from typing import Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


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

    test_model = os.getenv("INTEGRATION_TEST_MODEL", "gemma3:1b")

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
def skip_if_ollama_unavailable(ollama_available: bool) -> None:
    """Skip test if Ollama is not available."""
    if not ollama_available:
        pytest.skip("Ollama service is not available")


@pytest.fixture
def skip_if_test_model_unavailable(test_model_available: bool) -> None:
    """Skip test if test model is not available."""
    if not test_model_available:
        test_model = os.getenv("INTEGRATION_TEST_MODEL", "gemma3:1b")
        pytest.skip(f"{test_model} model is not available in Ollama")


def pytest_configure(config: Any) -> None:
    """Configure pytest with integration test markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test requiring external services",
    )
    config.addinivalue_line("markers", "ollama: mark test as requiring Ollama service")
    config.addinivalue_line("markers", "openai: mark test as requiring OpenAI API")
    config.addinivalue_line("markers", "gmail: mark test as requiring Gmail API access")
    config.addinivalue_line("markers", "slow: mark test as slow running (>10 seconds)")


def pytest_collection_modifyitems(config: Any, items: List[Any]) -> None:
    """Modify test collection to add markers for integration tests."""
    for item in items:
        # Add integration marker to all tests in integration_tests directory
        if "integration_tests" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
