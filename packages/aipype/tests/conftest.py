"""pytest configuration and shared fixtures for task pipeline framework tests."""

import pytest
import tempfile
from typing import Dict, Any, List, Optional, Generator
from unittest.mock import Mock, patch

from typing import override
from aipype import (
    BaseTask,
    BasePipelineAgent,
    TaskResult,
    TaskContext,
    DependencyResolver,
)
from dotenv import load_dotenv

# Load environment variables from .env file at the start of test session
load_dotenv()


class SuccessTask(BaseTask):
    """Task that always succeeds and returns a configured value."""

    @override
    def run(self) -> TaskResult:
        return TaskResult.success(self.config.get("return_value", "success"))


class FailureTask(BaseTask):
    """Task that always fails with a configured exception."""

    @override
    def run(self) -> TaskResult:
        error_msg = self.config.get("error_message", "Task failed")
        return TaskResult.failure(error_msg)


class SlowTask(BaseTask):
    """Task that simulates slower execution."""

    @override
    def run(self) -> TaskResult:
        import time

        delay = self.config.get("delay", 0.1)
        time.sleep(delay)
        return TaskResult.success(f"slept for {delay}s")


class MockAgent(BasePipelineAgent):
    """Test agent with configurable tasks using modern pipeline architecture."""

    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        tasks: Optional[List[BaseTask]] = None,
    ):
        super().__init__(name, config)
        self._test_tasks = tasks or []

    @override
    def setup_tasks(self) -> List[BaseTask]:
        return self._test_tasks


@pytest.fixture
def success_task() -> SuccessTask:
    """Create a task that succeeds."""
    return SuccessTask("test_success", {"return_value": "test_result"})


@pytest.fixture
def failure_task() -> FailureTask:
    """Create a task that fails."""
    return FailureTask("test_failure", {"error_message": "Test error"})


@pytest.fixture
def slow_task() -> SlowTask:
    """Create a task that takes some time."""
    return SlowTask("test_slow", {"delay": 0.05})


@pytest.fixture
def sample_agent() -> MockAgent:
    """Create a test agent with basic configuration."""
    tasks = [
        SuccessTask("task1", {"return_value": "result1"}),
        SuccessTask("task2", {"return_value": "result2"}),
    ]
    return MockAgent("test_agent", {"stop_on_failure": True}, tasks)  # type: ignore


# New fixtures for pipeline framework testing


@pytest.fixture
def task_context() -> Generator[TaskContext, None, None]:
    """Provide a clean TaskContext for testing."""
    context = TaskContext()
    yield context
    context.clear()


@pytest.fixture
def dependency_resolver(task_context: TaskContext) -> DependencyResolver:
    """Provide a DependencyResolver with clean context."""
    return DependencyResolver(task_context)


@pytest.fixture
def temp_output_dir() -> Generator[str, None, None]:
    """Provide a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_llm_response() -> Mock:
    """Provide a mock LLM response for testing."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Mock LLM response content"
    mock_response.model = "gpt-3.5-turbo"
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_response.usage.total_tokens = 150
    return mock_response


@pytest.fixture
def mock_search_response() -> Mock:
    """Provide a mock search response for testing."""
    mock_response = Mock()
    mock_response.query = "test query"
    mock_response.results = [
        Mock(
            title="Test Article 1",
            url="http://example1.com",
            snippet="Test content 1",
            position=1,
        ),
        Mock(
            title="Test Article 2",
            url="http://example2.com",
            snippet="Test content 2",
            position=2,
        ),
        Mock(
            title="Test Article 3",
            url="http://example3.com",
            snippet="Test content 3",
            position=3,
        ),
    ]
    mock_response.total_results = 3
    mock_response.search_time = 0.5
    return mock_response


@pytest.fixture
def sample_agent_config() -> Dict[str, Any]:
    """Provide sample agent configuration for testing."""
    return {
        "search_keywords": "artificial intelligence trends",
        "guideline": "Focus on business applications and practical use cases",
        "llm_provider": "openai",
        "llm_model": "gpt-3.5-turbo",
        "max_search_results": 5,
        "output_dir": "test_output",
        "stop_on_failure": True,
    }


@pytest.fixture(autouse=True)
def mock_external_apis(
    request: pytest.FixtureRequest,
) -> Generator[Dict[str, Any], None, None]:
    """Automatically mock external APIs for unit tests only (not integration tests)."""
    # Skip mocking for integration tests
    # Pytest internal node structure has dynamic attributes not easily typed
    if hasattr(request, "node") and hasattr(request.node, "iter_markers"):  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        # pytest marker iteration yields marker objects with unknown typing
        for marker in request.node.iter_markers():  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            # pytest marker objects have dynamic name attributes
            if marker.name == "integration":  # pyright: ignore[reportUnknownMemberType]
                # Return empty dict for integration tests - no mocking
                yield {}
                return
    with (
        patch("litellm.completion") as mock_llm,
        patch("aipype.utils.serper_searcher.SerperSearcher") as mock_searcher_class,
        patch("aipype.utils.url_fetcher.fetch_main_text") as mock_fetcher,
    ):
        # Default LLM mock
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Mock response"
        mock_response.model = "mock-model"
        mock_response.usage = Mock(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )
        mock_llm.return_value = mock_response

        # Default search mock
        mock_searcher = Mock()
        mock_search_response = Mock()
        mock_search_response.query = "mock query"
        mock_search_response.results = []
        mock_search_response.total_results = 0
        mock_search_response.search_time = 0.1
        mock_searcher.search.return_value = mock_search_response
        mock_searcher_class.return_value = mock_searcher

        # Default fetch mock
        mock_fetcher.return_value = {
            "url": "http://mock.com",
            "text": "Mock article content",
            "content_type": "text/html",
            "text_size": 100,
            "extraction_method": "mock",
        }

        yield {
            "llm": mock_llm,
            "searcher_class": mock_searcher_class,
            "searcher": mock_searcher,
            "fetcher": mock_fetcher,
        }


def pytest_configure(config: Any) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test requiring external resources",
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test that may take longer"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_collection_modifyitems(config: Any, items: List[Any]) -> None:
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add integration marker to tests in integration directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add performance marker to tests in performance directory
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)


# Test data factories
class TestDataFactory:
    """Factory for creating test data structures."""

    @staticmethod
    def create_search_results(count: int = 3) -> Dict[str, Any]:
        """Create mock search results."""
        return {
            "query": "test query",
            "results": [
                {
                    "url": f"http://example{i}.com",
                    "title": f"Test Article {i}",
                    "snippet": f"Test content snippet {i}",
                }
                for i in range(1, count + 1)
            ],
            "total_results": count,
            "search_time": 0.5,
        }

    @staticmethod
    def create_fetch_results(urls: List[str]) -> Dict[str, Any]:
        """Create mock fetch results for given URLs."""
        return {
            "total_urls": len(urls),
            "successful_fetches": len(urls),
            "failed_fetches": 0,
            "articles": [
                {
                    "url": url,
                    "title": f"Article from {url}",
                    "content": f"Full content from {url}. This is detailed information.",
                    "word_count": 150,
                }
                for url in urls
            ],
            "errors": [],
        }

    @staticmethod
    def create_llm_result(content: str = "Generated content") -> Dict[str, Any]:
        """Create mock LLM result."""
        return {
            "content": content,
            "model": "gpt-3.5-turbo",
            "provider": "openai",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }


@pytest.fixture
def test_data_factory() -> type[TestDataFactory]:
    """Provide TestDataFactory for creating test data."""
    return TestDataFactory


# Success criteria validation helpers
class SuccessCriteriaValidator:
    """Helper class to validate success criteria for the refactoring."""

    @staticmethod
    def validate_no_custom_run_methods(agent_class: type[BasePipelineAgent]) -> bool:
        """Validate that agent doesn't have custom run() orchestration."""
        if hasattr(agent_class, "run"):
            # If run() exists, it should be inherited from BasePipelineAgent
            return agent_class.run is BasePipelineAgent.run
        return True

    @staticmethod
    def validate_declarative_setup(agent: BasePipelineAgent) -> bool:
        """Validate that agent setup is declarative."""
        tasks = agent.setup_tasks()

        # Should return a list of tasks
        assert isinstance(tasks, list)
        assert len(tasks) > 0

        # Each task should have dependencies defined
        for task in tasks:
            assert hasattr(task, "get_dependencies")
            deps = task.get_dependencies()
            assert isinstance(deps, list)

        return True

    @staticmethod
    def validate_task_reusability(task: BaseTask) -> bool:
        """Validate that a task can be reused independently."""
        # Task should be self-contained with only dependency-based inputs
        assert hasattr(task, "config")
        assert hasattr(task, "get_dependencies")
        assert hasattr(task, "run")
        return True

    @staticmethod
    def validate_context_driven_data_flow(agent: BasePipelineAgent) -> bool:
        """Validate that data flows through context, not direct coupling."""
        tasks = agent.setup_tasks()

        # Tasks should not directly reference other task instances
        for task in tasks:
            deps = task.get_dependencies()
            for dep in deps:
                # Dependencies should reference context paths, not task objects
                assert hasattr(dep, "source_path")
                assert isinstance(dep.source_path, str)

        return True


@pytest.fixture
def success_criteria_validator() -> type[SuccessCriteriaValidator]:
    """Provide SuccessCriteriaValidator for validation checks."""
    return SuccessCriteriaValidator
