"""Tests for SerperSearcher class."""

import os
from typing import Any, Optional
from unittest.mock import patch, Mock
import pytest
import requests
from aipype import SerperSearcher, SearchResponse


class TestSerperSearcherInitialization:
    """Test SerperSearcher initialization and configuration."""

    def test_init_with_api_key_in_config(self) -> None:
        config = {"api_key": "test-key-123"}
        searcher = SerperSearcher(config)

        assert searcher.config == config
        assert searcher.api_key == "test-key-123"

    @patch.dict(os.environ, {"SERPER_API_KEY": "env-key-456"})
    def test_init_with_api_key_in_env(self) -> None:
        searcher = SerperSearcher()

        assert searcher.api_key == "env-key-456"

    def test_init_missing_api_key(self) -> None:
        # Clear environment variables for this test
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Serper API key is required"):
                SerperSearcher()

    def test_init_with_optional_config(self) -> None:
        config = {
            "api_key": "test-key",
            "country": "us",
            "language": "en",
            "safe_search": "active",
            "search_type": "search",
            "timeout": 45,
        }
        searcher = SerperSearcher(config)

        assert searcher.config == config

    @patch.dict(os.environ, {"SERPER_API_KEY": "env-key"})
    def test_config_api_key_overrides_env(self) -> None:
        config = {"api_key": "config-key"}
        searcher = SerperSearcher(config)

        assert searcher.api_key == "config-key"


class TestSerperSearcherParameterPreparation:
    """Test search parameter preparation."""

    @patch.dict(os.environ, {"SERPER_API_KEY": "test-key"})
    def test_prepare_search_params_basic(self) -> None:
        searcher = SerperSearcher()
        params = searcher._prepare_search_params("test query", 10)  # type: ignore

        expected = {"q": "test query", "num": 10}
        assert params == expected

    @patch.dict(os.environ, {"SERPER_API_KEY": "test-key"})
    def test_prepare_search_params_with_config(self) -> None:
        config = {
            "api_key": "test-key",
            "country": "us",
            "language": "en",
            "safe_search": "active",
            "search_type": "news",
        }
        searcher = SerperSearcher(config)
        params = searcher._prepare_search_params("test query", 5)  # type: ignore

        expected = {
            "q": "test query",
            "num": 5,
            "gl": "us",
            "hl": "en",
            "safe": "active",
            "type": "news",
        }
        assert params == expected

    @patch.dict(os.environ, {"SERPER_API_KEY": "test-key"})
    def test_prepare_search_params_with_kwargs(self) -> None:
        searcher = SerperSearcher()
        params = searcher._prepare_search_params(  # type: ignore
            "test query", 10, gl="uk", custom_param="value"
        )

        expected = {"q": "test query", "num": 10, "gl": "uk", "custom_param": "value"}
        assert params == expected

    @patch.dict(os.environ, {"SERPER_API_KEY": "test-key"})
    def test_prepare_search_params_max_results_limit(self) -> None:
        searcher = SerperSearcher()
        params = searcher._prepare_search_params("test", 150)  # type: ignore

        assert params["num"] == 100  # Should cap at MAX_RESULTS_LIMIT

    @patch.dict(os.environ, {"SERPER_API_KEY": "test-key"})
    def test_prepare_headers(self) -> None:
        searcher = SerperSearcher()
        headers = searcher._prepare_headers()  # type: ignore

        expected = {
            "X-API-KEY": "test-key",
            "Content-Type": "application/json",
            "User-Agent": "mi-agents/1.0",
        }
        assert headers == expected


class TestSerperSearcherResponseParsing:
    """Test Serper API response parsing."""

    @patch.dict(os.environ, {"SERPER_API_KEY": "test-key"})
    def test_parse_search_results_basic(self) -> None:
        searcher = SerperSearcher()

        api_response = {
            "organic": [
                {
                    "title": "Test Result 1",
                    "link": "https://example1.com",
                    "snippet": "This is snippet 1",
                    "position": 1,
                    "displayedLink": "example1.com",
                    "date": "2024-01-01",
                },
                {
                    "title": "Test Result 2",
                    "link": "https://example2.com",
                    "snippet": "This is snippet 2",
                    "position": 2,
                    "favicon": "https://example2.com/favicon.ico",
                },
            ],
            "searchInformation": {"totalResults": "1,234,567", "searchTime": 0.42},
        }

        response = searcher._parse_search_results(api_response, "test query")  # type: ignore

        assert isinstance(response, SearchResponse)
        assert response.query == "test query"
        assert len(response.results) == 2
        assert response.total_results == 1234567
        assert response.search_time == 0.42

        # Check first result
        result1 = response.results[0]
        assert result1.title == "Test Result 1"
        assert result1.url == "https://example1.com"
        assert result1.snippet == "This is snippet 1"
        assert result1.position == 1
        assert result1.metadata["displayed_link"] == "example1.com"
        assert result1.metadata["date"] == "2024-01-01"

    @patch.dict(os.environ, {"SERPER_API_KEY": "test-key"})
    def test_parse_search_results_with_metadata(self) -> None:
        searcher = SerperSearcher()

        api_response = {
            "organic": [],
            "searchInformation": {"totalResults": 0, "searchTime": 0.1},
            "answerBox": {"answer": "42"},
            "peopleAlsoAsk": [{"question": "What is the meaning?"}],
            "relatedSearches": [{"query": "related query"}],
            "knowledgeGraph": {"title": "Knowledge"},
        }

        response = searcher._parse_search_results(api_response, "test")  # type: ignore

        assert len(response.results) == 0
        assert response.metadata["answer_box"]["answer"] == "42"
        assert len(response.metadata["people_also_ask"]) == 1
        assert len(response.metadata["related_searches"]) == 1
        assert response.metadata["knowledge_graph"]["title"] == "Knowledge"

    @patch.dict(os.environ, {"SERPER_API_KEY": "test-key"})
    def test_parse_search_results_empty(self) -> None:
        searcher = SerperSearcher()

        api_response = {"searchInformation": {"totalResults": "0", "searchTime": 0.1}}
        response = searcher._parse_search_results(api_response, "no results")  # type: ignore

        assert len(response.results) == 0
        assert response.total_results == 0
        assert response.search_time == 0.1


class TestSerperSearcherSearch:
    """Test actual search functionality with mocked requests."""

    def _create_mock_response(
        self, status_code: int = 200, json_data: Optional[Any] = None
    ) -> Mock:
        """Helper to create mock response."""
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.json.return_value = json_data or {
            "organic": [
                {
                    "title": "Test Result",
                    "link": "https://example.com",
                    "snippet": "Test snippet",
                    "position": 1,
                }
            ],
            "searchInformation": {"totalResults": "1", "searchTime": 0.1},
        }
        mock_response.text = "Mock response text"
        return mock_response

    @patch("aipype.utils.serper_searcher.requests.post")
    @patch.dict(os.environ, {"SERPER_API_KEY": "test-key"})
    def test_search_successful(self, mock_post: Mock) -> None:
        mock_post.return_value = self._create_mock_response()

        searcher = SerperSearcher()
        response = searcher.search("test query", max_results=5)

        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://google.serper.dev/search"
        assert call_args[1]["json"]["q"] == "test query"
        assert call_args[1]["json"]["num"] == 5
        assert call_args[1]["headers"]["X-API-KEY"] == "test-key"

        # Verify response
        assert isinstance(response, SearchResponse)
        assert response.query == "test query"
        assert len(response.results) == 1

    @patch("aipype.utils.serper_searcher.requests.post")
    @patch.dict(os.environ, {"SERPER_API_KEY": "test-key"})
    def test_search_with_kwargs(self, mock_post: Mock) -> None:
        mock_post.return_value = self._create_mock_response()

        searcher = SerperSearcher()
        searcher.search("test", max_results=3, gl="us", hl="en")

        # Verify kwargs were passed
        call_json = mock_post.call_args[1]["json"]
        assert call_json["gl"] == "us"
        assert call_json["hl"] == "en"

    @patch("aipype.utils.serper_searcher.requests.post")
    @patch.dict(os.environ, {"SERPER_API_KEY": "test-key"})
    def test_search_empty_query(self, mock_post: Mock) -> None:
        searcher = SerperSearcher()

        with pytest.raises(ValueError, match="Search query cannot be empty"):
            searcher.search("")

    @patch("aipype.utils.serper_searcher.requests.post")
    @patch.dict(os.environ, {"SERPER_API_KEY": "test-key"})
    def test_search_invalid_max_results(self, mock_post: Mock) -> None:
        searcher = SerperSearcher()

        with pytest.raises(ValueError, match="max_results must be between 1 and 100"):
            searcher.search("test", max_results=0)

        with pytest.raises(ValueError, match="max_results must be between 1 and 100"):
            searcher.search("test", max_results=101)

    @patch("aipype.utils.serper_searcher.requests.post")
    @patch.dict(os.environ, {"SERPER_API_KEY": "test-key"})
    def test_search_api_error_401(self, mock_post: Mock) -> None:
        mock_post.return_value = self._create_mock_response(status_code=401)

        searcher = SerperSearcher()

        with pytest.raises(RuntimeError, match="Serper API authentication failed"):
            searcher.search("test")

    @patch("aipype.utils.serper_searcher.requests.post")
    @patch.dict(os.environ, {"SERPER_API_KEY": "test-key"})
    def test_search_api_error_429(self, mock_post: Mock) -> None:
        mock_post.return_value = self._create_mock_response(status_code=429)

        searcher = SerperSearcher()

        with pytest.raises(RuntimeError, match="Serper API rate limit exceeded"):
            searcher.search("test")

    @patch("aipype.utils.serper_searcher.requests.post")
    @patch.dict(os.environ, {"SERPER_API_KEY": "test-key"})
    def test_search_api_error_generic(self, mock_post: Mock) -> None:
        mock_post.return_value = self._create_mock_response(status_code=500)

        searcher = SerperSearcher()

        with pytest.raises(RuntimeError, match="Serper API request failed: 500"):
            searcher.search("test")

    @patch("aipype.utils.serper_searcher.requests.post")
    @patch.dict(os.environ, {"SERPER_API_KEY": "test-key"})
    def test_search_timeout(self, mock_post: Mock) -> None:
        mock_post.side_effect = requests.exceptions.Timeout()

        searcher = SerperSearcher()

        with pytest.raises(
            RuntimeError,
            match="SerperSearcher search operation failed: Request timed out",
        ):
            searcher.search("test")

    @patch("aipype.utils.serper_searcher.requests.post")
    @patch.dict(os.environ, {"SERPER_API_KEY": "test-key"})
    def test_search_connection_error(self, mock_post: Mock) -> None:
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")

        searcher = SerperSearcher()

        with pytest.raises(RuntimeError, match="Failed to connect to Serper API"):
            searcher.search("test")

    @patch("aipype.utils.serper_searcher.requests.post")
    @patch.dict(os.environ, {"SERPER_API_KEY": "test-key"})
    def test_search_request_exception(self, mock_post: Mock) -> None:
        mock_post.side_effect = requests.exceptions.RequestException("Request failed")

        searcher = SerperSearcher()

        with pytest.raises(
            RuntimeError,
            match="SerperSearcher search operation failed: Request to Serper API failed",
        ):
            searcher.search("test")

    @patch("aipype.utils.serper_searcher.requests.post")
    @patch.dict(os.environ, {"SERPER_API_KEY": "test-key"})
    def test_search_json_parse_error(self, mock_post: Mock) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_post.return_value = mock_response

        searcher = SerperSearcher()

        with pytest.raises(RuntimeError, match="Failed to parse Serper API response"):
            searcher.search("test")


class TestSerperSearcherStringRepresentation:
    """Test string representation."""

    @patch.dict(os.environ, {"SERPER_API_KEY": "test-key"})
    def test_str_basic(self) -> None:
        searcher = SerperSearcher()
        str_repr = str(searcher)

        assert str_repr == "SerperSearcher"

    @patch.dict(os.environ, {"SERPER_API_KEY": "test-key"})
    def test_str_with_config(self) -> None:
        config = {"api_key": "test-key", "country": "us", "language": "en"}
        searcher = SerperSearcher(config)
        str_repr = str(searcher)

        assert "SerperSearcher" in str_repr
        assert "country=us" in str_repr
        assert "language=en" in str_repr
