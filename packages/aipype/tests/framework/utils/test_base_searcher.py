"""Tests for BaseSearcher class."""

import pytest
from typing import Any, Optional

from typing import override
from aipype import SearchResult, SearchResponse, BaseSearcher


class TestSearchResult:
    """Test SearchResult class."""

    def test_init_basic(self) -> None:
        result = SearchResult("Test Title", "https://example.com", "Test snippet")

        assert result.title == "Test Title"
        assert result.url == "https://example.com"
        assert result.snippet == "Test snippet"
        assert result.position == 0
        assert result.metadata == {}

    def test_init_with_position_and_metadata(self) -> None:
        metadata = {"source": "google", "rank": 5}
        result = SearchResult(
            "Test Title",
            "https://example.com",
            "Test snippet",
            position=3,
            metadata=metadata,
        )

        assert result.position == 3
        assert result.metadata == metadata

    def test_to_dict(self) -> None:
        metadata = {"source": "test"}
        result = SearchResult(
            "Title", "https://example.com", "Snippet", position=1, metadata=metadata
        )

        expected = {
            "title": "Title",
            "url": "https://example.com",
            "snippet": "Snippet",
            "position": 1,
            "metadata": metadata,
        }

        assert result.to_dict() == expected

    def test_str_representation(self) -> None:
        result = SearchResult("Test", "https://example.com", "snippet", position=2)
        str_repr = str(result)

        assert "SearchResult" in str_repr
        assert "Test" in str_repr
        assert "https://example.com" in str_repr
        assert "position=2" in str_repr


class TestSearchResponse:
    """Test SearchResponse class."""

    def test_init_basic(self) -> None:
        results = [SearchResult("Title", "https://example.com", "snippet")]
        response = SearchResponse("test query", results)

        assert response.query == "test query"
        assert len(response.results) == 1
        assert response.total_results == 0
        assert response.search_time == 0.0
        assert response.metadata == {}

    def test_init_with_all_params(self) -> None:
        results = [
            SearchResult("Title 1", "https://example1.com", "snippet 1"),
            SearchResult("Title 2", "https://example2.com", "snippet 2"),
        ]
        metadata = {"provider": "test"}

        response = SearchResponse(
            "test query", results, total_results=100, search_time=0.5, metadata=metadata
        )

        assert response.query == "test query"
        assert len(response.results) == 2
        assert response.total_results == 100
        assert response.search_time == 0.5
        assert response.metadata == metadata

    def test_to_dict(self) -> None:
        results = [SearchResult("Title", "https://example.com", "snippet")]
        metadata = {"test": "data"}
        response = SearchResponse(
            "query", results, total_results=50, search_time=1.2, metadata=metadata
        )

        expected = {
            "query": "query",
            "results": [results[0].to_dict()],
            "total_results": 50,
            "search_time": 1.2,
            "metadata": metadata,
        }

        assert response.to_dict() == expected

    def test_str_representation(self) -> None:
        results = [SearchResult("Title", "https://example.com", "snippet")]
        response = SearchResponse("test query", results, total_results=10)
        str_repr = str(response)

        assert "SearchResponse" in str_repr
        assert "test query" in str_repr
        assert "results=1" in str_repr
        assert "total=10" in str_repr


class MockSearcher(BaseSearcher):
    """Mock searcher for testing BaseSearcher."""

    @override
    def _validate_config(self) -> None:
        if "required_param" in self.config:
            return
        raise ValueError("Missing required_param")

    @property
    @override
    def api_key(self) -> Optional[str]:
        return str(self.config.get("api_key", "test-key"))

    @override
    def search(
        self, query: str, max_results: int = 10, **kwargs: Any
    ) -> SearchResponse:
        self.validate_query(query)
        results = [
            SearchResult(
                f"Result {i}", f"https://example{i}.com", f"Snippet {i}", position=i
            )
            for i in range(min(max_results, 3))
        ]
        return SearchResponse(query, results, total_results=100, search_time=0.1)


class TestBaseSearcher:
    """Test BaseSearcher abstract class."""

    def test_init_with_config(self) -> None:
        config = {"required_param": "value", "api_key": "test-key"}
        searcher = MockSearcher(config)

        assert searcher.config == config

    def test_init_without_config(self) -> None:
        with pytest.raises(ValueError, match="Missing required_param"):
            MockSearcher()

    def test_validate_query_valid(self) -> None:
        config = {"required_param": "value"}
        searcher = MockSearcher(config)

        # Should not raise
        searcher.validate_query("valid query")
        searcher.validate_query("a" * 100)  # Long but not too long

    def test_validate_query_empty(self) -> None:
        config = {"required_param": "value"}
        searcher = MockSearcher(config)

        with pytest.raises(ValueError, match="Search query cannot be empty"):
            searcher.validate_query("")

        with pytest.raises(ValueError, match="Search query cannot be empty"):
            searcher.validate_query("   ")

    def test_validate_query_too_long(self) -> None:
        config = {"required_param": "value"}
        searcher = MockSearcher(config)

        long_query = "a" * 2001
        with pytest.raises(ValueError, match="Search query is too long"):
            searcher.validate_query(long_query)

    def test_search_integration(self) -> None:
        config = {"required_param": "value"}
        searcher = MockSearcher(config)

        response = searcher.search("test query", max_results=2)

        assert isinstance(response, SearchResponse)
        assert response.query == "test query"
        assert len(response.results) == 2
        assert response.total_results == 100

    def test_str_representation(self) -> None:
        config = {"required_param": "value"}
        searcher = MockSearcher(config)

        str_repr = str(searcher)
        assert "MockSearcher" in str_repr
