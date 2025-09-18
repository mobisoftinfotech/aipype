"""Base searcher class for implementing pluggable search providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from typing import override
from dotenv import load_dotenv
from .common import setup_logger

# Load environment variables from .env file if present
load_dotenv()


class SearchResult:
    """Represents a single search result."""

    def __init__(
        self,
        title: str,
        url: str,
        snippet: str,
        position: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize search result.

        Args:
            title: The title of the search result
            url: The URL of the search result
            snippet: A brief description/snippet of the result
            position: Position in search results (0-based)
            metadata: Additional metadata from the search provider
        """
        self.title = title
        self.url = url
        self.snippet = snippet
        self.position = position
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "position": self.position,
            "metadata": self.metadata,
        }

    @override
    def __str__(self) -> str:
        """String representation of search result."""
        return f"SearchResult(title='{self.title}', url='{self.url}', position={self.position})"

    @override
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()


class SearchResponse:
    """Represents the complete search response from a provider."""

    def __init__(
        self,
        query: str,
        results: List[SearchResult],
        total_results: int = 0,
        search_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize search response.

        Args:
            query: The original search query
            results: List of search results
            total_results: Total number of results available
            search_time: Time taken for the search in seconds
            metadata: Additional metadata from the search provider
        """
        self.query = query
        self.results = results
        self.total_results = total_results
        self.search_time = search_time
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert search response to dictionary."""
        return {
            "query": self.query,
            "results": [result.to_dict() for result in self.results],
            "total_results": self.total_results,
            "search_time": self.search_time,
            "metadata": self.metadata,
        }

    @override
    def __str__(self) -> str:
        """String representation of search response."""
        return f"SearchResponse(query='{self.query}', results={len(self.results)}, total={self.total_results})"

    @override
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()


class BaseSearcher(ABC):
    """Abstract base class for search providers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the searcher with configuration.

        Args:
            config: Configuration dictionary for the search provider
        """
        self.config = config or {}
        self.logger = setup_logger(f"searcher.{self.__class__.__name__}")

        # Validate configuration
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the searcher configuration.

        Should raise ValueError if required configuration is missing.
        """
        pass

    @property
    @abstractmethod
    def api_key(self) -> Optional[str]:
        """Get API key from config or environment variables.

        Returns:
            API key string or None if not required/found
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> SearchResponse:
        """Perform a search query.

        Args:
            query: The search query string
            max_results: Maximum number of results to return
            **kwargs: Additional search parameters specific to the provider

        Returns:
            SearchResponse object containing results

        Raises:
            RuntimeError: If the search fails
            ValueError: If query is invalid
        """
        pass

    def validate_query(self, query: str) -> None:
        """Validate search query.

        Args:
            query: The search query to validate

        Raises:
            ValueError: If query is invalid
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        if len(query.strip()) > 2000:
            raise ValueError("Search query is too long (max 2000 characters)")

    @override
    def __str__(self) -> str:
        """String representation of the searcher."""
        return f"{self.__class__.__name__}()"

    @override
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()
