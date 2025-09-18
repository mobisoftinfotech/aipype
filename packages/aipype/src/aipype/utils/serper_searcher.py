"""Serper API search implementation."""

import os
import time
from typing import Any, Dict, List, Optional

from typing import override
import requests
from .base_searcher import BaseSearcher, SearchResult, SearchResponse


class SerperSearcher(BaseSearcher):
    """Search provider implementation using Serper API."""

    API_BASE_URL = "https://google.serper.dev"
    SEARCH_ENDPOINT = "/search"
    DEFAULT_TIMEOUT = 30
    MAX_RESULTS_LIMIT = 100

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize Serper searcher.

        Required config or environment variables:
        - api_key: Serper API key (or SERPER_API_KEY environment variable)

        Optional config parameters:
        - timeout: Request timeout in seconds (default: 30)
        - country: Country code for search results (e.g., 'us', 'uk')
        - language: Language code for search results (e.g., 'en', 'es')
        - safe_search: Safe search setting ('active', 'moderate', 'off')
        - search_type: Type of search ('search', 'images', 'videos', 'places', 'news')
        """
        super().__init__(config)

    @override
    def _validate_config(self) -> None:
        """Validate Serper searcher configuration."""
        if not self.api_key:
            raise ValueError(
                "Serper API key is required. Set 'api_key' in config or SERPER_API_KEY environment variable."
            )

    @property
    @override
    def api_key(self) -> Optional[str]:
        """Get Serper API key from config or environment variables."""
        # First try explicit config
        if "api_key" in self.config:
            return (
                str(self.config["api_key"])
                if self.config["api_key"] is not None
                else None
            )

        # Then try environment variable
        return os.getenv("SERPER_API_KEY")

    def _prepare_search_params(
        self, query: str, max_results: int, **kwargs: Any
    ) -> Dict[str, Any]:
        """Prepare search parameters for Serper API."""
        params: Dict[str, Any] = {
            "q": query,
            "num": min(max_results, self.MAX_RESULTS_LIMIT),
        }

        # Add optional parameters from config
        if "country" in self.config:
            params["gl"] = self.config["country"]

        if "language" in self.config:
            params["hl"] = self.config["language"]

        if "safe_search" in self.config:
            params["safe"] = self.config["safe_search"]

        if "search_type" in self.config:
            params["type"] = self.config["search_type"]

        # Override with any kwargs
        params.update(kwargs)

        return params

    def _prepare_headers(self) -> Dict[str, str]:
        """Prepare HTTP headers for Serper API request."""
        if self.api_key is None:
            raise RuntimeError(
                "Serper API key not found in config or environment variables"
            )
        return {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "mi-agents/1.0",
        }

    def _parse_search_results(self, data: Dict[str, Any], query: str) -> SearchResponse:
        """Parse Serper API response into SearchResponse."""
        results: List[SearchResult] = []
        organic_results = data.get("organic", [])

        for i, item in enumerate(organic_results):
            result = SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                position=item.get("position", i),
                metadata={
                    "displayed_link": item.get("displayedLink", ""),
                    "favicon": item.get("favicon", ""),
                    "date": item.get("date", ""),
                },
            )
            results.append(result)

        # Extract metadata
        search_metadata: Dict[str, Any] = {
            "search_information": data.get("searchInformation", {}),
            "answer_box": data.get("answerBox"),
            "people_also_ask": data.get("peopleAlsoAsk", []),
            "related_searches": data.get("relatedSearches", []),
            "knowledge_graph": data.get("knowledgeGraph"),
        }

        # Get total results and search time
        search_info = data.get("searchInformation", {})
        total_results = search_info.get("totalResults", 0)
        if isinstance(total_results, str):
            # Convert string numbers like "1,234,567" to int
            total_results = int(total_results.replace(",", ""))

        search_time = search_info.get("searchTime", 0.0)

        return SearchResponse(
            query=query,
            results=results,
            total_results=total_results,
            search_time=search_time,
            metadata=search_metadata,
        )

    @override
    def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> SearchResponse:
        """Perform a search using Serper API.

        Args:
            query: The search query string
            max_results: Maximum number of results to return (max 100)
            **kwargs: Additional Serper API parameters:
                - gl: Country code (e.g., 'us', 'uk')
                - hl: Language code (e.g., 'en', 'es')
                - safe: Safe search ('active', 'moderate', 'off')
                - type: Search type ('search', 'images', 'videos', 'places', 'news')

        Returns:
            SearchResponse object containing results

        Raises:
            RuntimeError: If the search API call fails
            ValueError: If query is invalid
        """
        # Validate query
        self.validate_query(query)

        if max_results <= 0 or max_results > self.MAX_RESULTS_LIMIT:
            raise ValueError(
                f"max_results must be between 1 and {self.MAX_RESULTS_LIMIT}"
            )

        self.logger.info(f"Performing Serper search for query: '{query}'")
        self.logger.debug(f"Max results: {max_results}, Additional params: {kwargs}")

        start_time = time.time()

        try:
            # Prepare request
            url = f"{self.API_BASE_URL}{self.SEARCH_ENDPOINT}"
            headers = self._prepare_headers()
            params = self._prepare_search_params(query, max_results, **kwargs)
            timeout: int = self.config.get("timeout", self.DEFAULT_TIMEOUT)

            # Make API request
            self.logger.debug(f"Making request to {url} with params: {params}")
            response = requests.post(url, json=params, headers=headers, timeout=timeout)

            # Check response status
            if response.status_code == 401:
                raise RuntimeError(
                    "Serper API authentication failed. Check your API key."
                )
            elif response.status_code == 429:
                raise RuntimeError(
                    "Serper API rate limit exceeded. Please try again later."
                )
            elif response.status_code != 200:
                raise RuntimeError(
                    f"Serper API request failed: {response.status_code} - {response.text}"
                )

            # Parse response
            data = response.json()
            search_response = self._parse_search_results(data, query)

            elapsed_time = time.time() - start_time
            self.logger.info(
                f"Serper search completed: {len(search_response.results)} results in {elapsed_time:.2f}s"
            )

            return search_response

        except requests.exceptions.Timeout:
            timeout_val: int = self.config.get("timeout", self.DEFAULT_TIMEOUT)
            error_msg = f"SerperSearcher search operation failed: Request timed out after {timeout_val} seconds"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        except requests.exceptions.ConnectionError as e:
            error_msg = f"SerperSearcher search operation failed: Failed to connect to Serper API: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        except requests.exceptions.RequestException as e:
            error_msg = f"SerperSearcher search operation failed: Request to Serper API failed: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        except (ValueError, KeyError) as e:
            error_msg = f"SerperSearcher search operation failed: Failed to parse Serper API response: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        except Exception as e:
            error_msg = f"SerperSearcher search operation failed: Unexpected error during search: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    @override
    def __str__(self) -> str:
        """String representation of the Serper searcher."""
        config_info: List[str] = []
        if "country" in self.config:
            config_info.append(f"country={self.config['country']}")
        if "language" in self.config:
            config_info.append(f"language={self.config['language']}")

        config_str = f"({', '.join(config_info)})" if config_info else ""
        return f"SerperSearcher{config_str}"
