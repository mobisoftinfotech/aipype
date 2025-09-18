"""SearchTask - Simple web search task."""

from typing import Any, Dict, List, Optional

from typing import override
from .base_task import BaseTask
from .task_result import TaskResult
from .task_dependencies import TaskDependency
from .utils.serper_searcher import SerperSearcher


class SearchTask(BaseTask):
    """Simple web search task that executes searches and returns results."""

    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[TaskDependency]] = None,
    ):
        """Initialize search task.

        Args:
            name: Task name
            config: Task configuration
            dependencies: List of task dependencies

        Config parameters:
        - query: Search query string (required)
        - max_results: Maximum number of search results (default: 5)
        - serper_api_key: Optional API key for Serper search
        """
        super().__init__(name, config, dependencies)
        self.validation_rules = {
            "required": ["query"],
            "defaults": {"max_results": 5},
            "types": {"query": str, "max_results": int},
            "ranges": {"max_results": (1, 100)},
            # Validation lambdas intentionally use dynamic typing for flexibility across field types
            "custom": {"query": lambda x: x.strip() != ""},  # pyright: ignore[reportUnknownLambdaType,reportUnknownMemberType]
        }

    @override
    def run(self) -> TaskResult:
        """Execute the search and return results."""
        from datetime import datetime

        start_time = datetime.now()

        # Validate configuration using instance validation rules
        validation_failure = self._validate_or_fail(start_time)
        if validation_failure:
            return validation_failure

        try:
            query = self.config.get("query", "")
            max_results = self.config.get("max_results", 5)

            # Initialize searcher with configuration
            searcher_config: Dict[str, Any] = {}
            if "serper_api_key" in self.config:
                searcher_config["api_key"] = self.config["serper_api_key"]

            searcher = SerperSearcher(searcher_config)

            self.logger.info(f"Searching for: '{query}' (max {max_results} results)")

            # Perform search
            response = searcher.search(query, max_results=max_results)

            # Format results for return
            search_results: Dict[str, Any] = {
                "query": response.query,
                "total_results": response.total_results,
                "search_time": response.search_time,
                "results": [],
            }

            result_list: List[Dict[str, Any]] = search_results["results"]
            for result in response.results:
                result_list.append(
                    {
                        "title": result.title,
                        "url": result.url,
                        "snippet": result.snippet,
                        "position": result.position,
                    }
                )

            execution_time = (datetime.now() - start_time).total_seconds()

            self.logger.info(
                f"Found {len(response.results)} results in {response.search_time:.2f}s"
            )

            return TaskResult.success(
                data=search_results,
                execution_time=execution_time,
                metadata={
                    "task_name": self.name,
                    "query": query,
                    "max_results": max_results,
                    "actual_results": len(response.results),
                    "search_time": response.search_time,
                },
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"SearchTask search operation failed: Search task '{self.name}' failed: {str(e)}"
            self.logger.error(error_msg)

            return TaskResult.failure(
                error_message=error_msg,
                execution_time=execution_time,
                metadata={
                    "task_name": self.name,
                    "query": self.config.get("query", ""),
                    "error_type": type(e).__name__,
                },
            )

    @override
    def __str__(self) -> str:
        """String representation of the search task."""
        query = self.config.get("query", "")
        max_results = self.config.get("max_results", 5)
        dep_count = len(self.dependencies)

        return f"SearchTask(name='{self.name}', query='{query[:50]}...', max_results={max_results}, dependencies={dep_count})"
