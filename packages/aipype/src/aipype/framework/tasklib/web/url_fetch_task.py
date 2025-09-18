"""URL Fetch Task - Fetches main text content from URLs."""

from typing import List, Dict, Any, Optional

from typing import override

from ...base_task import BaseTask
from ...task_dependencies import TaskDependency
from ...task_result import TaskResult
from ...utils.url_fetcher import fetch_main_text


class URLFetchTask(BaseTask):
    """Task that fetches main text content from a list of URLs."""

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        dependencies: Optional[List[TaskDependency]] = None,
    ):
        """Initialize URL fetch task.

        Args:
            name: Task name
            config: Configuration dictionary containing:
                - urls: List of URLs to fetch (optional, can be resolved from dependencies)
                - max_urls: Maximum number of URLs to process (default: 5)
                - timeout: Request timeout in seconds (default: 30)
            dependencies: List of task dependencies
        """
        super().__init__(name, config)
        self.dependencies = dependencies or []
        self.validation_rules = {
            "required": ["urls"],
            "defaults": {"max_urls": 5, "timeout": 30},
            "types": {"urls": list, "max_urls": int, "timeout": (int, float)},
            "ranges": {"max_urls": (1, 50), "timeout": (1, 300)},
            # Validation lambdas intentionally use dynamic typing for flexibility across field types
            "custom": {"urls": lambda x: len(x) > 0},  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
        }
        self.urls = self.config.get("urls", [])
        self.max_urls = self.config.get("max_urls", 5)
        self.timeout = self.config.get("timeout", 30)

    @override
    def get_dependencies(self) -> List[TaskDependency]:
        """Get the list of task dependencies.

        Returns:
            List of TaskDependency objects
        """
        return self.dependencies

    @override
    def run(self) -> TaskResult:
        """Fetch main text from the provided URLs.

        Returns:
            Dictionary containing:
                - total_urls: Total number of URLs processed
                - successful_fetches: Number of successful fetches
                - failed_fetches: Number of failed fetches
                - articles: List of article content with HTTP 200 status only
                - all_articles: List of all fetched article content (including non-200)
                - status_200_count: Number of articles with HTTP 200 status
                - non_200_count: Number of articles with non-200 HTTP status
                - errors: List of error messages for failed fetches
        """
        from datetime import datetime

        start_time = datetime.now()

        # Validate configuration using instance validation rules
        validation_failure = self._validate_or_fail(start_time)
        if validation_failure:
            return validation_failure

        # Get URLs from config (they may have been updated after initialization)
        urls = self.config.get("urls", [])

        self.logger.info(f"Starting URL fetch task for {len(urls)} URLs")

        articles: List[Dict[str, Any]] = []
        errors: List[str] = []
        successful_fetches = 0
        failed_fetches = 0
        filtered_articles: List[Dict[str, Any]] = []
        status_200_count = 0
        non_200_count = 0

        # Limit to max_urls
        urls_to_process = urls[: self.max_urls]

        for i, url in enumerate(urls_to_process, 1):
            self.logger.info(f"Fetching URL {i}/{len(urls_to_process)}: {url}")

            try:
                # Fetch main text from URL
                fetch_result = fetch_main_text(
                    url, {"timeout": self.timeout, "include_metadata": True}
                )

                # Get status code from metadata
                status_code = fetch_result.get("metadata", {}).get("status_code", 200)

                # Extract relevant information
                article_data: Dict[str, Any] = {
                    "url": fetch_result["url"],
                    "title": self._extract_title_from_url(url),
                    "content": fetch_result["text"],
                    "content_type": fetch_result["content_type"],
                    "text_size": fetch_result["text_size"],
                    "extraction_method": fetch_result["extraction_method"],
                    "status_code": status_code,
                    "metadata": fetch_result.get("metadata", {}),
                }

                articles.append(article_data)
                successful_fetches += 1

                # Filter out non-200 status codes
                if status_code == 200:
                    filtered_articles.append(article_data)
                    status_200_count += 1
                    self.logger.info(
                        f"Successfully fetched {fetch_result['text_size']} characters from {url} (HTTP {status_code})"
                    )
                else:
                    non_200_count += 1
                    self.logger.warning(
                        f"Fetched {url} but got HTTP {status_code} - excluding from results"
                    )

            except Exception as e:
                error_msg = f"Failed to fetch {url}: {str(e)}"
                errors.append(error_msg)
                failed_fetches += 1
                self.logger.error(error_msg)

        result_data = {
            "total_urls": len(urls_to_process),
            "successful_fetches": successful_fetches,
            "failed_fetches": failed_fetches,
            "articles": filtered_articles,  # Only include HTTP 200 articles
            "all_articles": articles,  # Keep all fetched articles for debugging
            "status_200_count": status_200_count,
            "non_200_count": non_200_count,
            "errors": errors,
        }

        self.logger.info(
            f"URL fetch task completed: {successful_fetches} total successful, "
            f"{status_200_count} with HTTP 200, {non_200_count} with non-200 status, "
            f"{failed_fetches} failed"
        )

        execution_time = (datetime.now() - start_time).total_seconds()

        # Determine if this should be considered a success
        if status_200_count > 0:
            return TaskResult.success(
                data=result_data,
                execution_time=execution_time,
                metadata={
                    "task_type": "url_fetch",
                    "total_urls_processed": len(urls_to_process),
                    "successful_fetches": successful_fetches,
                    "status_200_count": status_200_count,
                    "failed_fetches": failed_fetches,
                },
            )
        else:
            # No successful HTTP 200 fetches
            error_msg = f"URLFetchTask fetch operation failed: No URLs fetched successfully with HTTP 200 status. Successful fetches: {successful_fetches}, Errors: {failed_fetches}"
            return TaskResult.failure(
                error_message=error_msg,
                execution_time=execution_time,
                metadata={
                    "task_type": "url_fetch",
                    "total_urls_processed": len(urls_to_process),
                    "successful_fetches": successful_fetches,
                    "failed_fetches": failed_fetches,
                    "errors": errors,
                },
            )

    def _extract_title_from_url(self, url: str) -> str:
        """Extract a simple title from URL for display purposes."""
        try:
            # Remove protocol and domain
            if "://" in url:
                url = url.split("://", 1)[1]

            # Remove domain
            if "/" in url:
                url = url.split("/", 1)[1]

            # Clean up the path
            url = url.replace("/", " ").replace("-", " ").replace("_", " ")
            url = " ".join(url.split())

            # Limit length
            if len(url) > 100:
                url = url[:97] + "..."

            return url if url else "Untitled"
        except Exception:
            return "Untitled"
