"""Web-related tasks for the TaskLib.

This module contains tasks for web operations such as:
- URL content fetching
- Web scraping
- API interactions
- Content extraction
"""

from .url_fetch_task import URLFetchTask
from .batch_article_summarize_task import BatchArticleSummarizeTask

__all__ = [
    "URLFetchTask",
    "BatchArticleSummarizeTask",
]
