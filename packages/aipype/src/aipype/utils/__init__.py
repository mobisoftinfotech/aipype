"""Utils package for common Python utilities."""

from .common import (
    setup_logger,
    timestamp,
    safe_dict_get,
    flatten_list,
    validate_required_fields,
)
from .base_searcher import BaseSearcher, SearchResult, SearchResponse
from .serper_searcher import SerperSearcher
from .url_fetcher import URLFetcher, fetch_url, fetch_url_headers, fetch_main_text
from .display import (
    format_header,
    format_separator,
    format_message_box,
    print_header,
    print_separator,
    print_message_box,
)

__all__ = [
    "setup_logger",
    "timestamp",
    "safe_dict_get",
    "flatten_list",
    "validate_required_fields",
    "BaseSearcher",
    "SearchResult",
    "SearchResponse",
    "SerperSearcher",
    "URLFetcher",
    "fetch_url",
    "fetch_url_headers",
    "fetch_main_text",
    "format_header",
    "format_separator",
    "format_message_box",
    "print_header",
    "print_separator",
    "print_message_box",
]
