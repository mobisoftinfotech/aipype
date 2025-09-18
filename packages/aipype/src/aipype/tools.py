"""Tool decorator and metadata extraction for function calling."""

import concurrent.futures
import inspect
import logging
import re
from functools import wraps
from typing import Any, Callable, Dict, List, get_type_hints

# Tool decorator creates functions with dynamic type signatures for LLM tool calling
# This allows functions to be introspected and converted to OpenAI-compatible schemas


class ToolMetadata:
    """Stores metadata for a tool function."""

    def __init__(self, func: Callable[..., Any], name: str, description: str):
        self.func = func
        self.name = name
        self.description = description
        self.parameters = self._extract_parameters(func)
        self.return_type = self._extract_return_type(func)

    def _extract_parameters(self, func: Callable[..., Any]) -> Dict[str, Any]:
        """Extract parameter information from function signature and docstring."""
        # Tool decorator preserves function signature for introspection
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)
        docstring = inspect.getdoc(func) or ""

        # Extract parameter descriptions from docstring
        param_descriptions = self._parse_docstring_args(docstring)

        parameters: Dict[str, Any] = {}

        for param_name, param in signature.parameters.items():
            # Skip *args and **kwargs
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue

            param_info: Dict[str, Any] = {
                "description": param_descriptions.get(
                    param_name, f"Parameter {param_name}"
                ),
                "optional": param.default is not param.empty,
            }

            # Extract type information and flatten schema
            if param_name in type_hints:
                param_type = type_hints[param_name]
                schema = self._type_to_json_schema(param_type)
                param_info["schema"] = schema
                # Flatten type for direct access (tests expect this)
                param_info["type"] = schema["type"]
            else:
                default_schema = {"type": "string"}  # Default type
                param_info["schema"] = default_schema
                param_info["type"] = "string"

            # Add default value if present
            if param.default is not param.empty and param.default is not None:
                param_info["default"] = param.default

            parameters[param_name] = param_info

        return parameters

    def _parse_docstring_args(self, docstring: str) -> Dict[str, str]:
        """Parse parameter descriptions from docstring Args section."""
        descriptions: Dict[str, str] = {}

        # Look for Args: section in docstring
        args_match = re.search(
            r"Args:\s*\n(.*?)(?:\n\s*\n|\n\s*Returns:|\n\s*Raises:|\Z)",
            docstring,
            re.DOTALL | re.IGNORECASE,
        )

        if not args_match:
            return descriptions

        args_section = args_match.group(1)

        # Parse parameter descriptions - handle multiline descriptions properly
        param_pattern = r"^\s*(\w+):\s*(.*?)(?=^\s*\w+:\s|\Z)"
        for match in re.finditer(param_pattern, args_section, re.MULTILINE | re.DOTALL):
            param_name = match.group(1).strip()
            # Clean up multiline description - collapse whitespace but preserve structure
            param_desc = re.sub(r"\s+", " ", match.group(2).strip())
            descriptions[param_name] = param_desc

        return descriptions

    def _extract_return_type(self, func: Callable[..., Any]) -> str:
        """Extract return type information from function signature."""
        # Tool decorator preserves return type annotations for schema generation
        type_hints = get_type_hints(func)
        return_annotation = type_hints.get("return", "any")
        schema = self._type_to_json_schema(return_annotation)
        # Return just the type string for backward compatibility
        return schema.get("type", "string")

    def _type_to_json_schema(self, type_hint: Any) -> Dict[str, Any]:
        """Convert Python type hint to JSON schema object."""
        # Handle common types
        if type_hint is str:
            return {"type": "string"}
        elif type_hint is int:
            return {"type": "integer"}
        elif type_hint is float:
            return {"type": "number"}
        elif type_hint is bool:
            return {"type": "boolean"}
        elif type_hint is list:
            # Generic list without type parameter
            return {"type": "array", "items": {"type": "string"}}
        elif type_hint is dict:
            return {"type": "object"}
        elif hasattr(type_hint, "__origin__"):
            # Handle generic types like List, Dict, Union, Optional
            origin = type_hint.__origin__
            if origin is list:
                # Handle List[T] with proper items schema
                if hasattr(type_hint, "__args__") and type_hint.__args__:
                    # Get the type parameter (e.g., float from List[float])
                    item_type = type_hint.__args__[0]
                    items_schema = self._type_to_json_schema(item_type)
                    return {"type": "array", "items": items_schema}
                else:
                    # Fallback for List without type parameter
                    return {"type": "array", "items": {"type": "string"}}
            elif origin is dict:
                return {"type": "object"}
            elif hasattr(type_hint, "__args__"):
                # For Union types, take first non-None type
                args = type_hint.__args__
                for arg in args:
                    if arg is not type(None):
                        return self._type_to_json_schema(arg)

        # Default to string for unknown types
        return {"type": "string"}


def tool(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to mark a function as a tool and extract metadata.

    Args:
        func: Function to be marked as a tool

    Returns:
        Decorated function with tool metadata attached

    Example:
        @tool
        def calculate(expression: str) -> float:
            '''Perform mathematical calculations.

            Args:
                expression: Mathematical expression to evaluate

            Returns:
                Calculated result
            '''
            return eval(expression)  # Use safe_eval in production
    """

    # Tool decorator preserves function behavior while adding metadata
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    # Extract metadata from function
    # Extract description from tool function for schema generation
    docstring = inspect.getdoc(func) or f"Execute {func.__name__}"

    # Extract description from docstring (first line or paragraph)
    description_lines = docstring.split("\n")
    description = description_lines[0].strip()

    # If first line is too short, try to get more context
    if len(description) < 10 and len(description_lines) > 1:
        for line in description_lines[1:]:
            line = line.strip()
            if (
                line
                and not line.startswith("Args:")
                and not line.startswith("Returns:")
            ):
                description = f"{description} {line}".strip()
                break

    metadata = ToolMetadata(func=func, name=func.__name__, description=description)

    # Store metadata on the wrapper function
    wrapper._tool_metadata = metadata  # type: ignore[attr-defined]
    wrapper._is_tool = True  # type: ignore[attr-defined]

    # Update metadata to reference the decorated wrapper function for validation
    metadata.func = wrapper

    # Return decorated function with tool metadata for LLM integration
    return wrapper


class ToolSchemaGenerator:
    """Generates OpenAI-compatible schemas from decorated functions."""

    @staticmethod
    def generate_schema(tool_func: Callable[..., Any]) -> Dict[str, Any]:
        """Generate OpenAI function schema from tool metadata.

        Args:
            tool_func: Function decorated with @tool

        Returns:
            OpenAI-compatible function schema

        Raises:
            ValueError: If function is not decorated with @tool
        """
        # Tool decorator creates dynamic metadata attributes for function calling
        if not hasattr(tool_func, "_tool_metadata"):
            raise ValueError(
                f"Function {tool_func.__name__} is not decorated with @tool"
            )

        # Tool decorator attaches metadata to function for schema generation
        metadata = tool_func._tool_metadata  # pyright: ignore[reportFunctionMemberAccess]

        # Build required parameters list
        required_params = [
            name
            for name, info in metadata.parameters.items()
            if not info.get("optional", False)
        ]

        # Build properties dictionary
        properties = {}
        for param_name, param_info in metadata.parameters.items():
            # Start with the schema object and add description
            property_schema = param_info["schema"].copy()
            property_schema["description"] = param_info["description"]

            # Add default value if present
            if "default" in param_info:
                property_schema["default"] = param_info["default"]

            properties[param_name] = property_schema

        return {
            "type": "function",
            "function": {
                "name": metadata.name,
                "description": metadata.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_params,
                },
            },
        }

    @staticmethod
    def validate_tool_function(tool_func: Callable[..., Any]) -> bool:
        """Validate that a function is properly decorated as a tool.

        Args:
            tool_func: Function to validate

        Returns:
            True if function is a valid tool, False otherwise
        """
        # Tool decorator creates validation attributes for tool registry
        if not hasattr(tool_func, "_is_tool"):
            return False

        if not hasattr(tool_func, "_tool_metadata"):
            return False

        try:
            # Try to generate schema to ensure metadata is valid
            # Tool decorator enables schema generation for validation
            ToolSchemaGenerator.generate_schema(tool_func)
            return True
        except Exception:
            return False


# Built-in search tool
logger = logging.getLogger(__name__)


@tool
def search_with_content(
    query: str,
    max_results: int = 5,
    max_content_results: int = 5,
    content_timeout: int = 15,
    html_method: str = "readability",
    skip_pdf: bool = False,
) -> Dict[str, Any]:
    """Search the web and automatically fetch content from the top results.

    This tool provides enhanced search functionality that not only returns search results
    but also automatically fetches and extracts readable content from the top URLs.
    Perfect for research tasks where you need both search results and their content.

    Args:
        query: The search query string to search for
        max_results: Maximum number of search results to return (1-10, default: 5)
        max_content_results: Maximum number of results to fetch content from (1-10, default: 5)
        content_timeout: Timeout in seconds for fetching content from each URL (default: 15)
        html_method: Method for HTML text extraction - "readability" or "basic" (default: "readability")
        skip_pdf: Whether to skip PDF files when fetching content (default: False)

    Returns:
        Dictionary containing search results with fetched content including query,
        total_results, search_time, results list with title/url/snippet/content for each,
        and content_stats with attempted/successful/failed/skipped counts.
    """
    # Import here to avoid circular imports
    from .utils.serper_searcher import SerperSearcher
    from .utils.url_fetcher import fetch_main_text

    # Validate and constrain parameters for tool calling context
    max_results = max(1, min(max_results, 10))  # Limit to 10 for tool calling
    max_content_results = max(1, min(max_content_results, max_results))
    content_timeout = max(5, min(content_timeout, 60))  # 5-60 seconds

    if not query or not query.strip():
        return {
            "query": query,
            "total_results": 0,
            "search_time": 0.0,
            "results": [],
            "content_stats": {
                "attempted": 0,
                "successful": 0,
                "failed": 0,
                "skipped": 0,
            },
            "error": "Query cannot be empty",
        }

    logger.info(f"Search tool called with query: '{query}', max_results: {max_results}")

    try:
        # Initialize searcher (will use SERPER_API_KEY from environment)
        searcher = SerperSearcher()

        # Perform search
        search_response = searcher.search(query, max_results=max_results)

        # Initialize result structure
        result_data: Dict[str, Any] = {
            "query": search_response.query,
            "total_results": search_response.total_results,
            "search_time": search_response.search_time,
            "results": [],
            "content_stats": {
                "attempted": 0,
                "successful": 0,
                "failed": 0,
                "skipped": 0,
            },
        }

        # Process search results
        search_results: List[Dict[str, Any]] = []
        for result in search_response.results:
            search_result: Dict[str, Any] = {
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet,
                "position": result.position,
                "content": None,
                "content_status": "not_fetched",
                "content_size": 0,
                "error": None,
            }
            search_results.append(search_result)

        result_data["results"] = search_results

        # Fetch content from top results if any results exist
        if search_results and max_content_results > 0:
            urls_to_fetch = [r["url"] for r in search_results[:max_content_results]]
            result_data["content_stats"]["attempted"] = len(urls_to_fetch)

            logger.info(f"Fetching content from {len(urls_to_fetch)} URLs")

            # Fetch content with parallel processing (limited workers for tool context)
            content_results = _fetch_content_parallel_for_tool(
                urls_to_fetch, content_timeout, html_method, skip_pdf, fetch_main_text
            )

            # Merge content results with search results
            url_to_content = {cr["url"]: cr for cr in content_results}

            for search_result in search_results[:max_content_results]:
                url = search_result["url"]
                content_result = url_to_content.get(url, {})

                search_result["content"] = content_result.get("content")
                search_result["content_status"] = content_result.get("status", "failed")
                search_result["content_size"] = (
                    len(content_result.get("content", ""))
                    if content_result.get("content")
                    else 0
                )
                search_result["error"] = content_result.get("error")

                # Update stats
                status = content_result.get("status", "failed")
                if status == "success":
                    result_data["content_stats"]["successful"] += 1
                elif status == "skipped":
                    result_data["content_stats"]["skipped"] += 1
                else:
                    result_data["content_stats"]["failed"] += 1

        logger.info(
            f"Search tool completed: {len(search_results)} results, "
            f"{result_data['content_stats']['successful']} content fetches successful"
        )

        return result_data

    except Exception as e:
        error_msg = f"Search with content failed: {str(e)}"
        logger.error(error_msg)

        return {
            "query": query,
            "total_results": 0,
            "search_time": 0.0,
            "results": [],
            "content_stats": {
                "attempted": 0,
                "successful": 0,
                "failed": 0,
                "skipped": 0,
            },
            "error": error_msg,
        }


def _fetch_single_content_for_tool(
    url: str,
    content_timeout: int,
    html_method: str,
    skip_pdf: bool,
    fetch_main_text: Callable[..., Any],
    max_content_length: int = 30000,
) -> Dict[str, Any]:
    """Fetch content from a single URL with error handling."""
    result: Dict[str, Any] = {
        "url": url,
        "status": "failed",
        "content": None,
        "error": None,
    }

    try:
        # Basic URL validation
        if not url or not url.startswith(("http://", "https://")):
            result["error"] = "Invalid URL format"
            return result

        # Skip PDFs if requested
        if skip_pdf and url.lower().endswith(".pdf"):
            result["status"] = "skipped"
            result["error"] = "PDF files skipped by configuration"
            return result

        # Fetch content with timeout and text extraction
        fetch_config = {
            "timeout": content_timeout,
            "html_method": html_method,
        }

        content_result = fetch_main_text(url, config=fetch_config)
        content_text = content_result.get("text", "")

        # Truncate if too long for tool calling context
        if len(content_text) > max_content_length:
            content_text = (
                content_text[:max_content_length]
                + "... [content truncated for tool calling]"
            )

        result.update(
            {
                "status": "success",
                "content": content_text,
            }
        )

    except Exception as e:
        result["error"] = f"Content fetch failed: {str(e)}"
        logger.warning(f"Failed to fetch content from {url}: {str(e)}")

    return result


def _fetch_content_parallel_for_tool(
    urls: List[str],
    content_timeout: int,
    html_method: str,
    skip_pdf: bool,
    fetch_main_text: Callable[..., Any],
) -> List[Dict[str, Any]]:
    """Fetch content from multiple URLs in parallel with limited workers."""
    results: List[Dict[str, Any]] = []

    # Use fewer workers for tool calling to avoid overwhelming the system
    max_workers = min(3, len(urls))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all fetch tasks
        future_to_url = {
            executor.submit(
                _fetch_single_content_for_tool,
                url,
                content_timeout,
                html_method,
                skip_pdf,
                fetch_main_text,
            ): url
            for url in urls
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_url):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                url = future_to_url[future]
                results.append(
                    {
                        "url": url,
                        "status": "failed",
                        "content": None,
                        "error": f"Parallel fetch failed: {str(e)}",
                    }
                )

    # Sort results by original URL order
    url_to_result = {r["url"]: r for r in results}
    return [
        url_to_result.get(
            url,
            {
                "url": url,
                "status": "failed",
                "content": None,
                "error": "Result not found",
            },
        )
        for url in urls
    ]
