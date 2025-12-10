"""Helper functions for creating tasks with clean APIs.

This module provides intuitive helper functions for creating LLM and search tasks
within @task decorated methods. These helpers simplify the declarative syntax by
providing sensible defaults and a cleaner interface.

Example::

    from aipype import PipelineAgent, task, llm, search

    class MyAgent(PipelineAgent):
        @task
        def find_articles(self) -> dict:
            return search(self.config["topic"], max_results=10)

        @task
        def summarize(self, find_articles: dict) -> str:
            return llm(
                prompt=f"Summarize: {find_articles}",
                model="gpt-4o",
                temperature=0.3
            )
"""

from typing import Any, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from .llm_task import LLMTask
from .search_task import SearchTask


def llm(
    prompt: str,
    *,
    model: str = "gpt-4o",
    provider: str = "openai",
    system: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    tools: Optional[List[Union[Callable[..., Any], Dict[str, Any]]]] = None,
    response_format: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None,
    timeout: float = 60.0,
    **kwargs: Any,
) -> LLMTask:
    """Create an LLMTask with a clean, intuitive API.

    This helper simplifies LLMTask creation by providing sensible defaults
    and a more intuitive parameter interface. Designed to be used within
    @task decorated methods - returns LLMTask for delegation by TaskWrapper.

    Args:
        prompt: The prompt to send to the LLM
        model: Model identifier (default: "gpt-4o")
        provider: LLM provider (default: "openai")
        system: System/context message
        temperature: Randomness control 0.0-2.0 (default: 0.7)
        max_tokens: Maximum response tokens (default: 1000)
        tools: List of @tool functions and/or MCP server configs
        response_format: Pydantic model or JSON schema for structured output
        timeout: Request timeout in seconds (default: 60)
        **kwargs: Additional LLMTask configuration

    Returns:
        Configured LLMTask ready for delegation

    Example::

        @task
        def summarize(self, articles: list) -> str:
            return llm(
                prompt=f"Summarize these articles: {articles}",
                model="gpt-4o",
                temperature=0.3,
                max_tokens=500
            )

    Example with structured output and tools::

        @task
        def analyze(self, data: dict) -> AnalysisResult:
            return llm(
                prompt=f"Analyze: {data}",
                model="gpt-4o",
                response_format=AnalysisResult,
                tools=[calculator, mcp_server("brave", "https://mcp.brave.com")]
            )
    """
    config: Dict[str, Any] = {
        "prompt": prompt,
        "llm_provider": provider,
        "llm_model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        **kwargs,
    }

    if system:
        config["context"] = system

    if tools:
        config["tools"] = tools

    if response_format:
        config["response_format"] = response_format

    # Use placeholder name - TaskWrapper handles delegation
    return LLMTask(name="_llm_delegated", config=config)


def search(
    query: str,
    *,
    max_results: int = 5,
    **kwargs: Any,
) -> SearchTask:
    """Create a SearchTask with a clean API.

    Args:
        query: Search query string
        max_results: Maximum number of results (default: 5)
        **kwargs: Additional SearchTask configuration

    Returns:
        Configured SearchTask ready for delegation

    Example::

        @task
        def find_articles(self) -> dict:
            topic = self.config.get("topic", "AI")
            return search(f"{topic} latest news", max_results=10)
    """
    config: Dict[str, Any] = {
        "query": query,
        "max_results": max_results,
        **kwargs,
    }

    return SearchTask(name="_search_delegated", config=config)


def mcp_server(
    label: str,
    url: str,
    *,
    allowed_tools: Optional[List[str]] = None,
    require_approval: str = "never",
) -> Dict[str, Any]:
    """Create MCP server configuration for use with llm().

    Model Context Protocol (MCP) servers provide external tools
    that can be called by LLMs. This helper creates the configuration
    dict needed by LLMTask.

    Args:
        label: Server label/name for identification
        url: MCP server URL endpoint
        allowed_tools: Optional list of specific tools to enable
        require_approval: Approval mode - "never", "always", or "once"

    Returns:
        MCP configuration dict for use in tools parameter

    Example::

        @task
        def research(self, topic: str) -> str:
            return llm(
                prompt=f"Research {topic} thoroughly",
                model="claude-sonnet-4",
                provider="anthropic",
                tools=[
                    mcp_server("brave", "https://mcp.brave.com"),
                    mcp_server("github", "https://mcp.github.com",
                              allowed_tools=["search_repos", "get_file"])
                ]
            )
    """
    config: Dict[str, Any] = {
        "type": "mcp",
        "server_label": label,
        "server_url": url,
        "require_approval": require_approval,
    }

    if allowed_tools:
        config["allowed_tools"] = allowed_tools

    return config


def transform(
    input_data: Any,
    fn: Callable[[Any], Any],
    *,
    output_name: str = "result",
) -> Dict[str, Any]:
    """Apply a transformation function and return result as dict.

    Unlike returning a TransformTask, this helper executes the
    transformation immediately and returns the result. Use this
    for simple data transformations within @task methods.

    Args:
        input_data: Data to transform
        fn: Transformation function
        output_name: Key name for the result in output dict

    Returns:
        Dict with transformed data under output_name and 'data' keys

    Example::

        @task
        def extract_urls(self, search_results: dict) -> dict:
            results = search_results.get("results", [])
            return transform(
                results,
                fn=lambda r: [item["url"] for item in r]
            )
    """
    result = fn(input_data)
    return {output_name: result, "data": result}
