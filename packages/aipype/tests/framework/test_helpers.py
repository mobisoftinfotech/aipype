"""Tests for helper functions in aipype.helpers."""

from typing import Any, List

from pydantic import BaseModel

from aipype.helpers import llm, search, mcp_server, transform
from aipype.llm_task import LLMTask
from aipype.search_task import SearchTask
from aipype.tools import tool


class TestLlmHelper:
    """Tests for llm() helper function."""

    def test_llm_basic(self) -> None:
        """Test basic llm() helper creates LLMTask."""
        task = llm(prompt="Hello", model="gpt-4o", provider="openai")

        assert isinstance(task, LLMTask)
        assert task.config["prompt"] == "Hello"
        assert task.config["llm_model"] == "gpt-4o"
        assert task.config["llm_provider"] == "openai"
        assert task.config["temperature"] == 0.7  # default
        assert task.config["max_tokens"] == 1000  # default
        assert task.config["timeout"] == 60.0  # default

    def test_llm_with_defaults(self) -> None:
        """Test llm() uses correct defaults."""
        task = llm(prompt="Test prompt")

        assert task.config["llm_model"] == "gpt-4o"
        assert task.config["llm_provider"] == "openai"
        assert task.config["temperature"] == 0.7
        assert task.config["max_tokens"] == 1000

    def test_llm_with_system(self) -> None:
        """Test llm() with system message."""
        task = llm(
            prompt="Analyze this",
            system="You are an expert analyst.",
        )

        assert task.config["context"] == "You are an expert analyst."

    def test_llm_with_custom_params(self) -> None:
        """Test llm() with custom parameters."""
        task = llm(
            prompt="Generate",
            model="claude-sonnet-4",
            provider="anthropic",
            temperature=0.3,
            max_tokens=2000,
            timeout=120.0,
        )

        assert task.config["llm_model"] == "claude-sonnet-4"
        assert task.config["llm_provider"] == "anthropic"
        assert task.config["temperature"] == 0.3
        assert task.config["max_tokens"] == 2000
        assert task.config["timeout"] == 120.0

    def test_llm_with_tools(self) -> None:
        """Test llm() with tools."""

        @tool
        def calculator(expr: str) -> float:
            """Calculate a mathematical expression."""
            return float(eval(expr))  # noqa: S307

        task = llm(
            prompt="Calculate 2+2",
            tools=[calculator],
        )

        assert "tools" in task.config
        assert calculator in task.config["tools"]

    def test_llm_with_mcp_tools(self) -> None:
        """Test llm() with MCP server tools."""
        mcp_config = mcp_server("brave", "https://mcp.brave.com")

        task = llm(
            prompt="Search for info",
            tools=[mcp_config],
        )

        assert "tools" in task.config
        assert mcp_config in task.config["tools"]

    def test_llm_with_mixed_tools(self) -> None:
        """Test llm() with both Python tools and MCP servers."""

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        mcp_config = mcp_server("search", "https://mcp.example.com")

        task = llm(
            prompt="Process data",
            tools=[add, mcp_config],
        )

        assert len(task.config["tools"]) == 2
        assert add in task.config["tools"]
        assert mcp_config in task.config["tools"]

    def test_llm_with_response_format(self) -> None:
        """Test llm() with Pydantic response format."""

        class Output(BaseModel):
            summary: str
            score: int

        task = llm(
            prompt="Analyze",
            response_format=Output,
        )

        assert task.config["response_format"] == Output

    def test_llm_with_dict_response_format(self) -> None:
        """Test llm() with dict response format."""
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "output",
                "schema": {
                    "type": "object",
                    "properties": {"result": {"type": "string"}},
                },
            },
        }

        task = llm(
            prompt="Generate",
            response_format=schema,
        )

        assert task.config["response_format"] == schema

    def test_llm_with_kwargs(self) -> None:
        """Test llm() passes through additional kwargs."""
        task = llm(
            prompt="Test",
            custom_param="value",
            another_param=123,
        )

        assert task.config["custom_param"] == "value"
        assert task.config["another_param"] == 123

    def test_llm_task_name(self) -> None:
        """Test llm() creates task with placeholder name."""
        task = llm(prompt="Test")

        assert task.name == "_llm_delegated"


class TestSearchHelper:
    """Tests for search() helper function."""

    def test_search_basic(self) -> None:
        """Test basic search() helper."""
        task = search("AI news", max_results=10)

        assert isinstance(task, SearchTask)
        assert task.config["query"] == "AI news"
        assert task.config["max_results"] == 10

    def test_search_with_defaults(self) -> None:
        """Test search() uses correct defaults."""
        task = search("test query")

        assert task.config["query"] == "test query"
        assert task.config["max_results"] == 5  # default

    def test_search_with_kwargs(self) -> None:
        """Test search() passes through additional kwargs."""
        task = search(
            "query",
            max_results=10,
            serper_api_key="test_key",
        )

        assert task.config["serper_api_key"] == "test_key"

    def test_search_task_name(self) -> None:
        """Test search() creates task with placeholder name."""
        task = search("test")

        assert task.name == "_search_delegated"


class TestMcpServerHelper:
    """Tests for mcp_server() helper function."""

    def test_mcp_server_basic(self) -> None:
        """Test mcp_server() creates correct config."""
        config = mcp_server("brave", "https://mcp.brave.com")

        assert config["type"] == "mcp"
        assert config["server_label"] == "brave"
        assert config["server_url"] == "https://mcp.brave.com"
        assert config["require_approval"] == "never"

    def test_mcp_server_with_allowed_tools(self) -> None:
        """Test mcp_server() with allowed_tools."""
        config = mcp_server(
            "github",
            "https://mcp.github.com",
            allowed_tools=["search", "get_file"],
        )

        assert config["allowed_tools"] == ["search", "get_file"]

    def test_mcp_server_with_require_approval(self) -> None:
        """Test mcp_server() with different require_approval values."""
        config_always = mcp_server(
            "test",
            "https://test.com",
            require_approval="always",
        )
        assert config_always["require_approval"] == "always"

        config_once = mcp_server(
            "test",
            "https://test.com",
            require_approval="once",
        )
        assert config_once["require_approval"] == "once"

    def test_mcp_server_without_allowed_tools(self) -> None:
        """Test mcp_server() without allowed_tools doesn't include key."""
        config = mcp_server("test", "https://test.com")

        assert "allowed_tools" not in config


class TestTransformHelper:
    """Tests for transform() helper function."""

    def test_transform_basic(self) -> None:
        """Test transform() helper."""
        data = [1, 2, 3]
        result = transform(data, fn=lambda x: sum(x))

        assert result["result"] == 6
        assert result["data"] == 6

    def test_transform_custom_output_name(self) -> None:
        """Test transform() with custom output name."""
        data = ["a", "b", "c"]
        result = transform(data, fn=lambda x: "-".join(x), output_name="joined")

        assert result["joined"] == "a-b-c"
        assert result["data"] == "a-b-c"

    def test_transform_with_list_extraction(self) -> None:
        """Test transform() for extracting list items."""
        data = [{"url": "http://a.com"}, {"url": "http://b.com"}]
        result = transform(data, fn=lambda items: [item["url"] for item in items])

        assert result["result"] == ["http://a.com", "http://b.com"]

    def test_transform_with_dict_input(self) -> None:
        """Test transform() with dict input."""
        data = {"values": [1, 2, 3], "factor": 2}
        result = transform(
            data,
            fn=lambda d: [v * d["factor"] for v in d["values"]],
        )

        assert result["result"] == [2, 4, 6]

    def test_transform_with_string_output(self) -> None:
        """Test transform() returning string."""
        data = ["hello", "world"]
        result = transform(data, fn=lambda x: " ".join(x).upper())

        assert result["result"] == "HELLO WORLD"
        assert result["data"] == "HELLO WORLD"

    def test_transform_with_none_result(self) -> None:
        """Test transform() with function returning None."""

        def return_none(x: Any) -> None:
            return None

        data = [1, 2, 3]
        result = transform(data, fn=return_none)

        assert result["result"] is None
        assert result["data"] is None

    def test_transform_with_complex_function(self) -> None:
        """Test transform() with more complex transformation."""

        def aggregate(items: List[dict[str, Any]]) -> dict[str, Any]:
            return {
                "count": len(items),
                "total": sum(item.get("value", 0) for item in items),
                "avg": sum(item.get("value", 0) for item in items) / len(items)
                if items
                else 0,
            }

        data = [{"value": 10}, {"value": 20}, {"value": 30}]
        result = transform(data, fn=aggregate, output_name="stats")

        assert result["stats"]["count"] == 3
        assert result["stats"]["total"] == 60
        assert result["stats"]["avg"] == 20.0
