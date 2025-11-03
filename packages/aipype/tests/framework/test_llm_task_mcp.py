"""Tests for LLMTask MCP (Model Context Protocol) integration."""

from typing import Any, Dict
from unittest.mock import MagicMock, patch

from aipype import LLMTask, tool


class TestLLMTaskMCPValidation:
    """Test MCP configuration validation in LLMTask."""

    def create_test_python_tool(self) -> Any:
        """Create a test Python tool."""

        @tool
        def calculate(expression: str) -> float:
            """Perform mathematical calculations.

            Args:
                expression: Mathematical expression to evaluate
            """
            return float(eval(expression))  # Unsafe but OK for tests

        return calculate

    def create_valid_mcp_config_type_mcp(self) -> Dict[str, Any]:
        """Create a valid MCP configuration with type 'mcp'."""
        return {
            "type": "mcp",
            "server_label": "test_server",
            "server_url": "https://mcp.example.com",
            "require_approval": "never",
        }

    def create_valid_mcp_config_type_url(self) -> Dict[str, Any]:
        """Create a valid MCP configuration with type 'url'."""
        return {
            "type": "url",
            "url": "https://mcp.example.com",
            "name": "test-mcp",
        }

    def test_mcp_config_validation_type_mcp_valid(self) -> None:
        """Valid MCP config with type 'mcp' passes validation."""
        mcp_config = self.create_valid_mcp_config_type_mcp()
        config = {
            "llm_provider": "anthropic",
            "llm_model": "claude-sonnet-4",
            "tools": [mcp_config],
        }

        task = LLMTask("mcp_test", config)
        assert len(task.mcp_tools) == 1
        assert task.mcp_tools[0] == mcp_config

    def test_mcp_config_validation_type_url_valid(self) -> None:
        """Valid MCP config with type 'url' passes validation."""
        mcp_config = self.create_valid_mcp_config_type_url()
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "tools": [mcp_config],
        }

        task = LLMTask("mcp_url_test", config)
        assert len(task.mcp_tools) == 1
        assert task.mcp_tools[0] == mcp_config

    def test_mcp_config_validation_missing_type(self) -> None:
        """MCP config missing 'type' field fails validation."""
        invalid_config = {
            "server_label": "test",
            "server_url": "https://example.com",
        }
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "tools": [invalid_config],
        }

        result = LLMTask("invalid_mcp", config).run()
        assert result.is_error()
        assert "tools failed custom validation" in result.error

    def test_mcp_config_validation_invalid_type(self) -> None:
        """MCP config with invalid type fails validation."""
        invalid_config = {
            "type": "invalid_type",
            "server_url": "https://example.com",
        }
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "tools": [invalid_config],
        }

        result = LLMTask("invalid_type", config).run()
        assert result.is_error()
        assert "tools failed custom validation" in result.error

    def test_mcp_config_type_mcp_missing_server_label(self) -> None:
        """MCP config type 'mcp' missing server_label fails validation."""
        invalid_config = {
            "type": "mcp",
            "server_url": "https://example.com",
        }
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "tools": [invalid_config],
        }

        result = LLMTask("missing_label", config).run()
        assert result.is_error()
        assert "tools failed custom validation" in result.error

    def test_mcp_config_type_mcp_missing_server_url(self) -> None:
        """MCP config type 'mcp' missing server_url fails validation."""
        invalid_config = {
            "type": "mcp",
            "server_label": "test",
        }
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "tools": [invalid_config],
        }

        result = LLMTask("missing_url", config).run()
        assert result.is_error()
        assert "tools failed custom validation" in result.error

    def test_mcp_config_type_url_missing_url(self) -> None:
        """MCP config type 'url' missing url field fails validation."""
        invalid_config = {
            "type": "url",
            "name": "test-mcp",
        }
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "tools": [invalid_config],
        }

        result = LLMTask("missing_url_field", config).run()
        assert result.is_error()
        assert "tools failed custom validation" in result.error

    def test_mcp_config_type_url_missing_name(self) -> None:
        """MCP config type 'url' missing name field fails validation."""
        invalid_config = {
            "type": "url",
            "url": "https://example.com",
        }
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "tools": [invalid_config],
        }

        result = LLMTask("missing_name_field", config).run()
        assert result.is_error()
        assert "tools failed custom validation" in result.error


class TestLLMTaskMixedTools:
    """Test mixed Python tools and MCP servers."""

    def create_test_python_tool(self) -> Any:
        """Create a test Python tool."""

        @tool
        def calculate(expression: str) -> float:
            """Perform mathematical calculations.

            Args:
                expression: Mathematical expression to evaluate
            """
            return float(eval(expression))  # Unsafe but OK for tests

        return calculate

    def create_mcp_config(self) -> Dict[str, Any]:
        """Create a test MCP configuration."""
        return {
            "type": "mcp",
            "server_label": "test_server",
            "server_url": "https://mcp.example.com",
        }

    def test_mixed_tools_setup(self) -> None:
        """LLMTask correctly separates Python tools from MCP configs."""
        python_tool = self.create_test_python_tool()
        mcp_config = self.create_mcp_config()

        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "tools": [python_tool, mcp_config],
        }

        task = LLMTask("mixed_tools", config)

        # Check Python tools setup
        assert task.supports_tools is True
        assert task.tool_registry is not None
        assert task.tool_executor is not None

        # Check MCP tools setup
        assert len(task.mcp_tools) == 1
        assert task.mcp_tools[0] == mcp_config

    def test_python_tools_only(self) -> None:
        """LLMTask with only Python tools has no MCP configs."""
        python_tool = self.create_test_python_tool()

        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "tools": [python_tool],
        }

        task = LLMTask("python_only", config)

        assert task.supports_tools is True
        assert len(task.mcp_tools) == 0

    def test_mcp_tools_only(self) -> None:
        """LLMTask with only MCP configs has no Python tool registry."""
        mcp_config = self.create_mcp_config()

        config = {
            "llm_provider": "anthropic",
            "llm_model": "claude-sonnet-4",
            "tools": [mcp_config],
        }

        task = LLMTask("mcp_only", config)

        assert task.supports_tools is False  # No Python tools
        assert task.tool_registry is None
        assert len(task.mcp_tools) == 1

    def test_multiple_mcp_servers(self) -> None:
        """LLMTask can handle multiple MCP server configurations."""
        mcp_config1 = {
            "type": "mcp",
            "server_label": "server1",
            "server_url": "https://mcp1.example.com",
        }
        mcp_config2 = {
            "type": "url",
            "url": "https://mcp2.example.com",
            "name": "server2",
        }

        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "tools": [mcp_config1, mcp_config2],
        }

        task = LLMTask("multiple_mcp", config)

        assert len(task.mcp_tools) == 2
        assert task.mcp_tools[0] == mcp_config1
        assert task.mcp_tools[1] == mcp_config2

    def test_complex_mixed_tools(self) -> None:
        """LLMTask handles complex mix of Python tools and MCP servers."""

        @tool
        def tool1(x: int) -> int:
            """Test tool 1."""
            return x * 2

        @tool
        def tool2(y: str) -> str:
            """Test tool 2."""
            return y.upper()

        mcp1 = {
            "type": "mcp",
            "server_label": "mcp1",
            "server_url": "https://mcp1.example.com",
        }
        mcp2 = {
            "type": "url",
            "url": "https://mcp2.example.com",
            "name": "mcp2",
        }

        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "tools": [tool1, mcp1, tool2, mcp2],
        }

        task = LLMTask("complex_mix", config)

        # Check Python tools
        assert task.supports_tools is True
        assert task.tool_registry is not None
        assert task.tool_registry.get_tool_count() == 2

        # Check MCP tools
        assert len(task.mcp_tools) == 2
        assert task.mcp_tools[0] == mcp1
        assert task.mcp_tools[1] == mcp2


class TestLLMTaskMCPOptionalFields:
    """Test MCP configuration optional fields."""

    def test_mcp_config_with_allowed_tools(self) -> None:
        """MCP config with allowed_tools field is accepted."""
        mcp_config = {
            "type": "mcp",
            "server_label": "test",
            "server_url": "https://example.com",
            "allowed_tools": ["tool1", "tool2"],
        }

        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "tools": [mcp_config],
        }

        task = LLMTask("mcp_allowed_tools", config)
        assert len(task.mcp_tools) == 1
        assert task.mcp_tools[0]["allowed_tools"] == ["tool1", "tool2"]

    def test_mcp_config_with_require_approval(self) -> None:
        """MCP config with require_approval field is accepted."""
        mcp_config = {
            "type": "mcp",
            "server_label": "test",
            "server_url": "https://example.com",
            "require_approval": "never",
        }

        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "tools": [mcp_config],
        }

        task = LLMTask("mcp_approval", config)
        assert len(task.mcp_tools) == 1
        assert task.mcp_tools[0]["require_approval"] == "never"

    def test_mcp_config_with_multiple_optional_fields(self) -> None:
        """MCP config with multiple optional fields is accepted."""
        mcp_config = {
            "type": "mcp",
            "server_label": "test",
            "server_url": "https://example.com",
            "allowed_tools": ["search", "summarize"],
            "require_approval": "never",
            "custom_field": "custom_value",
        }

        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "tools": [mcp_config],
        }

        task = LLMTask("mcp_multiple_fields", config)
        assert len(task.mcp_tools) == 1
        # All fields should be preserved
        assert task.mcp_tools[0]["allowed_tools"] == ["search", "summarize"]
        assert task.mcp_tools[0]["custom_field"] == "custom_value"


class TestLLMTaskMCPToolsCombination:
    """Test that MCP tools are properly combined with Python tools in LLM calls."""

    @patch("aipype.llm_task.litellm.completion")
    def test_mcp_tools_passed_to_litellm(self, mock_completion: MagicMock) -> None:
        """MCP configurations are passed to litellm.completion()."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )
        mock_response.model = "gpt-4"
        mock_completion.return_value = mock_response

        mcp_config = {
            "type": "mcp",
            "server_label": "test_server",
            "server_url": "https://mcp.example.com",
        }

        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "prompt": "Test prompt",
            "tools": [mcp_config],
        }

        task = LLMTask("mcp_litellm_test", config)
        result = task.run()

        # Check that completion was called
        assert mock_completion.called
        call_args = mock_completion.call_args

        # Verify MCP config was passed in tools parameter
        assert "tools" in call_args.kwargs
        tools = call_args.kwargs["tools"]
        assert len(tools) == 1
        assert tools[0] == mcp_config

        # Verify task completed successfully
        assert result.is_success()

    @patch("aipype.llm_task.litellm.completion")
    def test_mixed_tools_passed_to_litellm(self, mock_completion: MagicMock) -> None:
        """Mixed Python tools and MCP configs are both passed to litellm."""

        @tool
        def calculate(x: int) -> int:
            """Calculate double."""
            return x * 2

        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )
        mock_response.model = "gpt-4"
        mock_completion.return_value = mock_response

        mcp_config = {
            "type": "mcp",
            "server_label": "test",
            "server_url": "https://example.com",
        }

        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "prompt": "Test prompt",
            "tools": [calculate, mcp_config],
        }

        task = LLMTask("mixed_litellm_test", config)
        result = task.run()

        # Check that completion was called
        assert mock_completion.called
        call_args = mock_completion.call_args

        # Verify both Python tool schema and MCP config were passed
        assert "tools" in call_args.kwargs
        tools = call_args.kwargs["tools"]
        assert len(tools) == 2

        # First tool should be Python tool schema (OpenAI format)
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "calculate"

        # Second tool should be MCP config
        assert tools[1] == mcp_config

        # Verify task completed successfully
        assert result.is_success()
