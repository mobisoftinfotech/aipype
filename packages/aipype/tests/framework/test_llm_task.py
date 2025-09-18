"""Comprehensive tests for LLMTask - LLM task with context-aware prompt generation."""
# pyright: reportPrivateUsage=false

import os
import pytest
import tempfile
import json
import threading
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import Mock, patch


from aipype import LLMTask, TaskContext, TaskDependency, DependencyType


class TestLLMTaskInitialization:
    """Test LLMTask initialization and basic configuration."""

    def test_initialization_minimal_config(self) -> None:
        """LLMTask initializes with minimal required configuration."""
        config = {"llm_provider": "openai", "llm_model": "gpt-3.5-turbo"}

        task = LLMTask("test_llm", config)

        assert task.name == "test_llm"
        assert task.config["llm_provider"] == "openai"
        assert task.config["llm_model"] == "gpt-3.5-turbo"
        assert task.context_instance is None
        assert task.resolved_prompt is None
        assert task.resolved_context is None
        assert task.resolved_role is None

    def test_initialization_full_config(self) -> None:
        """LLMTask initializes with full configuration."""
        dependencies = [TaskDependency("data", "source.data", DependencyType.REQUIRED)]

        config = {
            "llm_provider": "anthropic",
            "llm_model": "claude-3-sonnet",
            "prompt_template": "Process ${data}",
            "context": "You are an expert analyst",
            "role": "Data Processor",
            "temperature": 0.5,
            "max_tokens": 500,
            "timeout": 30,
            "api_key": "test-key",
            "api_base": "https://custom.api.com",
        }

        task = LLMTask("full_llm", config, dependencies)

        assert task.name == "full_llm"
        assert len(task.get_dependencies()) == 1
        assert task.config["temperature"] == 0.5
        assert task.config["max_tokens"] == 500
        assert task.config["timeout"] == 30

    def test_initialization_sets_logs_file_from_env(self) -> None:
        """LLMTask sets logs file from environment variable."""
        with patch.dict(os.environ, {"MI_AGENT_LLM_LOGS_FILE": "/tmp/test_logs.jsonl"}):
            task = LLMTask("test", {"llm_provider": "openai", "llm_model": "gpt-4"})
            assert task.logs_file == "/tmp/test_logs.jsonl"

    def test_initialization_configures_litellm_timeout(self) -> None:
        """LLMTask configures litellm timeout when specified."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4", "timeout": 45}

        with patch("aipype.framework.llm_task.litellm") as mock_litellm:
            LLMTask("timeout_test", config)
            assert mock_litellm.request_timeout == 45

    def test_initialization_disables_litellm_verbose_logging(self) -> None:
        """LLMTask disables litellm verbose logging."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4"}

        with patch("aipype.framework.llm_task.litellm") as mock_litellm:
            LLMTask("verbose_test", config)
            assert mock_litellm.set_verbose is False


class TestLLMTaskConfigurationValidation:
    """Test LLMTask configuration validation."""

    def test_missing_required_config_provider(self) -> None:
        """LLMTask returns validation error when llm_provider is missing."""
        config = {"llm_model": "gpt-4"}

        task = LLMTask("missing_provider", config)
        result = task.run()
        assert result.is_error()
        assert (
            result.error is not None
            and "Missing required fields: ['llm_provider']" in result.error
        )

    def test_missing_required_config_model(self) -> None:
        """LLMTask returns validation error when llm_model is missing."""
        config = {"llm_provider": "openai"}

        task = LLMTask("missing_model", config)
        result = task.run()
        assert result.is_error()
        assert (
            result.error is not None
            and "Missing required fields: ['llm_model']" in result.error
        )

    def test_missing_both_required_configs(self) -> None:
        """LLMTask returns validation error when both required configs are missing."""
        config: Dict[str, Any] = {}

        task = LLMTask("missing_both", config)
        result = task.run()
        assert result.is_error()
        assert (
            result.error is not None
            and "Missing required fields: ['llm_provider', 'llm_model']" in result.error
        )

    def test_empty_provider_value(self) -> None:
        """LLMTask returns validation error when llm_provider is empty string."""
        config = {"llm_provider": "", "llm_model": "gpt-4"}

        task = LLMTask("empty_provider", config)
        result = task.run()
        assert result.is_error()
        assert (
            result.error is not None
            and "llm_provider failed custom validation" in result.error
        )

    def test_empty_model_value(self) -> None:
        """LLMTask returns validation error when llm_model is empty string."""
        config = {"llm_provider": "openai", "llm_model": ""}

        task = LLMTask("empty_model", config)
        result = task.run()
        assert result.is_error()
        assert (
            result.error is not None
            and "llm_model failed custom validation" in result.error
        )

    def test_none_provider_value(self) -> None:
        """LLMTask returns validation error when llm_provider is None."""
        config = {"llm_provider": None, "llm_model": "gpt-4"}

        task = LLMTask("none_provider", config)
        result = task.run()
        assert result.is_error()
        assert (
            result.error is not None
            and "Missing required fields: ['llm_provider']" in result.error
        )

    def test_none_model_value(self) -> None:
        """LLMTask returns validation error when llm_model is None."""
        config = {"llm_provider": "openai", "llm_model": None}

        task = LLMTask("none_model", config)
        result = task.run()
        assert result.is_error()
        assert (
            result.error is not None
            and "Missing required fields: ['llm_model']" in result.error
        )


class TestLLMTaskAPIConfiguration:
    """Test LLMTask API key and base URL configuration."""

    def test_get_api_key_from_config(self) -> None:
        """API key is retrieved from task config when provided."""
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "api_key": "config-api-key",
        }

        task = LLMTask("config_key_test", config)
        assert task._get_api_key() == "config-api-key"

    def test_get_api_key_none_in_config(self) -> None:
        """API key returns None when explicitly set to None in config."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4", "api_key": None}

        task = LLMTask("none_key_test", config)
        assert task._get_api_key() is None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-openai-key"})
    def test_get_api_key_from_environment_openai(self) -> None:
        """API key is retrieved from environment for OpenAI provider."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4"}

        task = LLMTask("env_openai_test", config)
        assert task._get_api_key() == "env-openai-key"

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-anthropic-key"})
    def test_get_api_key_from_environment_anthropic(self) -> None:
        """API key is retrieved from environment for Anthropic provider."""
        config = {"llm_provider": "anthropic", "llm_model": "claude-3-sonnet"}

        task = LLMTask("env_anthropic_test", config)
        assert task._get_api_key() == "env-anthropic-key"

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-claude-key"})
    def test_get_api_key_from_environment_claude_alias(self) -> None:
        """API key is retrieved from environment for Claude alias."""
        config = {"llm_provider": "claude", "llm_model": "claude-3-haiku"}

        task = LLMTask("env_claude_test", config)
        assert task._get_api_key() == "env-claude-key"

    def test_get_api_key_ollama_returns_none(self) -> None:
        """API key returns None for Ollama provider (no key required)."""
        config = {"llm_provider": "ollama", "llm_model": "llama2"}

        task = LLMTask("ollama_test", config)
        assert task._get_api_key() is None

    def test_get_api_key_unknown_provider_returns_none(self) -> None:
        """API key returns None for unknown provider."""
        config = {"llm_provider": "unknown_provider", "llm_model": "unknown_model"}

        task = LLMTask("unknown_test", config)
        assert task._get_api_key() is None

    def test_get_api_base_from_config(self) -> None:
        """API base URL is retrieved from task config when provided."""
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "api_base": "https://custom.openai.com",
        }

        task = LLMTask("config_base_test", config)
        assert task._get_api_base() == "https://custom.openai.com"

    def test_get_api_base_none_in_config(self) -> None:
        """API base URL returns None when explicitly set to None in config."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4", "api_base": None}

        task = LLMTask("none_base_test", config)
        assert task._get_api_base() is None

    @patch.dict(os.environ, {"OLLAMA_API_BASE": "http://localhost:11434"})
    def test_get_api_base_from_environment_ollama(self) -> None:
        """API base URL is retrieved from environment for Ollama."""
        config = {"llm_provider": "ollama", "llm_model": "llama2"}

        task = LLMTask("env_ollama_base_test", config)
        assert task._get_api_base() == "http://localhost:11434"

    @patch.dict(os.environ, {"OPENAI_API_BASE": "https://custom-openai.api.com"})
    def test_get_api_base_from_environment_openai(self) -> None:
        """API base URL is retrieved from environment for OpenAI."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4"}

        task = LLMTask("env_openai_base_test", config)
        assert task._get_api_base() == "https://custom-openai.api.com"

    def test_build_model_name_openai_no_prefix(self) -> None:
        """Model name is built without prefix for OpenAI."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4"}

        task = LLMTask("openai_model_test", config)
        assert task._build_model_name() == "gpt-4"

    def test_build_model_name_anthropic_with_prefix(self) -> None:
        """Model name is built with prefix for Anthropic."""
        config = {"llm_provider": "anthropic", "llm_model": "claude-3-sonnet"}

        task = LLMTask("anthropic_model_test", config)
        assert task._build_model_name() == "anthropic/claude-3-sonnet"

    def test_build_model_name_ollama_with_prefix(self) -> None:
        """Model name is built with prefix for Ollama."""
        config = {"llm_provider": "ollama", "llm_model": "llama2"}

        task = LLMTask("ollama_model_test", config)
        assert task._build_model_name() == "ollama/llama2"

    def test_build_model_name_claude_alias_with_prefix(self) -> None:
        """Model name is built with anthropic prefix for Claude alias."""
        config = {"llm_provider": "claude", "llm_model": "claude-3-haiku"}

        task = LLMTask("claude_model_test", config)
        assert task._build_model_name() == "anthropic/claude-3-haiku"


class TestLLMTaskTemplateResolution:
    """Test LLMTask template resolution system."""

    def test_set_context_stores_instance(self) -> None:
        """set_context method stores TaskContext instance."""
        task = LLMTask("test", {"llm_provider": "openai", "llm_model": "gpt-4"})
        context = TaskContext()

        task.set_context(context)

        assert task.context_instance is context

    def test_resolve_template_string_with_simple_substitution(self) -> None:
        """Template resolution works with simple variable substitution."""
        task = LLMTask("test", {"llm_provider": "openai", "llm_model": "gpt-4"})
        context = TaskContext()
        context.store_result("search", {"query": "test query"})
        task.set_context(context)

        # Add variable to config (simulating dependency resolution)
        task.config["search_query"] = "test query"

        template = "Search for: ${search_query}"
        result = task._resolve_template_string(template)

        assert result == "Search for: test query"

    def test_resolve_template_string_with_path_resolution(self) -> None:
        """Template resolution works with path-based variable resolution."""
        task = LLMTask("test", {"llm_provider": "openai", "llm_model": "gpt-4"})
        context = TaskContext()
        context.store_result("search", {"query": "AI trends", "count": 5})
        task.set_context(context)

        template = "Found ${search.count} results for ${search.query}"
        result = task._resolve_template_string(template)

        assert result == "Found 5 results for AI trends"

    def test_resolve_template_string_with_missing_variable(self) -> None:
        """Template resolution handles missing variables gracefully."""
        task = LLMTask("test", {"llm_provider": "openai", "llm_model": "gpt-4"})
        context = TaskContext()
        task.set_context(context)

        template = "Process ${missing_variable} data"

        with patch.object(task.logger, "warning") as mock_warning:
            result = task._resolve_template_string(template)
            mock_warning.assert_called_with(
                "Could not resolve template variable: missing_variable"
            )

        assert result == "Process ${missing_variable} data"  # Unchanged

    def test_resolve_template_string_with_error_in_resolution(self) -> None:
        """Template resolution handles errors during variable resolution."""
        task = LLMTask("test", {"llm_provider": "openai", "llm_model": "gpt-4"})
        context = TaskContext()
        task.set_context(context)

        # Mock context to raise an exception
        with patch.object(
            context, "get_path_value", side_effect=RuntimeError("Context error")
        ):
            template = "Process ${error_variable} data"

            with patch.object(task.logger, "error") as mock_error:
                result = task._resolve_template_string(template)
                mock_error.assert_called_with(
                    "Error resolving template variable 'error_variable': Context error"
                )

            assert result == "Process ${error_variable} data"  # Unchanged

    def test_format_list_for_prompt_simple_values(self) -> None:
        """List formatting works with simple values."""
        task = LLMTask("test", {"llm_provider": "openai", "llm_model": "gpt-4"})

        simple_list = [1, 2, 3, 4, 5]
        result = task._format_list_for_prompt(simple_list)

        assert result == "1, 2, 3, 4, 5"

    def test_format_list_for_prompt_mixed_values(self) -> None:
        """List formatting works with mixed value types."""
        task = LLMTask("test", {"llm_provider": "openai", "llm_model": "gpt-4"})

        mixed_list = [1, "text", 3.14]
        result = task._format_list_for_prompt(mixed_list)

        assert result == "1, text, 3.14"

    def test_format_list_for_prompt_articles(self) -> None:
        """List formatting works with article-like dictionaries."""
        task = LLMTask("test", {"llm_provider": "openai", "llm_model": "gpt-4"})

        articles = [
            {"title": "Article 1", "content": "Content 1"},
            {"title": "Article 2", "content": "Content 2"},
        ]
        result = task._format_list_for_prompt(articles)

        expected = "Article 1: Article 1\nContent 1\n\nArticle 2: Article 2\nContent 2"
        assert result == expected

    def test_format_list_for_prompt_search_results(self) -> None:
        """List formatting works with search result-like dictionaries."""
        task = LLMTask("test", {"llm_provider": "openai", "llm_model": "gpt-4"})

        search_results = [
            {"url": "http://example1.com", "title": "Result 1"},
            {"url": "http://example2.com", "title": "Result 2"},
        ]
        result = task._format_list_for_prompt(search_results)

        expected = "Result 1: Result 1 (http://example1.com)\n\nResult 2: Result 2 (http://example2.com)"
        assert result == expected

    def test_format_list_for_prompt_complex_mixed(self) -> None:
        """List formatting works with complex mixed content."""
        task = LLMTask("test", {"llm_provider": "openai", "llm_model": "gpt-4"})

        complex_list = ["simple string", 42, {"custom": "object"}]
        result = task._format_list_for_prompt(complex_list)

        expected = "- simple string\n- 42\n- {'custom': 'object'}"
        assert result == expected

    def test_format_list_for_prompt_empty_list(self) -> None:
        """List formatting handles empty lists."""
        task = LLMTask("test", {"llm_provider": "openai", "llm_model": "gpt-4"})

        empty_list: List[Any] = []
        result = task._format_list_for_prompt(empty_list)

        assert result == "[]"

    def test_format_dict_for_prompt_article_structure(self) -> None:
        """Dict formatting works with article-like structure."""
        task = LLMTask("test", {"llm_provider": "openai", "llm_model": "gpt-4"})

        article = {"title": "Test Article", "content": "Article content here"}
        result = task._format_dict_for_prompt(article)

        assert result == "Title: Test Article\nContent: Article content here"

    def test_format_dict_for_prompt_search_structure(self) -> None:
        """Dict formatting works with search-like structure."""
        task = LLMTask("test", {"llm_provider": "openai", "llm_model": "gpt-4"})

        search = {"query": "test query", "results": [1, 2, 3]}
        result = task._format_dict_for_prompt(search)

        assert result == "Search Query: test query\nResults: 3 items"

    def test_format_dict_for_prompt_url_structure(self) -> None:
        """Dict formatting works with URL-like structure."""
        task = LLMTask("test", {"llm_provider": "openai", "llm_model": "gpt-4"})

        url_item = {"url": "http://example.com", "title": "Example Page"}
        result = task._format_dict_for_prompt(url_item)

        assert result == "Example Page (http://example.com)"

    def test_format_dict_for_prompt_generic_structure(self) -> None:
        """Dict formatting works with generic structure."""
        task = LLMTask("test", {"llm_provider": "openai", "llm_model": "gpt-4"})

        generic = {
            "name": "test",
            "value": 42,
            "active": True,
            "tags": ["a", "b", "c"],
            "complex": {"nested": "data"},
        }
        result = task._format_dict_for_prompt(generic)

        assert "name: test" in result
        assert "value: 42" in result
        assert "active: True" in result
        assert "tags: [a, b, c]" in result
        assert "complex: dict" in result

    def test_format_dict_for_prompt_empty_dict(self) -> None:
        """Dict formatting handles empty dictionaries."""
        task = LLMTask("test", {"llm_provider": "openai", "llm_model": "gpt-4"})

        empty_dict: Dict[str, Any] = {}
        result = task._format_dict_for_prompt(empty_dict)

        assert result == "{}"

    def test_resolve_templates_all_types(self) -> None:
        """Template resolution works for all template types (prompt, context, role)."""
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "prompt_template": "Analyze ${topic} trends",
            "context": "You are analyzing ${domain} data",
            "role": "Expert in ${field}",
        }

        task = LLMTask("template_test", config)
        context = TaskContext()
        context.store_result(
            "vars", {"topic": "AI", "domain": "technology", "field": "machine learning"}
        )
        task.set_context(context)

        # Simulate config population from dependencies
        task.config.update(
            {"topic": "AI", "domain": "technology", "field": "machine learning"}
        )

        task._resolve_templates()

        assert task.resolved_prompt == "Analyze AI trends"
        assert task.resolved_context == "You are analyzing technology data"
        assert task.resolved_role == "Expert in machine learning"

    def test_resolve_templates_no_context(self) -> None:
        """Template resolution handles missing context gracefully."""
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "prompt_template": "Analyze ${topic} trends",
        }

        task = LLMTask("no_context_test", config)

        with patch.object(task.logger, "warning") as mock_warning:
            task._resolve_templates()
            mock_warning.assert_called_with(
                "Could not resolve template variable: topic"
            )

        # Template resolution should still happen, but variable won't be resolved
        assert task.resolved_prompt == "Analyze ${topic} trends"


class TestLLMTaskMessagePreparation:
    """Test LLM message preparation functionality."""

    def test_prepare_messages_prompt_only(self) -> None:
        """Message preparation works with prompt only."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4"}

        task = LLMTask("message_test", config)
        task.resolved_prompt = "Test prompt"

        messages = task._prepare_messages()

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Test prompt"

    def test_prepare_messages_with_context(self) -> None:
        """Message preparation works with context."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4"}

        task = LLMTask("message_test", config)
        task.resolved_prompt = "Test prompt"
        task.resolved_context = "You are a helpful assistant"

        messages = task._prepare_messages()

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Test prompt"

    def test_prepare_messages_with_role(self) -> None:
        """Message preparation works with role."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4"}

        task = LLMTask("message_test", config)
        task.resolved_prompt = "Test prompt"
        task.resolved_role = "Data Analyst"

        messages = task._prepare_messages()

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Your role: Data Analyst"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Test prompt"

    def test_prepare_messages_with_context_and_role(self) -> None:
        """Message preparation works with both context and role."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4"}

        task = LLMTask("message_test", config)
        task.resolved_prompt = "Test prompt"
        task.resolved_context = "You are a helpful assistant"
        task.resolved_role = "Data Analyst"

        messages = task._prepare_messages()

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert (
            messages[0]["content"]
            == "You are a helpful assistant Your role: Data Analyst"
        )
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Test prompt"

    def test_prepare_messages_fallback_to_config(self) -> None:
        """Message preparation falls back to config values when resolved values are None."""
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "prompt": "Config prompt",
            "context": "Config context",
        }

        task = LLMTask("fallback_test", config)
        # Don't set resolved values, should use config

        messages = task._prepare_messages()

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Config context"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Config prompt"

    def test_prepare_messages_empty_prompt(self) -> None:
        """Message preparation handles empty prompt."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4"}

        task = LLMTask("empty_prompt_test", config)

        messages = task._prepare_messages()

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == ""


class TestLLMTaskExecution:
    """Test LLM task execution and response processing."""

    def test_make_llm_call_success(self) -> None:
        """LLM API call succeeds with proper parameters."""
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000,
        }

        task = LLMTask("api_test", config)
        task.resolved_prompt = "Test prompt"

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"

        with patch(
            "aipype.framework.llm_task.litellm.completion", return_value=mock_response
        ) as mock_completion:
            result = task._make_llm_call()

            # Verify call parameters
            call_args = mock_completion.call_args[1]
            assert call_args["model"] == "gpt-4"
            assert call_args["temperature"] == 0.7
            assert call_args["max_tokens"] == 1000
            assert call_args["stream"] is False
            assert len(call_args["messages"]) == 1
            assert call_args["messages"][0]["content"] == "Test prompt"

            assert result == mock_response

    def test_make_llm_call_with_api_key(self) -> None:
        """LLM API call includes API key when available."""
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "api_key": "test-api-key",
        }

        task = LLMTask("api_key_test", config)
        task.resolved_prompt = "Test prompt"

        mock_response = Mock()

        with patch(
            "aipype.framework.llm_task.litellm.completion", return_value=mock_response
        ) as mock_completion:
            task._make_llm_call()

            call_args = mock_completion.call_args[1]
            assert call_args["api_key"] == "test-api-key"

    def test_make_llm_call_with_api_base(self) -> None:
        """LLM API call includes API base when available."""
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "api_base": "https://custom.api.com",
        }

        task = LLMTask("api_base_test", config)
        task.resolved_prompt = "Test prompt"

        mock_response = Mock()

        with patch(
            "aipype.framework.llm_task.litellm.completion", return_value=mock_response
        ) as mock_completion:
            task._make_llm_call()

            call_args = mock_completion.call_args[1]
            assert call_args["api_base"] == "https://custom.api.com"

    def test_make_llm_call_with_streaming(self) -> None:
        """LLM API call includes streaming parameter when enabled."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4", "stream": True}

        task = LLMTask("stream_test", config)
        task.resolved_prompt = "Test prompt"

        mock_response = Mock()

        with patch(
            "aipype.framework.llm_task.litellm.completion", return_value=mock_response
        ) as mock_completion:
            task._make_llm_call()

            call_args = mock_completion.call_args[1]
            assert call_args["stream"] is True

    def test_make_llm_call_failure(self) -> None:
        """LLM API call handles failures properly."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4"}

        task = LLMTask("failure_test", config)
        task.resolved_prompt = "Test prompt"

        with patch(
            "aipype.framework.llm_task.litellm.completion",
            side_effect=RuntimeError("API Error"),
        ):
            with pytest.raises(
                RuntimeError,
                match="LLMTask API call operation failed: LLM API call to.*failed: API Error",
            ):
                task._make_llm_call()

    def test_process_response_complete(self) -> None:
        """Response processing works with complete response data."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4"}

        task = LLMTask("response_test", config)

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated content"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4-turbo"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 25
        mock_response.usage.total_tokens = 75

        result = task._process_response(mock_response)

        assert result["content"] == "Generated content"
        assert result["model"] == "gpt-4-turbo"
        assert result["provider"] == "openai"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 50
        assert result["usage"]["completion_tokens"] == 25
        assert result["usage"]["total_tokens"] == 75

    def test_process_response_minimal(self) -> None:
        """Response processing works with minimal response data."""
        config = {"llm_provider": "anthropic", "llm_model": "claude-3-sonnet"}

        task = LLMTask("minimal_response_test", config)

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Minimal content"
        mock_response.model = "claude-3-sonnet-20240229"
        mock_response.usage = None
        # Remove finish_reason attribute to simulate it not being present
        del mock_response.choices[0].finish_reason

        result = task._process_response(mock_response)

        assert result["content"] == "Minimal content"
        assert result["model"] == "claude-3-sonnet-20240229"
        assert result["provider"] == "anthropic"
        assert "usage" not in result
        assert "finish_reason" not in result

    def test_process_response_failure(self) -> None:
        """Response processing handles processing errors."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4"}

        task = LLMTask("process_error_test", config)

        # Mock response that will cause an error (missing choices)
        mock_response = Mock()
        mock_response.choices = []

        with pytest.raises(RuntimeError, match="Failed to process LLM response"):
            task._process_response(mock_response)

    def test_run_complete_execution(self) -> None:
        """Complete run execution works end-to-end."""
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "prompt_template": "Analyze ${topic}",
            "temperature": 0.5,
        }

        dependencies = [
            TaskDependency("topic", "source.topic", DependencyType.REQUIRED)
        ]

        task = LLMTask("complete_run_test", config, dependencies)

        # Set up context
        context = TaskContext()
        context.store_result("source", {"topic": "machine learning"})
        task.set_context(context)

        # Simulate dependency resolution
        task.config["topic"] = "machine learning"

        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "ML analysis complete"
        mock_response.model = "gpt-4-turbo"

        with patch(
            "aipype.framework.llm_task.litellm.completion", return_value=mock_response
        ):
            result = task.run()
            assert result.is_success()
            assert result.data["content"] == "ML analysis complete"
            assert result.data["model"] == "gpt-4-turbo"
            assert result.data["provider"] == "openai"
            assert task.resolved_prompt == "Analyze machine learning"

    def test_run_execution_failure(self) -> None:
        """Run handles execution failures properly."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4"}

        task = LLMTask("run_failure_test", config)

        with patch.object(
            task, "_make_llm_call", side_effect=RuntimeError("Execution failed")
        ):
            result = task.run()
            assert result.is_error()
            assert result.error is not None and (
                "Contextual LLM task 'run_failure_test' failed: Execution failed"
                in result.error
            )


class TestLLMTaskLogging:
    """Test LLM interaction logging functionality."""

    def test_prepare_input_log_data(self) -> None:
        """Input log data preparation includes all relevant fields."""
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "temperature": 0.8,
            "max_tokens": 500,
            "stream": True,
        }

        task = LLMTask("log_test", config)
        task.resolved_prompt = "Test prompt"
        task.resolved_context = "Test context"
        task.resolved_role = "Test role"

        input_data = task._prepare_input_log_data()

        assert input_data["prompt"] == "Test prompt"
        assert input_data["context"] == "Test context"
        assert input_data["role"] == "Test role"
        assert input_data["temperature"] == 0.8
        assert input_data["max_tokens"] == 500
        assert input_data["stream"] is True

    def test_prepare_input_log_data_removes_none_values(self) -> None:
        """Input log data preparation removes None values."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4"}

        task = LLMTask("log_none_test", config)
        # Don't set resolved values (they'll be None)

        input_data = task._prepare_input_log_data()

        assert "context" not in input_data
        assert "role" not in input_data
        assert input_data["prompt"] == ""
        assert input_data["temperature"] == task.DEFAULT_TEMPERATURE
        assert input_data["max_tokens"] == task.DEFAULT_MAX_TOKENS
        assert input_data["stream"] is False

    def test_log_llm_interaction_success(self) -> None:
        """LLM interaction logging works successfully."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4"}

        task = LLMTask("logging_test", config)
        task.agent_name = "test_agent"

        input_data = {"prompt": "test prompt", "temperature": 0.7}
        output_data = {"content": "test response", "usage": {"total_tokens": 100}}

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_path = temp_file.name
            task.logs_file = temp_path

            task._log_llm_interaction(input_data, output_data)

            # Read and verify log entry
            with open(temp_path, "r") as f:
                log_line = f.read().strip()
                log_entry = json.loads(log_line)

                assert log_entry["agent_name"] == "test_agent"
                assert log_entry["task_name"] == "logging_test"
                assert log_entry["provider"] == "openai"
                assert log_entry["model"] == "gpt-4"
                assert log_entry["input"] == input_data
                assert log_entry["output"] == output_data
                assert "timestamp" in log_entry

            # Cleanup
            os.unlink(temp_path)

    def test_log_llm_interaction_no_logs_file(self) -> None:
        """LLM interaction logging does nothing when no logs file is set."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4"}

        task = LLMTask("no_logging_test", config)
        task.logs_file = None

        input_data = {"prompt": "test"}
        output_data = {"content": "test"}

        # Should not raise an error and complete silently
        task._log_llm_interaction(input_data, output_data)

    def test_log_llm_interaction_write_failure(self) -> None:
        """LLM interaction logging handles write failures gracefully."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4"}

        task = LLMTask("log_failure_test", config)
        task.logs_file = "/invalid/path/logs.jsonl"

        input_data = {"prompt": "test"}
        output_data = {"content": "test"}

        with patch.object(task.logger, "warning") as mock_warning:
            task._log_llm_interaction(input_data, output_data)
            mock_warning.assert_called_once()
            assert "Failed to log LLM interaction" in mock_warning.call_args[0][0]

    @patch.dict(os.environ, {"MI_AGENT_LLM_LOGS_FILE": "/tmp/test_env_logs.jsonl"})
    def test_log_llm_interaction_with_env_variable(self) -> None:
        """LLM interaction logging uses environment variable for log file."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4"}

        task = LLMTask("env_logging_test", config)
        assert task.logs_file == "/tmp/test_env_logs.jsonl"


class TestLLMTaskUtilityMethods:
    """Test LLM task utility methods and string representations."""

    def test_get_resolved_prompt(self) -> None:
        """get_resolved_prompt returns resolved prompt value."""
        task = LLMTask("test", {"llm_provider": "openai", "llm_model": "gpt-4"})

        assert task.get_resolved_prompt() is None

        task.resolved_prompt = "Resolved test prompt"
        assert task.get_resolved_prompt() == "Resolved test prompt"

    def test_get_resolved_context(self) -> None:
        """get_resolved_context returns resolved context value."""
        task = LLMTask("test", {"llm_provider": "openai", "llm_model": "gpt-4"})

        assert task.get_resolved_context() is None

        task.resolved_context = "Resolved test context"
        assert task.get_resolved_context() == "Resolved test context"

    def test_get_resolved_role(self) -> None:
        """get_resolved_role returns resolved role value."""
        task = LLMTask("test", {"llm_provider": "openai", "llm_model": "gpt-4"})

        assert task.get_resolved_role() is None

        task.resolved_role = "Resolved test role"
        assert task.get_resolved_role() == "Resolved test role"

    def test_preview_resolved_templates_with_context(self) -> None:
        """preview_resolved_templates works with context available."""
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "prompt_template": "Analyze ${topic}",
            "context": "You are an expert in ${field}",
            "role": "Analyst for ${domain}",
        }

        task = LLMTask("preview_test", config)
        context = TaskContext()
        context.store_result(
            "vars", {"topic": "AI", "field": "technology", "domain": "research"}
        )
        task.set_context(context)

        # Simulate dependency resolution
        task.config.update({"topic": "AI", "field": "technology", "domain": "research"})

        preview = task.preview_resolved_templates()

        assert preview["prompt"] == "Analyze AI"
        assert preview["context"] == "You are an expert in technology"
        assert preview["role"] == "Analyst for research"

    def test_preview_resolved_templates_no_context(self) -> None:
        """preview_resolved_templates works when no templates are configured."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4"}

        task = LLMTask("no_context_preview_test", config)

        preview = task.preview_resolved_templates()

        # Should return empty dict when no templates are configured
        assert preview == {}

    def test_preview_resolved_templates_partial_templates(self) -> None:
        """preview_resolved_templates works with partial template configuration."""
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "prompt_template": "Process data",
            # No context or role templates
        }

        task = LLMTask("partial_preview_test", config)
        context = TaskContext()
        task.set_context(context)

        preview = task.preview_resolved_templates()

        assert preview["prompt"] == "Process data"
        assert "context" not in preview
        assert "role" not in preview

    def test_string_representation(self) -> None:
        """String representation includes task details."""
        dependencies = [TaskDependency("data", "source.data", DependencyType.REQUIRED)]

        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "prompt_template": "Analyze ${data}",
        }

        task = LLMTask("str_test", config, dependencies)

        str_repr = str(task)

        assert "ContextualLLMTask" in str_repr
        assert "str_test" in str_repr
        assert "dependencies=1" in str_repr
        assert "has_template=True" in str_repr

    def test_string_representation_no_dependencies_no_template(self) -> None:
        """String representation works with no dependencies and no template."""
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "prompt": "Static prompt",  # No template
        }

        task = LLMTask("simple_str_test", config)

        str_repr = str(task)

        assert "dependencies=0" in str_repr
        assert "has_template=False" in str_repr


class TestLLMTaskProviderIntegration:
    """Test LLM task integration with different providers."""

    def test_openai_provider_configuration(self) -> None:
        """OpenAI provider configuration works correctly."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4-turbo"}

        task = LLMTask("openai_test", config)

        assert task._build_model_name() == "gpt-4-turbo"  # No prefix for OpenAI

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
            assert task._get_api_key() == "sk-test-key"

    def test_anthropic_provider_configuration(self) -> None:
        """Anthropic provider configuration works correctly."""
        config = {"llm_provider": "anthropic", "llm_model": "claude-3-sonnet-20240229"}

        task = LLMTask("anthropic_test", config)

        assert task._build_model_name() == "anthropic/claude-3-sonnet-20240229"

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test-key"}):
            assert task._get_api_key() == "sk-ant-test-key"

    def test_claude_alias_provider_configuration(self) -> None:
        """Claude alias provider configuration works correctly."""
        config = {"llm_provider": "claude", "llm_model": "claude-3-haiku-20240307"}

        task = LLMTask("claude_test", config)

        assert task._build_model_name() == "anthropic/claude-3-haiku-20240307"

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-claude-key"}):
            assert task._get_api_key() == "sk-ant-claude-key"

    def test_ollama_provider_configuration(self) -> None:
        """Ollama provider configuration works correctly."""
        config = {"llm_provider": "ollama", "llm_model": "llama2"}

        task = LLMTask("ollama_test", config)

        assert task._build_model_name() == "ollama/llama2"
        assert task._get_api_key() is None  # No API key for Ollama

        with patch.dict(os.environ, {"OLLAMA_API_BASE": "http://localhost:11434"}):
            assert task._get_api_base() == "http://localhost:11434"

    def test_cohere_provider_configuration(self) -> None:
        """Cohere provider configuration works correctly."""
        config = {"llm_provider": "cohere", "llm_model": "command"}

        task = LLMTask("cohere_test", config)

        assert task._build_model_name() == "cohere/command"

        with patch.dict(os.environ, {"COHERE_API_KEY": "cohere-test-key"}):
            assert task._get_api_key() == "cohere-test-key"

    def test_gemini_provider_configuration(self) -> None:
        """Gemini provider configuration works correctly."""
        config = {"llm_provider": "gemini", "llm_model": "gemini-pro"}

        task = LLMTask("gemini_test", config)

        assert task._build_model_name() == "gemini/gemini-pro"

        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-test-key"}):
            assert task._get_api_key() == "gemini-test-key"

    def test_huggingface_provider_configuration(self) -> None:
        """HuggingFace provider configuration works correctly."""
        config = {
            "llm_provider": "huggingface",
            "llm_model": "mistralai/Mistral-7B-Instruct-v0.1",
        }

        task = LLMTask("hf_test", config)

        assert (
            task._build_model_name() == "huggingface/mistralai/Mistral-7B-Instruct-v0.1"
        )

        with patch.dict(os.environ, {"HUGGINGFACE_API_KEY": "hf-test-key"}):
            assert task._get_api_key() == "hf-test-key"

    def test_azure_provider_configuration(self) -> None:
        """Azure provider configuration works correctly."""
        config = {"llm_provider": "azure", "llm_model": "gpt-4"}

        task = LLMTask("azure_test", config)

        assert task._build_model_name() == "azure/gpt-4"

        with patch.dict(os.environ, {"AZURE_API_KEY": "azure-test-key"}):
            assert task._get_api_key() == "azure-test-key"

    def test_unknown_provider_fallback(self) -> None:
        """Unknown provider falls back to no prefix."""
        config = {"llm_provider": "unknown_provider", "llm_model": "unknown_model"}

        task = LLMTask("unknown_test", config)

        assert task._build_model_name() == "unknown_model"  # No prefix
        assert task._get_api_key() is None  # No known environment variable


class TestLLMTaskErrorHandling:
    """Test LLM task error handling and edge cases."""

    def test_template_resolution_with_complex_errors(self) -> None:
        """Template resolution handles complex error scenarios."""
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "prompt_template": "Process ${valid_var} and ${error_var}",
        }

        task = LLMTask("error_template_test", config)
        context = TaskContext()
        context.store_result("data", {"valid": "good data"})
        task.set_context(context)

        # Add one valid variable
        task.config["valid_var"] = "good data"

        # error_var will be missing

        with patch.object(task.logger, "warning") as mock_warning:
            template = task.config["prompt_template"]
            result = task._resolve_template_string(template)

            # Should log warning for missing variable
            mock_warning.assert_called_with(
                "Could not resolve template variable: error_var"
            )

            # Should resolve valid variable but leave invalid one unchanged
            assert "good data" in result
            assert "${error_var}" in result

    def test_api_call_with_multiple_failures(self) -> None:
        """API call handles various types of failures."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4"}

        task = LLMTask("multi_error_test", config)
        task.resolved_prompt = "Test prompt"

        # Test different types of API failures
        error_scenarios = [
            ("Connection error", ConnectionError("Connection failed")),
            ("Timeout error", TimeoutError("Request timed out")),
            ("API error", RuntimeError("API rate limit exceeded")),
            ("Authentication error", PermissionError("Invalid API key")),
        ]

        for _, error in error_scenarios:
            with patch(
                "aipype.framework.llm_task.litellm.completion", side_effect=error
            ):
                with pytest.raises(
                    RuntimeError,
                    match=f"LLMTask API call operation failed: LLM API call to.*failed: {str(error)}",
                ):
                    task._make_llm_call()

    def test_response_processing_edge_cases(self) -> None:
        """Response processing handles various edge cases."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4"}

        task = LLMTask("response_edge_test", config)

        # Test response with None content - should return empty string, not fail
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None
        mock_response.model = "gpt-4"

        result = task._process_response(mock_response)
        assert result["content"] == ""
        assert result["model"] == "gpt-4"
        assert result["provider"] == "openai"

        # Test response with empty choices - should still fail
        mock_response.choices = []

        with pytest.raises(RuntimeError, match="Failed to process LLM response"):
            task._process_response(mock_response)

    def test_logging_with_unicode_content(self) -> None:
        """Logging handles Unicode content correctly."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4"}

        task = LLMTask("unicode_test", config)

        # Test with Unicode content
        input_data = {"prompt": "Analyze données français 中文 🚀"}
        output_data = {"content": "Response with émojis 😊 and 中文字符"}

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, encoding="utf-8"
        ) as temp_file:
            temp_path = temp_file.name
            task.logs_file = temp_path

            task._log_llm_interaction(input_data, output_data)

            # Read and verify Unicode content
            with open(temp_path, "r", encoding="utf-8") as f:
                log_line = f.read().strip()
                log_entry = json.loads(log_line)

                assert (
                    log_entry["input"]["prompt"] == "Analyze données français 中文 🚀"
                )
                assert (
                    log_entry["output"]["content"]
                    == "Response with émojis 😊 and 中文字符"
                )

            # Cleanup
            os.unlink(temp_path)

    def test_large_template_resolution(self) -> None:
        """Template resolution handles large templates efficiently."""
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "prompt_template": "Process " + "${var} " * 1000 + "efficiently",
        }

        task = LLMTask("large_template_test", config)
        context = TaskContext()
        task.set_context(context)

        # Add the variable
        task.config["var"] = "data"

        result = task._resolve_template_string(task.config["prompt_template"])

        assert result.startswith("Process data ")
        assert result.endswith(" efficiently")
        assert result.count("data") == 1000

    def test_concurrent_logging_safety(self) -> None:
        """Logging is safe for concurrent access."""
        import time

        config = {"llm_provider": "openai", "llm_model": "gpt-4"}

        task = LLMTask("concurrent_test", config)

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_path = temp_file.name
            task.logs_file = temp_path

            def log_worker(worker_id: int) -> None:
                for i in range(10):
                    input_data = {"prompt": f"Worker {worker_id} request {i}"}
                    output_data = {"content": f"Worker {worker_id} response {i}"}
                    task._log_llm_interaction(input_data, output_data)
                    time.sleep(0.001)  # Small delay

            # Start multiple logging threads
            threads: List[threading.Thread] = []
            for worker_id in range(5):
                thread = threading.Thread(target=log_worker, args=(worker_id,))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Verify all log entries were written
            with open(temp_path, "r") as f:
                lines = f.readlines()
                assert len(lines) == 50  # 5 workers * 10 entries each

            # Cleanup
            os.unlink(temp_path)

    def test_template_with_circular_references(self) -> None:
        """Template resolution handles circular reference attempts gracefully."""
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "prompt_template": "Process ${circular_var}",
        }

        task = LLMTask("circular_test", config)
        context = TaskContext()

        # Create a context that could potentially cause circular references
        context.store_result("circular", {"ref": "${circular_var}"})
        task.set_context(context)

        # This should not cause infinite recursion
        with patch.object(task.logger, "warning") as mock_warning:
            result = task._resolve_template_string(task.config["prompt_template"])

            # Should warn about missing variable and leave template unchanged
            mock_warning.assert_called_with(
                "Could not resolve template variable: circular_var"
            )
            assert result == "Process ${circular_var}"

    def test_invalid_json_in_logging(self) -> None:
        """Logging handles non-JSON-serializable data gracefully."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4"}

        task = LLMTask("json_error_test", config)

        # Create data that can't be JSON serialized (datetime objects)
        input_data: Dict[str, Any] = {"prompt": "test", "timestamp": datetime.now()}
        output_data: Dict[str, Any] = {"content": "test", "function": lambda x: x}  # pyright: ignore  # Function can't be serialized

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_path = temp_file.name
            task.logs_file = temp_path

            # This should not crash but should log a warning
            with patch.object(task.logger, "warning") as mock_warning:
                task._log_llm_interaction(input_data, output_data)
                mock_warning.assert_called_once()
                assert "Failed to log LLM interaction" in mock_warning.call_args[0][0]

            # Cleanup
            os.unlink(temp_path)
