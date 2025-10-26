"""Tests for LLMTask structured response support."""
# pyright: reportPrivateUsage=false

from unittest.mock import Mock, patch
from pydantic import BaseModel

from aipype import LLMTask


class Article(BaseModel):
    """Pydantic model for structured responses (used in tests)."""

    title: str
    summary: str
    key_points: list[str]


class Person(BaseModel):
    """Pydantic model for person data (used in tests)."""

    name: str
    age: int
    email: str


class TestLLMTaskStructuredResponseInitialization:
    """Test LLMTask initialization with response_format."""

    def test_initialization_with_pydantic_model(self) -> None:
        """LLMTask initializes with Pydantic model response_format."""
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "prompt": "Extract article data",
            "response_format": Article,
        }

        task = LLMTask("test_structured", config)

        assert task.supports_response_format is True
        assert task.response_format_type == "pydantic"
        assert task.response_format_config is not None
        assert "json_schema" in task.response_format_config

    def test_initialization_with_json_schema(self) -> None:
        """LLMTask initializes with JSON schema response_format."""
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "prompt": "Extract data",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "Article",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "summary": {"type": "string"},
                        },
                        "required": ["title", "summary"],
                    },
                    "strict": True,
                },
            },
        }

        task = LLMTask("test_json_schema", config)

        assert task.supports_response_format is True
        assert task.response_format_type == "json_schema"
        assert task.response_format_config == config["response_format"]

    def test_initialization_without_response_format(self) -> None:
        """LLMTask initializes without response_format."""
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "prompt": "Generate text",
        }

        task = LLMTask("test_no_format", config)

        assert task.supports_response_format is False
        assert task.response_format_type is None
        assert task.response_format_config is None


class TestLLMTaskStructuredResponseValidation:
    """Test response_format validation."""

    def test_validate_pydantic_model(self) -> None:
        """Validates Pydantic model as response_format."""
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "response_format": Article,
        }

        task = LLMTask("test", config)
        assert task._validate_response_format(Article) is True

    def test_validate_json_schema_dict(self) -> None:
        """Validates JSON schema dict as response_format."""
        valid_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "Test",
                "schema": {"type": "object"},
            },
        }

        config = {"llm_provider": "openai", "llm_model": "gpt-4o-mini"}
        task = LLMTask("test", config)
        assert task._validate_response_format(valid_schema) is True

    def test_validate_invalid_response_format(self) -> None:
        """Rejects invalid response_format."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4o-mini"}
        task = LLMTask("test", config)

        # String is invalid
        assert task._validate_response_format("invalid") is False

        # Dict without required fields is invalid
        assert task._validate_response_format({"type": "json_schema"}) is False

        # Dict with incomplete json_schema is invalid
        assert (
            task._validate_response_format(
                {"type": "json_schema", "json_schema": {"name": "Test"}}
            )
            is False
        )


class TestLLMTaskStructuredResponseConversion:
    """Test Pydantic to JSON schema conversion."""

    def test_convert_pydantic_to_schema(self) -> None:
        """Converts Pydantic model to litellm JSON schema format."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4o-mini"}
        task = LLMTask("test", config)

        schema = task._convert_pydantic_to_schema(Article)

        assert schema["type"] == "json_schema"
        assert "json_schema" in schema
        assert schema["json_schema"]["name"] == "Article"
        assert "schema" in schema["json_schema"]
        assert schema["json_schema"]["strict"] is True

    def test_convert_pydantic_schema_structure(self) -> None:
        """Converted schema has correct structure."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4o-mini"}
        task = LLMTask("test", config)

        schema = task._convert_pydantic_to_schema(Person)
        json_schema = schema["json_schema"]["schema"]

        # Check that properties are included
        assert "properties" in json_schema
        assert "name" in json_schema["properties"]
        assert "age" in json_schema["properties"]
        assert "email" in json_schema["properties"]

    def test_schema_has_additional_properties_false(self) -> None:
        """Converted schema includes additionalProperties: false for OpenAI strict mode."""
        config = {"llm_provider": "openai", "llm_model": "gpt-4o-mini"}
        task = LLMTask("test", config)

        schema = task._convert_pydantic_to_schema(Article)
        json_schema = schema["json_schema"]["schema"]

        # Root object should have additionalProperties: false
        assert json_schema.get("additionalProperties") is False

        # Nested objects should also have additionalProperties: false if present
        # (This is required by OpenAI's strict mode)


class TestLLMTaskStructuredResponseProcessing:
    """Test processing structured responses."""

    def test_process_response_with_structured_output(self) -> None:
        """Processes response with structured JSON output."""
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "response_format": Article,
        }

        task = LLMTask("test", config)
        task.supports_response_format = True

        # Mock response with JSON content
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content='{"title": "Test Article", "summary": "A test summary", "key_points": ["point1", "point2"]}',
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
        ]
        mock_response.usage = Mock(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )
        mock_response.model = "gpt-4o-mini"

        result = task._process_response(mock_response)

        assert "parsed_object" in result
        assert result["parsed_object"]["title"] == "Test Article"
        assert result["parsed_object"]["summary"] == "A test summary"
        assert len(result["parsed_object"]["key_points"]) == 2

    def test_process_response_with_invalid_json(self) -> None:
        """Handles invalid JSON in structured response gracefully."""
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "Test", "schema": {}},
            },
        }

        task = LLMTask("test", config)
        task.supports_response_format = True

        # Mock response with invalid JSON
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(content="This is not JSON", tool_calls=None),
                finish_reason="stop",
            )
        ]
        mock_response.usage = Mock(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )
        mock_response.model = "gpt-4o-mini"

        # Should not raise exception, just skip parsed_object
        result = task._process_response(mock_response)

        assert "parsed_object" not in result
        assert result["content"] == "This is not JSON"


class TestLLMTaskStructuredResponseIntegration:
    """Test full integration with mocked LLM calls."""

    def test_run_with_pydantic_response_format(self) -> None:
        """Runs task with Pydantic response_format and returns parsed object."""
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "prompt": "Extract article",
            "response_format": Article,
        }

        task = LLMTask("test", config)

        # Mock litellm.completion to return structured response
        with patch("aipype.llm_task.litellm") as mock_litellm:
            mock_response = Mock()
            mock_response.choices = [
                Mock(
                    message=Mock(
                        content='{"title": "AI News", "summary": "Latest updates", "key_points": ["GPT-5", "Claude-4"]}',
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ]
            mock_response.usage = Mock(
                prompt_tokens=50, completion_tokens=100, total_tokens=150
            )
            mock_response.model = "gpt-4o-mini"

            mock_litellm.completion.return_value = mock_response
            mock_litellm.utils.supports_response_schema.return_value = True

            result = task.run()

            assert result.is_success() is True
            assert "parsed_object" in result.data
            assert result.data["parsed_object"]["title"] == "AI News"
            assert len(result.data["parsed_object"]["key_points"]) == 2

            # Verify response_format was passed to litellm.completion
            call_args = mock_litellm.completion.call_args
            assert "response_format" in call_args[1]

    def test_run_warns_on_unsupported_model(self) -> None:
        """Warns when model doesn't support response_format."""
        config = {
            "llm_provider": "openai",
            "llm_model": "gpt-3.5-turbo",  # Assume this doesn't support structured output
            "prompt": "Extract data",
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "Test", "schema": {}},
            },
        }

        task = LLMTask("test", config)

        with patch("aipype.llm_task.litellm") as mock_litellm:
            # Setup mock to return False for supports_response_schema
            mock_utils = Mock()
            mock_utils.supports_response_schema.return_value = False
            mock_litellm.utils = mock_utils

            mock_response = Mock()
            mock_response.choices = [
                Mock(
                    message=Mock(content="text response", tool_calls=None),
                    finish_reason="stop",
                )
            ]
            mock_response.usage = Mock(
                prompt_tokens=10, completion_tokens=20, total_tokens=30
            )
            mock_response.model = "gpt-3.5-turbo"

            mock_litellm.completion.return_value = mock_response

            # Should complete but log warning
            with patch.object(task.logger, "warning") as mock_warning:
                result = task.run()

                assert result.is_success() is True
                # Check if warning was called with message about unsupported response_format
                warning_calls = [
                    str(call[0][0]) for call in mock_warning.call_args_list
                ]
                assert any(
                    "does not support response_format" in msg for msg in warning_calls
                )
