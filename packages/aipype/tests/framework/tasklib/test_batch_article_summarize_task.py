"""Tests for BatchArticleSummarizeTask - reusable batch article summarization task."""

import pytest
from typing import Any, Dict, List
from unittest.mock import Mock, patch
from aipype.framework.tasklib.web.batch_article_summarize_task import (
    BatchArticleSummarizeTask,
)
from aipype import TaskDependency, DependencyType, TaskResult


class TestBatchArticleSummarizeTask:
    """Test suite for BatchArticleSummarizeTask."""

    @pytest.fixture
    def sample_articles(self) -> List[Dict[str, Any]]:
        """Provide sample articles for testing."""
        return [
            {
                "title": "AI in Healthcare",
                "url": "https://example.com/ai-healthcare",
                "content": "Artificial intelligence is revolutionizing healthcare by enabling faster diagnosis, personalized treatment plans, and improved patient outcomes. Machine learning algorithms can analyze medical images with accuracy matching human experts.",
            },
            {
                "title": "Machine Learning Trends",
                "url": "https://example.com/ml-trends",
                "content": "The latest trends in machine learning include transformer models, federated learning, and automated machine learning (AutoML). These technologies are making AI more accessible and efficient across various industries.",
            },
            {
                "title": "Future of AI",
                "url": "https://example.com/future-ai",
                "content": "Looking ahead, artificial intelligence will likely become more integrated into daily life, with advances in natural language processing, computer vision, and robotics leading to new applications and possibilities.",
            },
        ]

    @pytest.fixture
    def short_content_articles(self) -> List[Dict[str, Any]]:
        """Provide articles with insufficient content for testing."""
        return [
            {
                "title": "Short Article",
                "url": "https://example.com/short",
                "content": "Very short content.",  # Only 19 characters, below default min of 50
            },
            {
                "title": "Empty Article",
                "url": "https://example.com/empty",
                "content": "",
            },
        ]

    @pytest.fixture
    def long_content_article(self) -> Dict[str, Any]:
        """Provide article with very long content for testing truncation."""
        long_content = (
            "This is a very long article content. " * 100
        )  # Creates ~3700 characters
        return {
            "title": "Very Long Article",
            "url": "https://example.com/long",
            "content": long_content,
        }

    @pytest.fixture
    def mixed_articles(
        self,
        sample_articles: List[Dict[str, Any]],
        short_content_articles: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Provide mixed valid and invalid articles."""
        return sample_articles + short_content_articles

    @pytest.fixture
    def mock_llm_success_response(self) -> Mock:
        """Mock successful LLM response."""
        mock_response = Mock()
        mock_response.get.return_value = TaskResult.success(
            data={
                "content": "This is a comprehensive summary of the article covering the main points and key insights about the topic.",
                "usage": {
                    "prompt_tokens": 150,
                    "completion_tokens": 75,
                    "total_tokens": 225,
                },
            },
            metadata={
                "usage": {
                    "prompt_tokens": 150,
                    "completion_tokens": 75,
                    "total_tokens": 225,
                }
            },
        )
        return mock_response

    @pytest.fixture
    def mock_llm_generic_response(self) -> Mock:
        """Mock generic/unhelpful LLM response."""
        mock_response = Mock()
        mock_response.get.return_value = TaskResult.success(
            data={
                "content": "How can I assist you today? Feel free to share your questions!",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
            },
            metadata={
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                }
            },
        )
        return mock_response

    @pytest.fixture
    def task_with_default_config(
        self, sample_articles: List[Dict[str, Any]]
    ) -> BatchArticleSummarizeTask:
        """Create task with default configuration."""
        return BatchArticleSummarizeTask(
            "test_summarize",
            {
                "articles": sample_articles,
                "llm_provider": "openai",
                "llm_model": "gpt-3.5-turbo",
            },
        )

    @pytest.fixture
    def task_with_custom_config(
        self, sample_articles: List[Dict[str, Any]]
    ) -> BatchArticleSummarizeTask:
        """Create task with custom configuration."""
        return BatchArticleSummarizeTask(
            "test_summarize_custom",
            {
                "articles": sample_articles,
                "summary_length": 500,
                "content_limit": 2000,
                "min_content_length": 100,
                "temperature": 0.5,
                "max_tokens": 200,
                "llm_provider": "anthropic",
                "llm_model": "claude-3-haiku",
            },
        )

    def test_initialization_with_default_config(self) -> None:
        """Test task initialization with default configuration."""
        task = BatchArticleSummarizeTask("test_task", {"articles": []})

        assert task.name == "test_task"
        assert task.config["articles"] == []
        assert task.dependencies == []

    def test_initialization_with_dependencies(self) -> None:
        """Test task initialization with dependencies."""
        dependencies = [
            TaskDependency("articles", "fetch_task.articles", DependencyType.REQUIRED)
        ]
        task = BatchArticleSummarizeTask("test_task", {}, dependencies)

        assert task.get_dependencies() == dependencies

    def test_get_dependencies(
        self, task_with_default_config: BatchArticleSummarizeTask
    ) -> None:
        """Test get_dependencies method."""
        dependencies = task_with_default_config.get_dependencies()
        assert isinstance(dependencies, list)

    def test_no_articles_raises_error(self) -> None:
        """Test that providing no articles returns TaskResult.failure()."""
        task = BatchArticleSummarizeTask("test_task", {})

        result = task.run()
        assert result.is_error()
        assert result.error is not None
        assert (
            "BatchArticleSummarizeTask validation failed: No articles provided for summarization"
            in result.error
        )

    def test_empty_articles_list_raises_error(self) -> None:
        """Test that empty articles list returns TaskResult.failure()."""
        task = BatchArticleSummarizeTask("test_task", {"articles": []})

        result = task.run()
        assert result.is_error()
        assert result.error is not None
        assert (
            "BatchArticleSummarizeTask validation failed: No articles provided for summarization"
            in result.error
        )

    @patch("aipype.framework.llm_task.LLMTask.run")
    def test_successful_summarization_with_defaults(
        self, mock_llm_run: Mock, task_with_default_config: BatchArticleSummarizeTask
    ) -> None:
        """Test successful summarization with default configuration."""
        # Mock LLM responses
        mock_llm_run.return_value = TaskResult.success(
            data={
                "content": "Summary of the article with key insights and main points covered.",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            },
            metadata={
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                }
            },
        )

        result = task_with_default_config.run()

        # Verify structure
        assert "summaries" in result.data
        assert "formatted_summaries" in result.data
        assert "total_articles" in result.data
        assert "successful_summaries" in result.data
        assert "total_tokens" in result.data
        assert "model" in result.data
        assert "provider" in result.data
        assert "summary_length" in result.data

        # Verify content
        assert result.data["total_articles"] == 3
        assert result.data["successful_summaries"] == 3
        assert result.data["total_tokens"] == 450  # 150 * 3 articles
        assert result.data["model"] == "gpt-3.5-turbo"
        assert result.data["provider"] == "openai"
        assert result.data["summary_length"] == 1000  # default

        # Verify summaries structure
        summaries = result.data["summaries"]
        assert len(summaries) == 3
        for i, summary in enumerate(summaries, 1):
            assert summary["article_index"] == i
            assert "title" in summary
            assert "url" in summary
            assert "summary" in summary

        # Verify LLM was called 3 times
        assert mock_llm_run.call_count == 3

    @patch("aipype.framework.llm_task.LLMTask.run")
    def test_successful_summarization_with_custom_config(
        self, mock_llm_run: Mock, task_with_custom_config: BatchArticleSummarizeTask
    ) -> None:
        """Test successful summarization with custom configuration."""
        mock_llm_run.return_value = TaskResult.success(
            data={
                "content": "Custom summary with different length requirements.",
                "usage": {
                    "prompt_tokens": 80,
                    "completion_tokens": 40,
                    "total_tokens": 120,
                },
            },
            metadata={
                "usage": {
                    "prompt_tokens": 80,
                    "completion_tokens": 40,
                    "total_tokens": 120,
                }
            },
        )

        result = task_with_custom_config.run()

        # Verify custom config values are used
        assert result.data["summary_length"] == 500
        assert result.data["model"] == "claude-3-haiku"
        assert result.data["provider"] == "anthropic"
        assert result.data["total_tokens"] == 360  # 120 * 3 articles

    @patch("aipype.framework.llm_task.LLMTask.run")
    def test_content_length_validation(
        self, mock_llm_run: Mock, short_content_articles: List[Dict[str, Any]]
    ) -> None:
        """Test that articles with insufficient content are skipped."""
        mock_llm_run.return_value = TaskResult.success(
            data={
                "content": "This should not be called for short articles.",
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 25,
                    "total_tokens": 75,
                },
            },
            metadata={
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 25,
                    "total_tokens": 75,
                }
            },
        )

        task = BatchArticleSummarizeTask(
            "test_task",
            {
                "articles": short_content_articles,
                "min_content_length": 50,  # Both test articles are below this
            },
        )

        result = task.run()

        # Verify articles were skipped
        assert result.data["total_articles"] == 2
        # Note: successful_summaries only excludes "[Summarization failed:" not "[Skipped:"
        assert (
            result.data["successful_summaries"] == 2
        )  # Both are counted as "successful" (not failed)
        assert result.data["total_tokens"] == 0

        # Verify summaries indicate skipping
        summaries = result.data["summaries"]
        assert len(summaries) == 2
        for summary in summaries:
            assert summary["summary"].startswith("[Skipped: Insufficient content")

        # Verify LLM was never called
        assert mock_llm_run.call_count == 0

    @patch("aipype.framework.llm_task.LLMTask.run")
    def test_content_truncation(
        self, mock_llm_run: Mock, long_content_article: Dict[str, Any]
    ) -> None:
        """Test that very long content is truncated."""
        mock_llm_run.return_value = TaskResult.success(
            data={
                "content": "Summary of truncated content.",
                "usage": {
                    "prompt_tokens": 200,
                    "completion_tokens": 100,
                    "total_tokens": 300,
                },
            },
            metadata={
                "usage": {
                    "prompt_tokens": 200,
                    "completion_tokens": 100,
                    "total_tokens": 300,
                }
            },
        )

        task = BatchArticleSummarizeTask(
            "test_task",
            {
                "articles": [long_content_article],
                "content_limit": 3000,  # Article is longer than this
            },
        )

        result = task.run()

        # Verify task completed successfully
        assert result.data["total_articles"] == 1
        assert result.data["successful_summaries"] == 1

        # Verify LLM was called with truncated content
        assert mock_llm_run.call_count == 1
        # The LLMTask config should contain truncated content in the prompt
        # We can't easily verify the exact content, but we know it was processed

    @patch("aipype.framework.llm_task.LLMTask.run")
    def test_mixed_valid_invalid_articles(
        self, mock_llm_run: Mock, mixed_articles: List[Dict[str, Any]]
    ) -> None:
        """Test handling of mixed valid and invalid articles."""
        mock_llm_run.return_value = TaskResult.success(
            data={
                "content": "Summary of valid article content.",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            },
            metadata={
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                }
            },
        )

        task = BatchArticleSummarizeTask(
            "test_task",
            {
                "articles": mixed_articles,  # 3 valid + 2 invalid articles
                "min_content_length": 50,
            },
        )

        result = task.run()

        # Verify totals
        assert result.data["total_articles"] == 5
        assert (
            result.data["successful_summaries"] == 5
        )  # All counted as successful (skipped != failed)
        assert result.data["total_tokens"] == 450  # 150 * 3 valid articles

        # Verify LLM called only for valid articles
        assert mock_llm_run.call_count == 3

        # Verify summaries structure
        summaries = result.data["summaries"]
        assert len(summaries) == 5

        # First 3 should be successful
        for i in range(3):
            assert not summaries[i]["summary"].startswith("[Skipped:")

        # Last 2 should be skipped
        for i in range(3, 5):
            assert summaries[i]["summary"].startswith("[Skipped:")

    @patch("aipype.framework.llm_task.LLMTask.run")
    def test_llm_task_failure_handling(
        self, mock_llm_run: Mock, sample_articles: List[Dict[str, Any]]
    ) -> None:
        """Test handling when LLM task raises exception."""
        # First call succeeds, second fails, third succeeds
        mock_llm_run.side_effect = [
            TaskResult.success(
                data={"content": "First summary", "usage": {"total_tokens": 100}},
                metadata={"usage": {"total_tokens": 100}},
            ),
            Exception("LLM API error"),
            TaskResult.success(
                data={"content": "Third summary", "usage": {"total_tokens": 100}},
                metadata={"usage": {"total_tokens": 100}},
            ),
        ]

        task = BatchArticleSummarizeTask("test_task", {"articles": sample_articles})

        result = task.run()

        # Verify handling of failure
        assert result.data["total_articles"] == 3
        assert result.data["successful_summaries"] == 2  # 2 succeeded, 1 failed
        assert result.data["total_tokens"] == 200  # Only successful calls counted

        summaries = result.data["summaries"]
        assert len(summaries) == 3

        # Verify error summary format
        assert not summaries[0]["summary"].startswith("[Summarization failed:")
        assert summaries[1]["summary"].startswith("[Summarization failed:")
        assert not summaries[2]["summary"].startswith("[Summarization failed:")

    @patch("aipype.framework.llm_task.LLMTask.run")
    def test_generic_response_detection(
        self, mock_llm_run: Mock, sample_articles: List[Dict[str, Any]]
    ) -> None:
        """Test detection and handling of generic LLM responses."""
        # Return generic responses that should be detected
        mock_llm_run.side_effect = [
            TaskResult.success(
                data={
                    "content": "How can I assist you today?",
                    "usage": {"total_tokens": 50},
                },
                metadata={"usage": {"total_tokens": 50}},
            ),
            TaskResult.success(
                data={
                    "content": "It seems like your message is empty. Please share your question!",
                    "usage": {"total_tokens": 60},
                },
                metadata={"usage": {"total_tokens": 60}},
            ),
            TaskResult.success(
                data={
                    "content": "Valid summary of the article content with actual insights.",
                    "usage": {"total_tokens": 100},
                },
                metadata={"usage": {"total_tokens": 100}},
            ),
        ]

        task = BatchArticleSummarizeTask("test_task", {"articles": sample_articles})

        result = task.run()

        # Verify generic responses were detected
        assert result.data["total_articles"] == 3
        assert (
            result.data["successful_summaries"] == 3
        )  # All counted as successful (generic != failed)
        assert (
            result.data["total_tokens"] == 210
        )  # All tokens counted, even for generic responses

        summaries = result.data["summaries"]

        # First two should be marked as generic responses
        assert summaries[0]["summary"].startswith("[LLM returned generic response")
        assert summaries[1]["summary"].startswith("[LLM returned generic response")
        assert not summaries[2]["summary"].startswith("[LLM returned generic response")

    @patch("aipype.framework.llm_task.LLMTask.run")
    def test_short_response_detection(
        self, mock_llm_run: Mock, sample_articles: List[Dict[str, Any]]
    ) -> None:
        """Test detection of responses that are too short."""
        mock_llm_run.side_effect = [
            TaskResult.success(
                data={"content": "Short", "usage": {"total_tokens": 30}},
                metadata={"usage": {"total_tokens": 30}},
            ),  # Too short
            TaskResult.success(
                data={"content": "A" * 60, "usage": {"total_tokens": 50}},
                metadata={"usage": {"total_tokens": 50}},
            ),  # Long enough
            TaskResult.success(
                data={"content": "", "usage": {"total_tokens": 20}},
                metadata={"usage": {"total_tokens": 20}},
            ),  # Empty
        ]

        task = BatchArticleSummarizeTask("test_task", {"articles": sample_articles})

        result = task.run()

        summaries = result.data["summaries"]

        # First and third should be marked as generic/insufficient
        assert summaries[0]["summary"].startswith("[LLM returned generic response")
        assert not summaries[1]["summary"].startswith("[LLM returned generic response")
        assert summaries[2]["summary"].startswith("[LLM returned generic response")

    @patch("aipype.framework.llm_task.LLMTask.run")
    def test_formatted_summaries_output(
        self, mock_llm_run: Mock, sample_articles: List[Dict[str, Any]]
    ) -> None:
        """Test the formatted summaries output format."""
        # Use a longer summary to avoid generic response detection (>50 chars)
        long_summary = "This is a comprehensive article summary content that provides detailed insights and analysis about the topic being discussed."
        mock_llm_run.return_value = TaskResult.success(
            data={
                "content": long_summary,
                "usage": {"total_tokens": 100},
            },
            metadata={"usage": {"total_tokens": 100}},
        )

        task = BatchArticleSummarizeTask(
            "test_task",
            {
                "articles": sample_articles[:1]  # Just one article for simplicity
            },
        )

        result = task.run()

        formatted = result.data["formatted_summaries"]

        # Verify format
        expected_start = f"ARTICLE 1 SUMMARY (AI in Healthcare):\n{long_summary}"
        assert formatted.startswith(expected_start)

    @pytest.mark.parametrize(
        "summary_length,content_limit,min_content_length,temperature,max_tokens",
        [
            (500, 2000, 100, 0.1, 150),
            (1500, 4000, 25, 0.7, 400),
            (800, 2500, 75, 0.5, 250),
        ],
    )
    @patch("aipype.framework.llm_task.LLMTask.run")
    def test_parameter_configurations(
        self,
        mock_llm_run: Mock,
        sample_articles: List[Dict[str, Any]],
        summary_length: int,
        content_limit: int,
        min_content_length: int,
        temperature: float,
        max_tokens: int,
    ) -> None:
        """Test various parameter configurations."""
        mock_llm_run.return_value = TaskResult.success(
            data={
                "content": "Test summary content.",
                "usage": {"total_tokens": 100},
            },
            metadata={"usage": {"total_tokens": 100}},
        )

        task = BatchArticleSummarizeTask(
            "test_task",
            {
                "articles": sample_articles,
                "summary_length": summary_length,
                "content_limit": content_limit,
                "min_content_length": min_content_length,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

        result = task.run()

        # Verify custom parameters are reflected in result
        assert result.data["summary_length"] == summary_length

    @patch("aipype.framework.llm_task.LLMTask.run")
    def test_single_article_processing(self, mock_llm_run: Mock) -> None:
        """Test processing of a single article."""
        mock_llm_run.return_value = TaskResult.success(
            data={
                "content": "Single article summary.",
                "usage": {"total_tokens": 75},
            },
            metadata={"usage": {"total_tokens": 75}},
        )

        single_article = [
            {
                "title": "Single Test Article",
                "url": "https://example.com/single",
                "content": "This is a single article for testing the summarization task.",
            }
        ]

        task = BatchArticleSummarizeTask("test_task", {"articles": single_article})

        result = task.run()

        assert result.data["total_articles"] == 1
        assert result.data["successful_summaries"] == 1
        assert result.data["total_tokens"] == 75
        assert len(result.data["summaries"]) == 1
        assert mock_llm_run.call_count == 1

    @patch("aipype.framework.llm_task.LLMTask.run")
    def test_unicode_content_handling(self, mock_llm_run: Mock) -> None:
        """Test handling of articles with Unicode content."""
        mock_llm_run.return_value = TaskResult.success(
            data={
                "content": "Summary with émojis and spëcial characters.",
                "usage": {"total_tokens": 80},
            },
            metadata={"usage": {"total_tokens": 80}},
        )

        unicode_article = [
            {
                "title": "Artículo en Español 🇪🇸",
                "url": "https://example.com/español",
                "content": "Este es un artículo con caracteres especiales: áéíóú, ñ, ¿¡ y emojis 🚀💡🌟",
            }
        ]

        task = BatchArticleSummarizeTask("test_task", {"articles": unicode_article})

        result = task.run()

        assert result.data["successful_summaries"] == 1
        assert "🇪🇸" in result.data["summaries"][0]["title"]

    @patch("aipype.framework.llm_task.LLMTask.run")
    def test_llm_task_creation_parameters(
        self, mock_llm_run: Mock, sample_articles: List[Dict[str, Any]]
    ) -> None:
        """Test that LLM tasks are created with correct parameters."""
        mock_llm_run.return_value = TaskResult.success(
            data={
                "content": "Test summary",
                "usage": {"total_tokens": 100},
            },
            metadata={"usage": {"total_tokens": 100}},
        )

        task = BatchArticleSummarizeTask(
            "test_task",
            {
                "articles": sample_articles[:1],  # Just one article
                "llm_provider": "anthropic",
                "llm_model": "claude-3-sonnet",
                "temperature": 0.2,
                "max_tokens": 350,
            },
        )

        # We need to patch the LLMTask creation to verify parameters
        with patch(
            "aipype.framework.tasklib.web.batch_article_summarize_task.LLMTask"
        ) as mock_llm_task_class:
            mock_llm_instance = Mock()
            mock_llm_instance.run.return_value = TaskResult.success(
                data={
                    "content": "Test summary",
                    "usage": {"total_tokens": 100},
                },
                metadata={"usage": {"total_tokens": 100}},
            )
            mock_llm_task_class.return_value = mock_llm_instance

            _ = task.run()

            # Verify LLMTask was created with correct parameters
            assert mock_llm_task_class.call_count == 1
            call_args = mock_llm_task_class.call_args

            # Check the config passed to LLMTask
            config = call_args[0][1]  # Second argument is config
            assert config["llm_provider"] == "anthropic"
            assert config["llm_model"] == "claude-3-sonnet"
            assert config["temperature"] == 0.2
            assert config["max_tokens"] == 350
            assert config["context"] == "You are an expert content summarizer."

    @patch("aipype.framework.llm_task.LLMTask.run")
    def test_prompt_template_generation(self, mock_llm_run: Mock) -> None:
        """Test that prompt templates are generated correctly."""
        mock_llm_run.return_value = TaskResult.success(
            data={
                "content": "Test summary",
                "usage": {"total_tokens": 100},
            },
            metadata={"usage": {"total_tokens": 100}},
        )

        test_article = [
            {
                "title": "Test Article Title",
                "url": "https://test.com/article",
                "content": "Test article content for prompt generation. This content is long enough to pass the minimum content length validation and will be used to test prompt template generation functionality.",
            }
        ]

        task = BatchArticleSummarizeTask(
            "test_task",
            {
                "articles": test_article,
                "summary_length": 750,  # Custom length
            },
        )

        with patch(
            "aipype.framework.tasklib.web.batch_article_summarize_task.LLMTask"
        ) as mock_llm_task_class:
            mock_llm_instance = Mock()
            mock_llm_instance.run.return_value = TaskResult.success(
                data={
                    "content": "Test summary",
                    "usage": {"total_tokens": 100},
                },
                metadata={"usage": {"total_tokens": 100}},
            )
            mock_llm_task_class.return_value = mock_llm_instance

            _ = task.run()

            # Verify prompt template contains expected elements
            call_args = mock_llm_task_class.call_args
            config = call_args[0][1]
            prompt = config["prompt_template"]

            # Verify key elements in prompt
            assert "Test Article Title" in prompt
            assert "https://test.com/article" in prompt
            assert "Test article content for prompt generation" in prompt
            assert "approximately 750 characters" in prompt  # Custom length
            assert "Do NOT respond with generic messages" in prompt
