"""Tests for enhanced task types with context awareness and dependencies."""

from typing import Any, Dict, List
from unittest.mock import Mock, patch
from aipype import (
    LLMTask,
    SearchTask,
    TransformTask,
    ConditionalTask,
    TaskContext,
    TaskDependency,
    DependencyType,
)


class TestLLMTask:
    """Test suite for LLMTask - LLM task with context awareness."""

    def test_contextual_llm_task_uses_context_data_in_prompts(self) -> None:
        """LLMTask can use context data in prompts."""
        # Setup context with previous task results
        context = TaskContext()
        context.store_result(
            "search",
            {
                "query": "artificial intelligence",
                "results": [{"title": "AI Basics", "url": "http://ai.com"}],
            },
        )
        context.store_result(
            "fetch",
            {
                "articles": [
                    {
                        "title": "AI Basics",
                        "content": "AI is transforming industries...",
                    }
                ]
            },
        )

        # Create LLM task with context-aware prompt template
        prompt_template = """Based on the search query "${search.query}" and the following articles:

${fetch.articles}

Please write a comprehensive summary."""

        dependencies = [
            TaskDependency("search_query", "search.query", DependencyType.REQUIRED),
            TaskDependency(
                "article_content", "fetch.articles", DependencyType.REQUIRED
            ),
        ]

        task = LLMTask(
            "summarize",
            {
                "prompt_template": prompt_template,
                "llm_provider": "openai",
                "llm_model": "gpt-3.5-turbo",
                "temperature": 0.7,
            },
            dependencies,
        )

        # Mock the LLM API call
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated summary content"
        mock_response.model = "gpt-3.5-turbo"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150

        with patch("litellm.completion", return_value=mock_response):
            # Simulate dependency resolution that PipelineAgent would do
            from aipype import DependencyResolver

            resolver = DependencyResolver(context)
            resolved_config = resolver.resolve_dependencies(task)
            task.config.update(resolved_config)

            # Set context and run task
            task.set_context(context)
            result = task.run()

        # Verify the task succeeded and returned proper content
        assert result.is_success()
        assert result.data["content"] == "Generated summary content"

        # Verify the resolved prompt contained context data
        resolved_prompt = task.get_resolved_prompt()
        assert resolved_prompt is not None
        assert "artificial intelligence" in resolved_prompt
        assert "AI is transforming industries" in resolved_prompt

    def test_contextual_llm_task_with_complex_transformations(self) -> None:
        """LLMTask handles complex data transformations in prompts."""
        context = TaskContext()
        context.store_result(
            "fetch",
            {
                "articles": [
                    {"title": "Article 1", "content": "Content 1", "word_count": 500},
                    {"title": "Article 2", "content": "Content 2", "word_count": 750},
                ],
                "total_articles": 2,
            },
        )

        # Create transformation function for articles
        def format_articles_for_prompt(articles: List[Dict[str, Any]]) -> str:
            formatted: List[str] = []
            for i, article in enumerate(articles, 1):
                formatted.append(
                    f"Article {i}: {article['title']}\n{article['content']}\nWord Count: {article['word_count']}"
                )
            return "\n\n".join(formatted)

        dependencies = [
            TaskDependency(
                "formatted_articles",
                "fetch.articles",
                DependencyType.REQUIRED,
                transform_func=format_articles_for_prompt,
            ),
            TaskDependency(
                "total_count", "fetch.total_articles", DependencyType.REQUIRED
            ),
        ]

        task = LLMTask(
            "analyze",
            {
                "prompt_template": "Analyze these ${total_count} articles:\n\n${formatted_articles}\n\nProvide insights.",
                "llm_provider": "anthropic",
                "llm_model": "claude-3-sonnet",
            },
            dependencies,
        )

        # Mock LLM response
        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value.choices = [Mock()]
            mock_completion.return_value.choices[0].message.content = "Analysis result"
            mock_completion.return_value.model = "claude-3-sonnet"
            mock_completion.return_value.usage = None

            # Simulate dependency resolution that PipelineAgent would do
            from aipype import DependencyResolver

            resolver = DependencyResolver(context)
            resolved_config = resolver.resolve_dependencies(task)
            task.config.update(resolved_config)

            task.set_context(context)
            task.run()

        # Verify transformation was applied
        resolved_prompt = task.get_resolved_prompt()
        assert resolved_prompt is not None
        assert "Analyze these 2 articles:" in resolved_prompt
        assert "Article 1: Article 1\nContent 1\nWord Count: 500" in resolved_prompt
        assert "Article 2: Article 2\nContent 2\nWord Count: 750" in resolved_prompt

    def test_contextual_llm_task_handles_missing_dependencies(self) -> None:
        """LLMTask handles missing optional dependencies gracefully."""
        context = TaskContext()
        context.store_result("search", {"query": "test query"})
        # Note: not providing optional context data

        dependencies = [
            TaskDependency("query", "search.query", DependencyType.REQUIRED),
            TaskDependency(
                "extra_context",
                "optional.data",
                DependencyType.OPTIONAL,
                default_value="No additional context",
            ),
        ]

        task = LLMTask(
            "generate",
            {
                "prompt_template": "Query: ${query}\nContext: ${extra_context}\nGenerate response.",
                "llm_provider": "openai",
                "llm_model": "gpt-4",
            },
            dependencies,
        )

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value.choices = [Mock()]
            mock_completion.return_value.choices[
                0
            ].message.content = "Generated response"
            mock_completion.return_value.model = "gpt-4"

            # Simulate dependency resolution that PipelineAgent would do
            from aipype import DependencyResolver

            resolver = DependencyResolver(context)
            resolved_config = resolver.resolve_dependencies(task)
            task.config.update(resolved_config)

            task.set_context(context)
            task.run()

        # Verify default value was used
        resolved_prompt = task.get_resolved_prompt()
        assert resolved_prompt is not None
        assert "Query: test query" in resolved_prompt
        assert "Context: No additional context" in resolved_prompt


class TestSearchTask:
    """Test suite for SearchTask - Simple web search task."""

    def test_search_task_with_static_query(self) -> None:
        """SearchTask executes search with static query."""
        task = SearchTask(
            "simple_search",
            {"query": "machine learning", "max_results": 5},
        )

        # Mock search API
        mock_searcher = Mock()
        mock_searcher.search.return_value = Mock(
            query="machine learning",
            results=[
                Mock(
                    title="ML Article",
                    url="http://example.com",
                    snippet="ML content",
                    position=1,
                )
            ],
            total_results=1,
            search_time=0.5,
        )

        with patch(
            "aipype.framework.search_task.SerperSearcher",
            return_value=mock_searcher,
        ):
            result = task.run()

        # Verify search was executed
        assert result.is_success()
        assert result.data["query"] == "machine learning"
        assert result.data["total_results"] == 1
        assert result.data["search_time"] == 0.5
        assert len(result.data["results"]) == 1
        assert result.data["results"][0]["title"] == "ML Article"
        mock_searcher.search.assert_called_once_with("machine learning", max_results=5)

    def test_search_task_with_dependency_resolved_query(self) -> None:
        """SearchTask can use query resolved from dependencies."""
        context = TaskContext()
        context.store_result("user_input", {"topic": "artificial intelligence"})

        # Create search task with dependency that provides the query
        dependencies = [
            TaskDependency("query", "user_input.topic", DependencyType.REQUIRED),
        ]

        task = SearchTask(
            "dependency_search",
            {"max_results": 3},
            dependencies,
        )

        # Mock search API
        mock_searcher = Mock()
        mock_searcher.search.return_value = Mock(
            query="artificial intelligence",
            results=[],
            total_results=0,
            search_time=0.2,
        )

        with patch(
            "aipype.framework.search_task.SerperSearcher",
            return_value=mock_searcher,
        ):
            # Simulate dependency resolution that PipelineAgent would do
            from aipype import DependencyResolver

            resolver = DependencyResolver(context)
            resolved_config = resolver.resolve_dependencies(task)
            task.config.update(resolved_config)

            result = task.run()

        # Verify query was resolved from dependency
        assert result.is_success()
        assert result.data["query"] == "artificial intelligence"
        mock_searcher.search.assert_called_once_with(
            "artificial intelligence", max_results=3
        )

    def test_search_task_requires_query(self) -> None:
        """SearchTask raises error when no query is provided."""
        task = SearchTask("no_query_search", {"max_results": 5})

        result = task.run()
        assert result.is_error()
        assert (
            result.error is not None
            and "Missing required fields: ['query']" in result.error
        )


class TestTransformTask:
    """Test suite for TransformTask - Generic data transformation task."""

    def test_transform_task_applies_simple_transformation(self) -> None:
        """TransformTask applies simple data transformations."""
        context = TaskContext()
        context.store_result(
            "search",
            {
                "results": [
                    {"url": "http://example.com/1", "title": "Article 1"},
                    {"url": "http://example.com/2", "title": "Article 2"},
                    {"url": "http://example.com/3", "title": "Article 3"},
                ]
            },
        )

        # Extract URLs transformation
        def extract_urls(search_results: List[Dict[str, Any]]) -> List[str]:
            return [result["url"] for result in search_results]

        dependencies = [
            TaskDependency("search_data", "search.results", DependencyType.REQUIRED)
        ]

        task = TransformTask(
            "extract_urls",
            {
                "input_field": "search_data",
                "transform_function": extract_urls,
                "output_name": "url_list",
            },
            dependencies,
        )

        # Simulate dependency resolution that PipelineAgent would do
        from aipype import DependencyResolver

        resolver = DependencyResolver(context)
        resolved_config = resolver.resolve_dependencies(task)
        task.config.update(resolved_config)

        task.set_context(context)
        result = task.run()

        # Verify transformation result
        assert result.is_success()
        assert result.data["url_list"] == [
            "http://example.com/1",
            "http://example.com/2",
            "http://example.com/3",
        ]
        assert result.data["transformation"] == "extract_urls"

    def test_transform_task_applies_complex_transformation(self) -> None:
        """TransformTask handles complex multi-input transformations."""
        context = TaskContext()
        context.store_result("search", {"data": {"query": "AI trends", "total": 5}})
        context.store_result(
            "fetch",
            {
                "data": {
                    "articles": [
                        {
                            "title": "AI Article 1",
                            "content": "Content 1",
                            "word_count": 500,
                        },
                        {
                            "title": "AI Article 2",
                            "content": "Content 2",
                            "word_count": 750,
                        },
                    ]
                }
            },
        )

        # Complex transformation combining multiple inputs
        def create_summary_report(
            search_data: Dict[str, Any], fetch_data: Dict[str, Any]
        ) -> Dict[str, Any]:
            return {
                "search_query": search_data["query"],
                "total_search_results": search_data["total"],
                "articles_fetched": len(fetch_data["articles"]),
                "total_words": sum(
                    article["word_count"] for article in fetch_data["articles"]
                ),
                "article_titles": [
                    article["title"] for article in fetch_data["articles"]
                ],
            }

        dependencies = [
            TaskDependency("search_data", "search.data", DependencyType.REQUIRED),
            TaskDependency("fetch_data", "fetch.data", DependencyType.REQUIRED),
        ]

        task = TransformTask(
            "create_report",
            {
                "transform_function": create_summary_report,
                "input_fields": ["search_data", "fetch_data"],
                "output_name": "summary_report",
            },
            dependencies,
        )

        # Simulate dependency resolution that PipelineAgent would do
        from aipype import DependencyResolver

        resolver = DependencyResolver(context)
        resolved_config = resolver.resolve_dependencies(task)
        task.config.update(resolved_config)

        task.set_context(context)
        result = task.run()

        # Verify complex transformation
        assert result.is_success()
        report = result.data["summary_report"]
        assert report["search_query"] == "AI trends"
        assert report["total_search_results"] == 5
        assert report["articles_fetched"] == 2
        assert report["total_words"] == 1250
        assert report["article_titles"] == ["AI Article 1", "AI Article 2"]

    def test_transform_task_with_validation(self) -> None:
        """TransformTask validates input and output data."""
        context = TaskContext()
        context.store_result("data", {"numbers": [1, 2, 3, 4, 5]})

        # Transformation with validation
        def calculate_statistics(numbers: List[int]) -> Dict[str, Any]:
            if not numbers:
                raise ValueError("Numbers list cannot be empty")

            return {
                "count": len(numbers),
                "sum": sum(numbers),
                "average": sum(numbers) / len(numbers),
                "min": min(numbers),
                "max": max(numbers),
            }

        dependencies = [
            TaskDependency("input_data", "data.numbers", DependencyType.REQUIRED)
        ]

        task = TransformTask(
            "calculate_stats",
            {
                "input_field": "input_data",
                "transform_function": calculate_statistics,
                "output_name": "statistics",
                "validate_output": True,
            },
            dependencies,
        )

        # Simulate dependency resolution that PipelineAgent would do
        from aipype import DependencyResolver

        resolver = DependencyResolver(context)
        resolved_config = resolver.resolve_dependencies(task)
        task.config.update(resolved_config)

        task.set_context(context)
        result = task.run()

        # Verify calculation results
        assert result.is_success()
        stats = result.data["statistics"]
        assert stats["count"] == 5
        assert stats["sum"] == 15
        assert stats["average"] == 3.0
        assert stats["min"] == 1
        assert stats["max"] == 5

    def test_transform_task_static_data_generator_with_no_dependencies(self) -> None:
        """TransformTask can generate static data with no dependencies."""

        # Static data generator (like setup_context in llm_agent.py)
        def generate_config(_: Any) -> Dict[str, Any]:
            return {
                "user_question": "What is AI?",
                "topic": "machine learning",
                "max_results": 10,
            }

        task = TransformTask(
            "setup_context",
            {
                "transform_function": generate_config,
                "output_name": "user_inputs",
                "validate_input": False,  # Key: disable input validation for static generators
            },
            [],  # No dependencies
        )

        result = task.run()

        # Verify static data generation
        assert result.is_success()
        assert result.data["user_inputs"]["user_question"] == "What is AI?"
        assert result.data["user_inputs"]["topic"] == "machine learning"
        assert result.data["user_inputs"]["max_results"] == 10
        assert result.data["transformation"] == "setup_context"
        assert result.data["output_type"] == "dict"

    def test_transform_task_fails_without_input_validation_disabled(self) -> None:
        """TransformTask with no dependencies fails unless input validation is disabled."""

        def generate_config(_: Any) -> Dict[str, Any]:
            return {"data": "test"}

        task = TransformTask(
            "static_generator",
            {
                "transform_function": generate_config,
                "output_name": "result",
                # Note: validate_input defaults to True
            },
            [],  # No dependencies
        )

        # Should fail because no input data available and validation enabled
        result = task.run()
        assert result.is_error()
        assert (
            result.error is not None
            and "No input data available for transformation" in result.error
        )


class TestConditionalTaskIntegration:
    """Test suite for ConditionalTask integration with contextual framework components.

    NOTE: Detailed ConditionalTask tests are in test_conditional_task.py.
    This focuses on integration with context and dependency resolution.
    """

    def test_conditional_task_pipeline_integration(self) -> None:
        """ConditionalTask integrates properly with pipeline context and dependencies."""
        context = TaskContext()

        # Simulate a pipeline where validation task has completed
        context.store_result(
            "validation",
            {
                "data": {
                    "quality_score": 8.5,
                    "total_items": 100,
                    "successful_items": 85,
                    "error_rate": 0.15,
                }
            },
        )

        # Create a quality gate condition using multiple context values
        def quality_gate_condition(validation_data: Dict[str, Any]) -> bool:
            score_ok = validation_data["quality_score"] >= 7.0
            success_rate_ok = (
                validation_data["successful_items"] / validation_data["total_items"]
            ) >= 0.8
            error_rate_ok = validation_data["error_rate"] <= 0.2
            return score_ok and success_rate_ok and error_rate_ok

        def proceed_action(validation_data: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "decision": "proceed",
                "next_phase": "content_generation",
                "confidence": "high",
                "items_to_process": validation_data["successful_items"],
            }

        dependencies = [
            TaskDependency(
                "validation_data", "validation.data", DependencyType.REQUIRED
            )
        ]

        task = ConditionalTask(
            "pipeline_quality_gate",
            {
                "condition_function": quality_gate_condition,
                "condition_inputs": ["validation_data"],
                "action_function": proceed_action,
                "action_inputs": ["validation_data"],
            },
            dependencies,
        )

        # Simulate dependency resolution that PipelineAgent would do
        from aipype import DependencyResolver

        resolver = DependencyResolver(context)
        resolved_config = resolver.resolve_dependencies(task)
        task.config.update(resolved_config)

        task.set_context(context)
        result = task.run()

        # Verify integration worked correctly
        assert result.is_success()
        assert result.data["condition_result"] is True
        assert result.data["executed"] is True
        assert result.data["action_result"]["decision"] == "proceed"
        assert result.data["action_result"]["items_to_process"] == 85
        assert result.data["action_result"]["confidence"] == "high"
