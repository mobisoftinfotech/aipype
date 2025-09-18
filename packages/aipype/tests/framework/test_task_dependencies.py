"""Tests for TaskDependency system - declarative dependency specification for tasks."""

import pytest
from typing import Any, Dict, List, Optional

from typing import override
from aipype import (
    TaskDependency,
    DependencyType,
    DependencyResolver,
    create_required_dependency,
    create_optional_dependency,
    extract_urls_from_results,
    combine_article_content,
    format_search_query,
    TaskContext,
    BaseTask,
)


class MockTask(BaseTask):
    """Mock task for testing dependency system."""

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        dependencies: Optional[List[TaskDependency]] = None,
    ):
        super().__init__(name, config, dependencies)
        self.run_called = False
        self.resolved_config: Dict[str, Any] = {}

    @override
    def run(self) -> Any:
        self.run_called = True
        return {"task_name": self.name, "config": self.config}

    def set_resolved_config(self, config: Dict[str, Any]) -> None:
        self.resolved_config = config


class TestTaskDependency:
    """Test suite for TaskDependency class."""

    def test_task_declares_required_dependencies(self) -> None:
        """Tasks can declare required dependencies that must be satisfied."""
        # Create required dependencies
        url_dep = TaskDependency(
            name="urls",
            source_path="search.results[].url",
            dependency_type=DependencyType.REQUIRED,
            description="URLs from search results",
        )

        content_dep = TaskDependency(
            name="search_query",
            source_path="search.query",
            dependency_type=DependencyType.REQUIRED,
            description="Original search query",
        )

        # Verify dependency properties
        assert url_dep.name == "urls"
        assert url_dep.source_path == "search.results[].url"
        assert url_dep.dependency_type == DependencyType.REQUIRED
        assert url_dep.is_required()
        assert not url_dep.is_optional()

        assert content_dep.name == "search_query"
        assert content_dep.is_required()

    def test_task_declares_optional_dependencies(self) -> None:
        """Tasks can declare optional dependencies with default values."""
        # Create optional dependency with default
        timeout_dep = TaskDependency(
            name="timeout",
            source_path="config.timeout",
            dependency_type=DependencyType.OPTIONAL,
            default_value=30,
            description="Request timeout in seconds",
        )

        max_results_dep = TaskDependency(
            name="max_results",
            source_path="search.max_results",
            dependency_type=DependencyType.OPTIONAL,
            default_value=5,
            description="Maximum search results",
        )

        # Verify optional properties
        assert timeout_dep.is_optional()
        assert not timeout_dep.is_required()
        assert timeout_dep.default_value == 30

        assert max_results_dep.is_optional()
        assert max_results_dep.default_value == 5

    def test_dependency_validation(self) -> None:
        """TaskDependency validates input parameters."""
        # Valid dependency
        dep = TaskDependency(
            name="valid_dep",
            source_path="task.field",
            dependency_type=DependencyType.REQUIRED,
        )
        assert dep.name == "valid_dep"

        # Invalid dependency - empty name
        with pytest.raises(ValueError, match="Dependency name cannot be empty"):
            TaskDependency(
                name="",
                source_path="task.field",
                dependency_type=DependencyType.REQUIRED,
            )

        # Invalid dependency - empty source path
        with pytest.raises(ValueError, match="Source path cannot be empty"):
            TaskDependency(
                name="dep", source_path="", dependency_type=DependencyType.REQUIRED
            )

        # Invalid dependency - malformed source path
        with pytest.raises(ValueError, match="Invalid source path format"):
            TaskDependency(
                name="dep",
                source_path="invalid_path",  # Missing task.field format
                dependency_type=DependencyType.REQUIRED,
            )

    def test_dependency_type_enum(self) -> None:
        """DependencyType enum provides correct values."""
        assert DependencyType.REQUIRED.value == "required"
        assert DependencyType.OPTIONAL.value == "optional"

        # Test all enum members
        all_types = [
            DependencyType.REQUIRED,
            DependencyType.OPTIONAL,
        ]
        assert len(all_types) == 2


class TestDependencyResolver:
    """Test suite for DependencyResolver class."""

    def test_resolver_validates_required_dependencies(self) -> None:
        """DependencyResolver fails when required dependencies are missing."""
        context = TaskContext()
        resolver = DependencyResolver(context)

        # Create task with required dependency
        url_dep = TaskDependency(
            name="urls",
            source_path="search.results[].url",
            dependency_type=DependencyType.REQUIRED,
        )

        task = MockTask("test_task", {}, [url_dep])

        # Should fail because search task hasn't run
        with pytest.raises(
            ValueError, match="Required dependency 'urls' not satisfied"
        ):
            resolver.resolve_dependencies(task)

    def test_resolver_uses_default_values(self) -> None:
        """DependencyResolver uses defaults when optional dependencies missing."""
        context = TaskContext()
        resolver = DependencyResolver(context)

        # Create task with optional dependencies
        timeout_dep = TaskDependency(
            name="timeout",
            source_path="config.timeout",
            dependency_type=DependencyType.OPTIONAL,
            default_value=30,
        )

        max_dep = TaskDependency(
            name="max_results",
            source_path="search.max_results",
            dependency_type=DependencyType.OPTIONAL,
            default_value=10,
        )

        task = MockTask("test_task", {}, [timeout_dep, max_dep])

        # Should succeed and use defaults
        resolved_config = resolver.resolve_dependencies(task)
        assert resolved_config["timeout"] == 30
        assert resolved_config["max_results"] == 10

    def test_dependency_path_resolution(self) -> None:
        """Complex paths like '${search.results[].url}' resolve correctly."""
        context = TaskContext()
        resolver = DependencyResolver(context)

        # Setup context with search results
        search_result = {
            "query": "test query",
            "results": [
                {"url": "http://example.com/1", "title": "Article 1"},
                {"url": "http://example.com/2", "title": "Article 2"},
                {"url": "http://example.com/3", "title": "Article 3"},
            ],
            "total": 3,
        }
        context.store_result("search", search_result)

        # Create dependencies for different path types
        urls_dep = TaskDependency(
            name="urls",
            source_path="search.results[].url",
            dependency_type=DependencyType.REQUIRED,
        )

        query_dep = TaskDependency(
            name="original_query",
            source_path="search.query",
            dependency_type=DependencyType.REQUIRED,
        )

        first_url_dep = TaskDependency(
            name="first_url",
            source_path="search.results[0].url",
            dependency_type=DependencyType.REQUIRED,
        )

        task = MockTask("test_task", {}, [urls_dep, query_dep, first_url_dep])

        # Resolve dependencies
        resolved_config = resolver.resolve_dependencies(task)

        # Verify array extraction
        assert resolved_config["urls"] == [
            "http://example.com/1",
            "http://example.com/2",
            "http://example.com/3",
        ]

        # Verify simple field access
        assert resolved_config["original_query"] == "test query"

        # Verify array index access
        assert resolved_config["first_url"] == "http://example.com/1"

    def test_dependency_transformation(self) -> None:
        """Dependencies can transform data during resolution."""
        context = TaskContext()
        resolver = DependencyResolver(context)

        # Setup context with article content
        fetch_result = {
            "articles": [
                {"title": "Article 1", "content": "Content 1", "word_count": 500},
                {"title": "Article 2", "content": "Content 2", "word_count": 750},
            ]
        }
        context.store_result("fetch", fetch_result)

        # Create dependency with transformation function
        def combine_content(articles: List[Dict[str, Any]]) -> str:
            return "\n\n".join(
                [f"{art['title']}: {art['content']}" for art in articles]
            )

        content_dep = TaskDependency(
            name="combined_content",
            source_path="fetch.articles",
            dependency_type=DependencyType.REQUIRED,
            transform_func=combine_content,
        )

        task = MockTask("test_task", {}, [content_dep])

        # Resolve with transformation
        resolved_config = resolver.resolve_dependencies(task)

        expected_content = "Article 1: Content 1\n\nArticle 2: Content 2"
        assert resolved_config["combined_content"] == expected_content

    def test_resolver_handles_missing_optional_dependencies(self) -> None:
        """Resolver gracefully handles missing optional dependencies."""
        context = TaskContext()
        resolver = DependencyResolver(context)

        # Create mix of available and missing optional dependencies
        available_dep = TaskDependency(
            name="timeout",
            source_path="config.timeout",
            dependency_type=DependencyType.OPTIONAL,
            default_value=30,
        )

        missing_dep = TaskDependency(
            name="api_key",
            source_path="auth.api_key",
            dependency_type=DependencyType.OPTIONAL,
            default_value="default_key",
        )

        # Store only one dependency source
        context.store_result("config", {"timeout": 60})
        # auth.api_key is missing, should use default

        task = MockTask("test_task", {}, [available_dep, missing_dep])

        # Should succeed with mix of resolved and default values
        resolved_config = resolver.resolve_dependencies(task)
        assert resolved_config["timeout"] == 60  # From context
        assert resolved_config["api_key"] == "default_key"  # Default value

    def test_resolver_preserves_existing_config(self) -> None:
        """Resolver preserves existing task config and adds dependencies."""
        context = TaskContext()
        resolver = DependencyResolver(context)

        # Setup context
        context.store_result("search", {"query": "test"})

        # Create task with existing config
        existing_config = {"temperature": 0.7, "max_tokens": 1000, "provider": "openai"}

        query_dep = TaskDependency(
            name="search_query",
            source_path="search.query",
            dependency_type=DependencyType.REQUIRED,
        )

        task = MockTask("test_task", existing_config.copy(), [query_dep])

        # Resolve dependencies
        resolved_config = resolver.resolve_dependencies(task)

        # Should have both existing config and resolved dependencies
        assert resolved_config["temperature"] == 0.7
        assert resolved_config["max_tokens"] == 1000
        assert resolved_config["provider"] == "openai"
        assert resolved_config["search_query"] == "test"

    def test_resolver_dependency_override_behavior(self) -> None:
        """Resolver can override existing config or preserve it based on settings."""
        context = TaskContext()
        resolver = DependencyResolver(context)

        # Setup context
        context.store_result("search", {"max_results": 20})

        # Create task with config that conflicts with dependency
        existing_config = {"max_results": 10}

        override_dep = TaskDependency(
            name="max_results",
            source_path="search.max_results",
            dependency_type=DependencyType.REQUIRED,
            override_existing=True,
        )

        task = MockTask("test_task", existing_config.copy(), [override_dep])

        # Should override existing config value
        resolved_config = resolver.resolve_dependencies(task)
        assert (
            resolved_config["max_results"] == 20
        )  # From dependency, not original config

    def test_resolver_error_handling(self) -> None:
        """Resolver provides clear error messages for dependency failures."""
        context = TaskContext()
        resolver = DependencyResolver(context)

        # Missing required dependency
        missing_dep = TaskDependency(
            name="required_field",
            source_path="missing_task.field",
            dependency_type=DependencyType.REQUIRED,
        )

        task = MockTask("test_task", {}, [missing_dep])

        with pytest.raises(ValueError) as exc_info:
            resolver.resolve_dependencies(task)

        error_msg = str(exc_info.value)
        assert "Required dependency 'required_field' not satisfied" in error_msg
        assert "missing_task.field" in error_msg


class TestIntegratedDependencySystem:
    """Integration tests for the complete dependency system."""

    def test_end_to_end_dependency_resolution(self) -> None:
        """Complete workflow from task setup to dependency resolution."""
        context = TaskContext()
        resolver = DependencyResolver(context)

        # Simulate search task completion
        search_result = {
            "query": "artificial intelligence",
            "results": [
                {"url": "http://ai.com/article1", "title": "AI Basics"},
                {"url": "http://ai.com/article2", "title": "AI Advanced"},
            ],
            "total": 2,
        }
        context.store_result("search_task", search_result)

        # Simulate fetch task completion
        fetch_result = {
            "successful_fetches": 2,
            "articles": [
                {
                    "title": "AI Basics",
                    "content": "AI content 1",
                    "url": "http://ai.com/article1",
                },
                {
                    "title": "AI Advanced",
                    "content": "AI content 2",
                    "url": "http://ai.com/article2",
                },
            ],
        }
        context.store_result("fetch_task", fetch_result)

        # Create LLM task with complex dependencies
        prompt_template_dep = TaskDependency(
            name="article_content",
            source_path="fetch_task.articles",
            dependency_type=DependencyType.REQUIRED,
            transform_func=lambda articles: "\n".join(
                [f"Title: {a['title']}\nContent: {a['content']}" for a in articles]
            ),
        )

        query_dep = TaskDependency(
            name="search_query",
            source_path="search_task.query",
            dependency_type=DependencyType.REQUIRED,
        )

        temperature_dep = TaskDependency(
            name="temperature",
            source_path="config.llm_temperature",
            dependency_type=DependencyType.OPTIONAL,
            default_value=0.7,
        )

        llm_config = {
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "context": "You are an expert writer",
        }

        llm_task = MockTask(
            "llm_task", llm_config, [prompt_template_dep, query_dep, temperature_dep]
        )

        # Resolve all dependencies
        resolved_config = resolver.resolve_dependencies(llm_task)

        # Verify complete resolution
        assert "llm_provider" in resolved_config  # Original config preserved
        assert "llm_model" in resolved_config
        assert "context" in resolved_config

        assert resolved_config["search_query"] == "artificial intelligence"
        assert resolved_config["temperature"] == 0.7  # Default used
        assert (
            "Title: AI Basics" in resolved_config["article_content"]
        )  # Transformation applied
        assert "Title: AI Advanced" in resolved_config["article_content"]

    def test_task_dependency_string_representations(self) -> None:
        """TaskDependency has proper string representations."""
        dep = TaskDependency(
            name="test_dep",
            source_path="task.field",
            dependency_type=DependencyType.REQUIRED,
            description="Test dependency",
        )

        str_repr = str(dep)
        assert "TaskDependency" in str_repr
        assert "test_dep" in str_repr
        assert "task.field" in str_repr
        assert "required" in str_repr

        # __repr__ should match __str__
        assert repr(dep) == str(dep)

    def test_dependency_resolver_validate_dependencies(self) -> None:
        """DependencyResolver.validate_dependencies returns validation errors."""
        context = TaskContext()
        resolver = DependencyResolver(context)

        # Setup partial context data
        context.store_result("available_task", {"field": "value"})

        # Create task with mix of valid and invalid dependencies
        valid_dep = TaskDependency(
            name="valid_field",
            source_path="available_task.field",
            dependency_type=DependencyType.REQUIRED,
        )

        invalid_dep = TaskDependency(
            name="invalid_field",
            source_path="missing_task.field",
            dependency_type=DependencyType.REQUIRED,
        )

        optional_invalid_dep = TaskDependency(
            name="optional_field",
            source_path="missing_task.other_field",
            dependency_type=DependencyType.OPTIONAL,
        )

        task = MockTask("test_task", {}, [valid_dep, invalid_dep, optional_invalid_dep])

        # Validate dependencies
        errors = resolver.validate_dependencies(task)

        # Should have error for required dependency only
        assert len(errors) == 1
        assert "Required dependency 'invalid_field'" in errors[0]
        assert "missing_task.field" in errors[0]

    def test_dependency_resolver_get_dependency_info(self) -> None:
        """DependencyResolver.get_dependency_info returns comprehensive dependency info."""
        context = TaskContext()
        resolver = DependencyResolver(context)

        # Setup context with various data types
        context.store_result("string_task", {"field": "text_value"})
        context.store_result("number_task", {"field": 42})
        context.store_result("list_task", {"field": [1, 2, 3]})
        context.store_result("dict_task", {"field": {"nested": "value"}})

        # Create dependencies with various configurations
        string_dep = TaskDependency(
            name="string_field",
            source_path="string_task.field",
            dependency_type=DependencyType.REQUIRED,
            description="String dependency",
        )

        number_dep = TaskDependency(
            name="number_field",
            source_path="number_task.field",
            dependency_type=DependencyType.OPTIONAL,
            default_value=0,
        )

        list_dep = TaskDependency(
            name="list_field",
            source_path="list_task.field",
            dependency_type=DependencyType.REQUIRED,
            transform_func=lambda x: len(x),
        )

        dict_dep = TaskDependency(
            name="dict_field",
            source_path="dict_task.field",
            dependency_type=DependencyType.REQUIRED,
        )

        missing_dep = TaskDependency(
            name="missing_field",
            source_path="nonexistent.field",
            dependency_type=DependencyType.REQUIRED,
        )

        task = MockTask(
            "test_task", {}, [string_dep, number_dep, list_dep, dict_dep, missing_dep]
        )

        # Get dependency info
        info_list = resolver.get_dependency_info(task)

        assert len(info_list) == 5

        # Check string dependency info
        string_info = next(info for info in info_list if info["name"] == "string_field")
        assert string_info["source_path"] == "string_task.field"
        assert string_info["type"] == "required"
        assert string_info["required"] is True
        assert string_info["description"] == "String dependency"
        assert string_info["status"] == "resolved"
        assert string_info["resolved_type"] == "str"
        assert string_info["value_preview"] == "text_value"

        # Check number dependency info
        number_info = next(info for info in info_list if info["name"] == "number_field")
        assert number_info["type"] == "optional"
        assert number_info["has_default"] is True
        assert number_info["value_preview"] == "42"

        # Check list dependency info (with transformation)
        list_info = next(info for info in info_list if info["name"] == "list_field")
        assert list_info["has_transform"] is True
        assert list_info["resolved_type"] == "int"  # After transformation
        assert list_info["value_preview"] == "3"  # len([1,2,3])

        # Check dict dependency info
        dict_info = next(info for info in info_list if info["name"] == "dict_field")
        assert dict_info["resolved_type"] == "dict"
        assert "dict(1 items)" in dict_info["value_preview"]

        # Check missing dependency info
        missing_info = next(
            info for info in info_list if info["name"] == "missing_field"
        )
        assert missing_info["status"] == "error"
        assert "error" in missing_info
        assert "not found in context" in missing_info["error"]

    def test_transform_function_error_handling(self) -> None:
        """Dependencies handle transformation function errors gracefully."""
        context = TaskContext()
        resolver = DependencyResolver(context)

        # Setup context
        context.store_result("task", {"field": "invalid_input"})

        # Create dependency with failing transform function
        def failing_transform(value: Any) -> int:
            return int(value)  # Will fail for "invalid_input"

        transform_dep = TaskDependency(
            name="transformed_field",
            source_path="task.field",
            dependency_type=DependencyType.REQUIRED,
            transform_func=failing_transform,
        )

        task = MockTask("test_task", {}, [transform_dep])

        # Should raise ValueError with transformation error
        with pytest.raises(ValueError) as exc_info:
            resolver.resolve_dependencies(task)

        error_msg = str(exc_info.value)
        assert "Failed to transform dependency 'transformed_field'" in error_msg

    def test_optional_dependency_without_default_returns_none(self) -> None:
        """Optional dependencies without defaults return None when missing."""
        context = TaskContext()
        resolver = DependencyResolver(context)

        # Create optional dependency without default value
        optional_dep = TaskDependency(
            name="optional_field",
            source_path="missing_task.field",
            dependency_type=DependencyType.OPTIONAL,
            # No default_value specified
        )

        task = MockTask("test_task", {}, [optional_dep])

        # Should succeed and not include the dependency in config
        resolved_config = resolver.resolve_dependencies(task)

        # Optional dependency with None value should not be included in config
        assert "optional_field" not in resolved_config


class TestUtilityFunctions:
    """Test suite for utility functions in task_dependencies module."""

    def test_create_required_dependency(self) -> None:
        """create_required_dependency creates properly configured required dependency."""
        # Test basic required dependency
        dep = create_required_dependency("test_field", "source.path")

        assert dep.name == "test_field"
        assert dep.source_path == "source.path"
        assert dep.dependency_type == DependencyType.REQUIRED
        assert dep.is_required()
        assert dep.transform_func is None

        # Test with transform function
        def transform_func(x: Any) -> str:
            return str(x).upper()

        dep_with_transform = create_required_dependency(
            "transformed_field", "source.path", transform_func=transform_func
        )

        assert dep_with_transform.transform_func is transform_func

    def test_create_optional_dependency(self) -> None:
        """create_optional_dependency creates properly configured optional dependency."""
        # Test basic optional dependency
        dep = create_optional_dependency("test_field", "source.path", "default_value")

        assert dep.name == "test_field"
        assert dep.source_path == "source.path"
        assert dep.dependency_type == DependencyType.OPTIONAL
        assert dep.is_optional()
        assert dep.default_value == "default_value"
        assert dep.transform_func is None

        # Test with transform function
        def transform_func(x: Any) -> int:
            return len(str(x))

        dep_with_transform = create_optional_dependency(
            "transformed_field", "source.path", 42, transform_func=transform_func
        )

        assert dep_with_transform.transform_func is transform_func
        assert dep_with_transform.default_value == 42

    def test_extract_urls_from_results(self) -> None:
        """extract_urls_from_results extracts URLs from search results properly."""
        # Test with valid search results
        search_results = {
            "results": [
                {"url": "http://example.com/1", "title": "Article 1"},
                {"url": "http://example.com/2", "title": "Article 2"},
                {"url": "http://example.com/3", "title": "Article 3"},
            ]
        }

        urls = extract_urls_from_results(search_results)
        expected_urls = [
            "http://example.com/1",
            "http://example.com/2",
            "http://example.com/3",
        ]
        assert urls == expected_urls

        # Test with empty results
        empty_results: Dict[str, List[Any]] = {"results": []}
        assert extract_urls_from_results(empty_results) == []

        # Test with missing results key
        invalid_results = {"query": "test"}
        assert extract_urls_from_results(invalid_results) == []

        # Test with non-dict input
        assert extract_urls_from_results("invalid") == []  # pyright: ignore
        assert extract_urls_from_results(None) == []  # pyright: ignore

        # Test with results that aren't a list
        invalid_list_results = {"results": "not_a_list"}
        assert extract_urls_from_results(invalid_list_results) == []

        # Test with results missing URL field
        missing_url_results = {
            "results": [
                {"title": "Article 1"},  # Missing url
                {"url": "http://example.com/2", "title": "Article 2"},
                {"title": "Article 3"},  # Missing url
            ]
        }
        urls = extract_urls_from_results(missing_url_results)
        assert urls == ["http://example.com/2"]

    def test_combine_article_content(self) -> None:
        """combine_article_content combines articles with proper formatting."""
        # Test with valid articles
        articles = [
            {"title": "Article 1", "content": "Content 1"},
            {"title": "Article 2", "content": "Content 2"},
            {"title": "Article 3", "content": "Content 3"},
        ]

        combined = combine_article_content(articles)
        expected = (
            "Title: Article 1\nContent: Content 1\n\n"
            "Title: Article 2\nContent: Content 2\n\n"
            "Title: Article 3\nContent: Content 3"
        )
        assert combined == expected

        # Test with custom separator
        combined_custom = combine_article_content(articles, separator=" | ")
        expected_custom = (
            "Title: Article 1\nContent: Content 1 | "
            "Title: Article 2\nContent: Content 2 | "
            "Title: Article 3\nContent: Content 3"
        )
        assert combined_custom == expected_custom

        # Test with empty articles
        assert combine_article_content([]) == ""

        # Test with non-list input
        assert combine_article_content("invalid") == ""  # pyright: ignore
        assert combine_article_content(None) == ""  # pyright: ignore

        # Test with articles missing title or content
        incomplete_articles = [
            {"title": "Article 1"},  # Missing content
            {"content": "Content 2"},  # Missing title
            {"title": "Article 3", "content": "Content 3"},  # Complete
        ]

        combined_incomplete = combine_article_content(incomplete_articles)
        # Should skip articles without content but include complete ones
        assert "Title: Article 3\nContent: Content 3" in combined_incomplete
        # Articles without content should not appear in output
        assert "Title: Article 1\nContent:" not in combined_incomplete

    def test_format_search_query(self) -> None:
        """format_search_query formats queries with filters properly."""
        # Test basic query without filters
        query = format_search_query("machine learning")
        assert query == "machine learning"

        # Test with site filter
        query_with_site = format_search_query(
            "python tutorial", {"site": "stackoverflow.com"}
        )
        assert query_with_site == "python tutorial site:stackoverflow.com"

        # Test with filetype filter
        query_with_filetype = format_search_query("research paper", {"filetype": "pdf"})
        assert query_with_filetype == "research paper filetype:pdf"

        # Test with date range filter
        query_with_date = format_search_query(
            "news articles", {"date_range": "2023-01-01"}
        )
        assert query_with_date == "news articles after:2023-01-01"

        # Test with multiple filters
        query_with_multiple = format_search_query(
            "AI research",
            {"site": "arxiv.org", "filetype": "pdf", "date_range": "2023-01-01"},
        )
        expected_multiple = "AI research site:arxiv.org filetype:pdf after:2023-01-01"
        assert query_with_multiple == expected_multiple

        # Test with None filters
        query_with_none = format_search_query("test query", None)
        assert query_with_none == "test query"

        # Test with empty filters dict
        query_with_empty = format_search_query("test query", {})
        assert query_with_empty == "test query"

        # Test with unknown filter keys (should be ignored)
        query_with_unknown = format_search_query(
            "test query", {"unknown_filter": "value", "site": "example.com"}
        )
        assert query_with_unknown == "test query site:example.com"
