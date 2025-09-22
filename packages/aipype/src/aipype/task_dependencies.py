"""Declarative task dependency system for automated data flow in AI pipelines.

This module provides TaskDependency and DependencyResolver for creating automated
data flow between tasks in PipelineAgent workflows. Dependencies are specified
declaratively and resolved automatically, enabling complex data transformations
and pipeline orchestration without manual data passing.

Key Features:
    * **Declarative Dependencies**: Specify data flow using simple path syntax
    * **Automatic Resolution**: Dependencies resolved before task execution
    * **Data Transformation**: Optional transform functions for data preprocessing
    * **Type Safety**: Required vs optional dependencies with validation
    * **Path Access**: Flexible dot-notation path access for nested data

Dependency Flow:
    #. **Declaration**: Tasks declare dependencies using source paths
    #. **Resolution**: DependencyResolver extracts data from TaskContext
    #. **Transformation**: Optional transform functions process data
    #. **Injection**: Resolved data added to task configuration
    #. **Execution**: Task runs with all dependencies satisfied

**Quick Example**

.. code-block:: python

    # Search task produces results
    search_task = SearchTask("search", {"query": "AI news"})

    # LLM task depends on search results
    llm_task = LLMTask("summarize", {
        "prompt": "Summarize: ${articles}",
        "llm_provider": "openai"
    }, dependencies=[
        TaskDependency("articles", "search.results", REQUIRED)
    ])

    # Pipeline automatically resolves search.results -> articles

**Path Syntax**

Dependencies use dot notation to access nested data:

* `"task_name.field"` - Access field in task result
* `"task_name.data.nested"` - Access nested field
* `"task_name.results[].url"` - Extract URLs from result list
* `"task_name.metadata.count"` - Access metadata fields

**Transform Functions**

Use transform_func to preprocess dependency data:

.. code-block:: python

    TaskDependency(
        "urls",
        "search.results",
        REQUIRED,
        transform_func=lambda results: [r['url'] for r in results]
    )

**Dependency Types**

* **REQUIRED**: Task execution fails if dependency unavailable
* **OPTIONAL**: Uses default_value if dependency unavailable

    See Also:
        * TaskDependency: Individual dependency specification
        * DependencyResolver: Automatic dependency resolution engine
        * TaskContext: Shared data store for inter-task communication
        * Built-in transform functions: extract_urls_from_results, combine_article_content
"""

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from typing import override

if TYPE_CHECKING:
    from .base_task import BaseTask
from .task_context import TaskContext
from .utils.common import setup_logger


class DependencyType(Enum):
    """Enumeration of dependency types for task data flow.

    Dependency types control how missing dependencies are handled:
    * REQUIRED: Task execution fails if dependency cannot be resolved
    * OPTIONAL: Default value used if dependency unavailable

**Example**

.. code-block:: python

    # Required dependency - task fails if search_task not available
    TaskDependency("query_results", "search_task.results", REQUIRED)

    # Optional dependency - uses default if config_task unavailable
    TaskDependency("settings", "config_task.options", OPTIONAL, default_value={})
    """

    REQUIRED = "required"
    OPTIONAL = "optional"


class TaskDependency:
    """Specification for automatic data flow between tasks in a pipeline.

    TaskDependency defines how a task receives data from other tasks in the
    pipeline. Dependencies are resolved automatically by the PipelineAgent
    before task execution, with resolved data added to the task's configuration.

    The dependency system supports flexible path-based data access, optional
    data transformation, and both required and optional dependencies with
    appropriate error handling.

    Core Components:
        * **name**: Key in target task's config where resolved data is stored
        * **source_path**: Dot-notation path to source data in TaskContext
        * **dependency_type**: REQUIRED or OPTIONAL behavior for missing data
        * **transform_func**: Optional function to process resolved data
        * **default_value**: Fallback value for optional dependencies

    Path Syntax:
        Source paths use dot notation to access nested data:
        * "task.field" - Direct field access
        * "task.data.nested.value" - Nested object access
        * "task.results[].url" - Array element extraction
        * "task.metadata.statistics" - Metadata access

**Transform Functions**

Optional preprocessing of resolved data before injection:

.. code-block:: python

    def extract_titles(articles):
        return [article.get('title', 'Untitled') for article in articles]

    TaskDependency(
        "titles",
        "search.results",
        REQUIRED,
        transform_func=extract_titles
    )

**Common Patterns**

**Basic data passing:**

.. code-block:: python

    TaskDependency("search_results", "search_task.results", REQUIRED)

**Data transformation:**

.. code-block:: python

    TaskDependency(
        "article_urls",
        "search_task.results",
        REQUIRED,
        transform_func=lambda results: [r['url'] for r in results]
    )

**Optional configuration:**

.. code-block:: python

    TaskDependency(
        "processing_options",
        "config_task.settings",
        OPTIONAL,
        default_value={"batch_size": 100, "timeout": 30}
    )

**Nested data access:**

.. code-block:: python

    TaskDependency("api_endpoint", "config_task.api.endpoints.primary", REQUIRED)

**Complex transformation:**

.. code-block:: python

    def combine_and_filter(search_results):
        # Filter recent articles and combine text
        recent = [r for r in search_results if is_recent(r)]
        return ' '.join(r.get('content', '') for r in recent)

    TaskDependency(
        "combined_content",
        "search_task.results",
        REQUIRED,
        transform_func=combine_and_filter
    )

**Error Handling**

* **Required dependencies**: Missing data causes task execution failure
* **Optional dependencies**: Missing data uses default_value
* **Transform errors**: Transform function exceptions cause dependency failure
* **Path errors**: Invalid paths cause resolution failure with detailed messages

**Best Practices**

* Use descriptive names that indicate the data being passed
* Prefer specific paths over broad data passing
* Use transform functions to shape data for consuming tasks
* Provide meaningful default values for optional dependencies
* Document complex transform functions for maintainability

**Thread Safety**

TaskDependency instances are immutable after creation and thread-safe.
However, transform functions should be thread-safe if used in parallel execution.

    See Also:
        * DependencyType: REQUIRED vs OPTIONAL dependency behavior
        * DependencyResolver: Automatic resolution engine
        * TaskContext: Source of dependency data
        * Built-in transform utilities: extract_urls_from_results, etc.
    """

    def __init__(
        self,
        name: str,
        source_path: str,
        dependency_type: DependencyType,
        default_value: Any = None,
        transform_func: Optional[Callable[[Any], Any]] = None,
        override_existing: bool = False,
        description: str = "",
    ):
        """Initialize a task dependency specification.

        Creates a dependency that will be resolved automatically by the
        PipelineAgent before task execution. The resolved data will be
        injected into the task's configuration under the specified name.

        Args:
            name: Key name where resolved data will be stored in the target
                task's config dictionary. Should be descriptive and follow
                naming conventions (snake_case recommended). This is how
                the consuming task will access the dependency data.

            source_path: Dot-notation path to the source data in TaskContext.
                Format: "source_task_name.field_path". Examples:

                * "search.results" - Access results field from search task
                * "fetch.data.articles[].content" - Extract content from article array
                * "config.api.endpoints.primary" - Access nested configuration
                * "process.metadata.item_count" - Access metadata fields

            dependency_type: How to handle missing dependencies:

                * DependencyType.REQUIRED: Task execution fails if unavailable
                * DependencyType.OPTIONAL: Uses default_value if unavailable

            default_value: Value to use when optional dependency is unavailable.
                Only relevant for OPTIONAL dependencies. Can be any type that
                makes sense for the consuming task. Common patterns:

                * Empty list: []
                * Empty dict: {}
                * Default config: {"timeout": 30, "retries": 3}
                * None: None (explicit null)

            transform_func: Optional function to preprocess resolved data before
                injection. Function signature: (resolved_data: Any) -> Any.
                Common uses:

                * Extract specific fields: lambda x: [item['url'] for item in x]
                * Filter data: lambda x: [item for item in x if item['valid']]
                * Combine data: lambda x: ' '.join(x)
                * Format data: lambda x: {"processed": x, "count": len(x)}

            override_existing: Whether to override existing values in task config.

                * False (default): Only inject if key doesn't exist in config
                * True: Always inject, overriding existing config values

                Use with caution as it can override user-provided configuration.

            description: Human-readable description of this dependency's purpose.
                Helpful for documentation and debugging. Should explain what
                data is being passed and how it will be used.

        Example:
            Basic dependency::

                # Pass search results to summarization task

                TaskDependency(
                    name="search_results",
                    source_path="web_search.results",
                    dependency_type=DependencyType.REQUIRED
                )

            Transformed dependency::

                TaskDependency(
                    name="article_urls",
                    source_path="search.results",
                    dependency_type=DependencyType.REQUIRED,
                    transform_func=lambda results: [r['url'] for r in results],
                    description="Extract URLs from search results for fetching"
                )

            Optional dependency with default::

                TaskDependency(
                    name="processing_config",
                    source_path="config_loader.settings",
                    dependency_type=DependencyType.OPTIONAL,
                    default_value={"batch_size": 100, "parallel": True},
                    description="Processing configuration with sensible defaults"
                )

        Raises:
            ValueError: If name is empty, source_path is empty, or source_path
                doesn't contain a dot (invalid format).

        Note:
            * Dependencies are resolved in the order they are defined
            * Transform functions should be deterministic and thread-safe
            * Source task must complete successfully for REQUIRED dependencies
            * Path resolution supports nested objects and array access

        See Also:
            * DependencyType: For dependency type behavior
            * DependencyResolver: For resolution implementation details
            * Built-in transform functions: extract_urls_from_results, etc.
        """
        if not name:
            raise ValueError("Dependency name cannot be empty")

        if not source_path:
            raise ValueError("Source path cannot be empty")

        if source_path and "." not in source_path:
            raise ValueError("Invalid source path format - must be 'task.field' format")

        self.name = name
        self.source_path = source_path
        self.dependency_type = dependency_type
        self.default_value = default_value
        self.transform_func = transform_func
        self.override_existing = override_existing
        self.description = description

    def is_required(self) -> bool:
        """Check if this is a required dependency."""
        return self.dependency_type == DependencyType.REQUIRED

    def is_optional(self) -> bool:
        """Check if this is an optional dependency."""
        return self.dependency_type == DependencyType.OPTIONAL

    @override
    def __str__(self) -> str:
        """String representation of the dependency."""
        return f"TaskDependency(name='{self.name}', source='{self.source_path}', type={self.dependency_type.value})"

    @override
    def __repr__(self) -> str:
        """Detailed representation of the dependency."""
        return self.__str__()


class DependencyResolver:
    """Resolves task dependencies from context and builds task configuration."""

    def __init__(self, context: TaskContext):
        """Initialize dependency resolver with context.

        Args:
            context: TaskContext to resolve dependencies from
        """
        self.context = context
        self.logger = setup_logger("dependency_resolver")

    def resolve_dependencies(self, task: "BaseTask") -> Dict[str, Any]:
        """Resolve all dependencies for a task and return merged configuration.

        Args:
            task: Task instance with get_dependencies() method

        Returns:
            Dictionary with resolved dependencies merged with existing config

        Raises:
            ValueError: If required dependencies cannot be satisfied
        """
        # Start with existing task config
        resolved_config = (
            task.config.copy() if hasattr(task, "config") and task.config else {}
        )

        # Get task dependencies
        dependencies = task.get_dependencies()

        # Resolve each dependency
        for dependency in dependencies:
            try:
                resolved_value = self._resolve_single_dependency(dependency)

                # Only add to config if resolution succeeded or if we should override
                if resolved_value is not None or dependency.override_existing:
                    resolved_config[dependency.name] = resolved_value

                self.logger.debug(
                    f"Resolved dependency '{dependency.name}' for task '{task.name}'"
                )

            except Exception as e:
                if dependency.is_required():
                    error_msg = f"Required dependency '{dependency.name}' not satisfied: {str(e)}"
                    if dependency.source_path:
                        error_msg += f" (source: {dependency.source_path})"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
                else:
                    self.logger.warning(
                        f"Optional dependency '{dependency.name}' failed to resolve: {str(e)}"
                    )

        return resolved_config

    def _resolve_single_dependency(self, dependency: TaskDependency) -> Any:
        """Resolve a single dependency.

        Args:
            dependency: TaskDependency to resolve

        Returns:
            Resolved value

        Raises:
            ValueError: If dependency cannot be resolved
        """
        return self._resolve_context_dependency(dependency)

    def _resolve_context_dependency(self, dependency: TaskDependency) -> Any:
        """Resolve a dependency from context using its source path.

        Args:
            dependency: Context dependency to resolve

        Returns:
            Resolved value from context
        """
        # Get value from context
        raw_value = self.context.get_path_value(dependency.source_path)

        # If value is None, try default for optional dependencies
        if raw_value is None:
            if dependency.is_optional() and dependency.default_value is not None:
                raw_value = dependency.default_value
            elif dependency.is_required():
                raise ValueError(
                    f"Required path '{dependency.source_path}' not found in context"
                )
            else:
                # Optional dependency with no default - return None
                return None

        # Apply transformation if specified
        if dependency.transform_func:
            try:
                return dependency.transform_func(raw_value)
            except Exception as e:
                raise ValueError(
                    f"Failed to transform dependency '{dependency.name}': {str(e)}"
                )

        return raw_value

    def validate_dependencies(self, task: "BaseTask") -> List[str]:
        """Validate that all dependencies for a task can be satisfied.

        Args:
            task: Task to validate dependencies for

        Returns:
            List of validation error messages (empty if all valid)
        """
        errors: List[str] = []
        dependencies = task.get_dependencies()

        for dependency in dependencies:
            try:
                self._resolve_single_dependency(dependency)
            except Exception as e:
                if dependency.is_required():
                    errors.append(f"Required dependency '{dependency.name}': {str(e)}")
                # Optional dependencies that fail validation are just warnings

        return errors

    def get_dependency_info(self, task: "BaseTask") -> List[Dict[str, Any]]:
        """Get information about all dependencies for a task.

        Args:
            task: Task to get dependency info for

        Returns:
            List of dependency information dictionaries
        """
        dependencies = task.get_dependencies()
        info: List[Dict[str, Any]] = []

        for dependency in dependencies:
            dep_info: Dict[str, Any] = {
                "name": dependency.name,
                "source_path": dependency.source_path,
                "type": dependency.dependency_type.value,
                "required": dependency.is_required(),
                "description": dependency.description,
                "has_default": dependency.default_value is not None,
                "has_transform": dependency.transform_func is not None,
            }

            # Try to resolve and add status
            try:
                resolved_value = self._resolve_single_dependency(dependency)
                dep_info["status"] = "resolved"
                dep_info["resolved_type"] = type(resolved_value).__name__

                # Add value preview for small values
                if isinstance(resolved_value, (str, int, float, bool)):
                    dep_info["value_preview"] = str(resolved_value)
                elif isinstance(resolved_value, (list, dict)):
                    dep_info["value_preview"] = (
                        f"{type(resolved_value).__name__}({len(resolved_value)} items)"  # pyright: ignore
                    )

            except Exception as e:
                dep_info["status"] = "error" if dependency.is_required() else "warning"
                dep_info["error"] = str(e)

            info.append(dep_info)

        return info


# Utility functions for creating common dependency patterns


def create_required_dependency(
    name: str, source_path: str, transform_func: Optional[Callable[[Any], Any]] = None
) -> TaskDependency:
    """Create a required dependency.

    Args:
        name: Name of the dependency
        source_path: Context path to resolve
        transform_func: Optional transformation function

    Returns:
        TaskDependency instance
    """
    return TaskDependency(
        name, source_path, DependencyType.REQUIRED, transform_func=transform_func
    )


def create_optional_dependency(
    name: str,
    source_path: str,
    default_value: Any,
    transform_func: Optional[Callable[[Any], Any]] = None,
) -> TaskDependency:
    """Create an optional dependency with default value.

    Args:
        name: Name of the dependency
        source_path: Context path to resolve
        default_value: Default value if resolution fails
        transform_func: Optional transformation function

    Returns:
        TaskDependency instance
    """
    return TaskDependency(
        name,
        source_path,
        DependencyType.OPTIONAL,
        default_value=default_value,
        transform_func=transform_func,
    )


# Common transformation functions


def extract_urls_from_results(search_results: Dict[str, Any]) -> List[str]:
    """Extract URLs from search results.

    Args:
        search_results: Search results dictionary

    Returns:
        List of URLs
    """
    if not isinstance(search_results, dict) or "results" not in search_results:  # pyright: ignore
        return []

    results = search_results["results"]
    if not isinstance(results, list):
        return []

    urls: List[str] = []
    for result in results:  # pyright: ignore
        if isinstance(result, dict) and "url" in result:
            urls.append(result["url"])  # pyright: ignore

    return urls


def combine_article_content(
    articles: List[Dict[str, Any]], separator: str = "\n\n"
) -> str:
    """Combine content from multiple articles.

    Args:
        articles: List of article dictionaries
        separator: Separator between articles

    Returns:
        Combined content string
    """
    if not isinstance(articles, list):  # pyright: ignore
        return ""

    content_parts: List[str] = []
    for article in articles:
        if isinstance(article, dict):  # pyright: ignore
            title: str = str(article.get("title", "Untitled"))
            content: str = str(article.get("content", ""))
            if content:
                content_parts.append(f"Title: {title}\nContent: {content}")

    return separator.join(content_parts)


def format_search_query(keywords: str, filters: Optional[Dict[str, Any]] = None) -> str:
    """Format search query with optional filters.

    Args:
        keywords: Base search keywords
        filters: Optional filters to apply

    Returns:
        Formatted search query
    """
    query = keywords

    if filters:
        if "site" in filters:
            query += f" site:{filters['site']}"
        if "filetype" in filters:
            query += f" filetype:{filters['filetype']}"
        if "date_range" in filters:
            query += f" after:{filters['date_range']}"

    return query
