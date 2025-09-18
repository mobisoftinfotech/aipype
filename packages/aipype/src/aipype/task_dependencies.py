"""TaskDependency system - declarative dependency specification for tasks."""

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from typing import override

if TYPE_CHECKING:
    from .base_task import BaseTask
from .task_context import TaskContext
from .utils.common import setup_logger


class DependencyType(Enum):
    """Types of task dependencies."""

    REQUIRED = "required"
    OPTIONAL = "optional"


class TaskDependency:
    """Represents a dependency specification for a task."""

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
        """Initialize task dependency.

        Args:
            name: Name of the dependency in the resolved config
            source_path: Context path to resolve (e.g., 'search.results[].url')
            dependency_type: Type of dependency (required, optional)
            default_value: Default value for optional dependencies
            transform_func: Function to transform the resolved value
            override_existing: Whether to override existing config values
            description: Human-readable description of the dependency
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
