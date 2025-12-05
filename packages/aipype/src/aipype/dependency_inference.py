"""Automatic dependency inference from function signatures.

This module provides utilities for inferring task dependencies from
method signatures. It analyzes parameter names and type hints to
automatically create TaskDependency objects.

Example:
    @task
    def process(
        self,
        fetch_data: dict,  # Inferred: depends on "fetch_data" task
        config: Annotated[str, Depends("settings.api_key")]  # Explicit path
    ) -> str:
        ...

    deps = infer_dependencies_from_signature(process, {"fetch_data"})
    # Returns: [TaskDependency("fetch_data", "fetch_data.data", REQUIRED),
    #           TaskDependency("config", "settings.api_key", REQUIRED)]
"""

import inspect
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Set,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from .decorators import Depends
from .task_dependencies import DependencyType, TaskDependency


def infer_dependencies_from_signature(
    method: Callable[..., Any],
    known_task_names: Set[str],
) -> List[TaskDependency]:
    """Infer TaskDependency objects from a method's signature.

    This function analyzes a method's parameters and type hints to
    automatically create TaskDependency objects. Dependencies are
    inferred when:

    1. A parameter name matches a known task name
    2. A parameter has Annotated[T, Depends("path")] type hint

    Args:
        method: The method to analyze
        known_task_names: Set of task names that can be dependencies

    Returns:
        List of TaskDependency objects

    Example:
        @task
        def process(
            self,
            fetch_data: dict,  # Inferred: depends on "fetch_data" task
            config: Annotated[str, Depends("settings.api_key")]  # Explicit path
        ) -> str:
            ...

        deps = infer_dependencies_from_signature(process, {"fetch_data"})
    """
    sig = inspect.signature(method)

    # Get type hints including Annotated metadata
    try:
        hints = get_type_hints(method, include_extras=True)
    except Exception:
        # Fallback if type hints can't be resolved
        hints = {}

    dependencies: List[TaskDependency] = []

    for param_name, param in sig.parameters.items():
        # Skip 'self' parameter
        if param_name == "self":
            continue

        hint = hints.get(param_name)
        source_path: Optional[str] = None
        dep_type = DependencyType.REQUIRED
        default_value: Any = None

        # Check for explicit Depends in Annotated type
        source_path = _extract_depends_path(hint)

        # If no explicit path, infer from parameter name
        if source_path is None and param_name in known_task_names:
            # Default to getting the full task output data
            source_path = f"{param_name}.data"

        # Check if optional (has default or Optional type)
        if param.default is not inspect.Parameter.empty:
            dep_type = DependencyType.OPTIONAL
            default_value = param.default
        elif _is_optional_type(hint):
            dep_type = DependencyType.OPTIONAL
            default_value = None

        # Only create dependency if we have a source path
        if source_path is not None:
            dependencies.append(
                TaskDependency(
                    name=param_name,
                    source_path=source_path,
                    dependency_type=dep_type,
                    default_value=default_value,
                )
            )

    return dependencies


def _extract_depends_path(hint: Any) -> Optional[str]:
    """Extract Depends path from an Annotated type hint.

    Args:
        hint: Type hint to analyze

    Returns:
        The dependency path if found, None otherwise
    """
    if hint is None:
        return None

    # Check if it's Annotated[T, Depends(...)]
    if hasattr(hint, "__metadata__"):
        for meta in hint.__metadata__:
            if isinstance(meta, Depends):
                return meta.path

    return None


def _is_optional_type(hint: Any) -> bool:
    """Check if a type hint represents an Optional type.

    Args:
        hint: Type hint to check

    Returns:
        True if the type is Optional[T] or Union[T, None]
    """
    if hint is None:
        return False

    origin = get_origin(hint)

    # Check for Union types (Optional is Union[T, None])
    if origin is Union:
        args = get_args(hint)
        return type(None) in args

    return False


def get_dependency_task_name(source_path: str) -> str:
    """Extract the task name from a dependency source path.

    Args:
        source_path: Dot-notation path like "task_name.field.nested"

    Returns:
        The task name (first part before the dot)

    Example:
        get_dependency_task_name("fetch_data.results") -> "fetch_data"
    """
    return source_path.split(".")[0]
