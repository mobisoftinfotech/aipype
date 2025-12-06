"""Decorators for declarative task definition.

This module provides the @task decorator and Depends class for defining
pipeline tasks with automatic dependency inference.

Example::

    from aipype import PipelineAgent, task, Depends
    from typing import Annotated

    class MyAgent(PipelineAgent):
        @task
        def fetch_data(self) -> dict:
            return search("AI news", max_results=5)

        @task
        def process(self, fetch_data: dict) -> str:
            # fetch_data parameter auto-wired from fetch_data task
            return llm(f"Summarize: {fetch_data}", model="gpt-4o")

        @task
        def with_explicit_path(
            self,
            content: Annotated[str, Depends("process.content")]
        ) -> str:
            return llm(f"Expand: {content}", model="gpt-4o")
"""

from functools import wraps
from typing import Any, Callable, TypeVar, override

F = TypeVar("F", bound=Callable[..., Any])


def task(func: F) -> F:
    """Mark a method as a pipeline task with automatic dependency inference.

    The @task decorator marks methods in a PipelineAgent as pipeline
    tasks. Dependencies are automatically inferred from the method's parameter
    names - if a parameter name matches another task's name, it becomes a
    dependency.

    Args:
        func: The method to mark as a task

    Returns:
        The decorated method with task metadata attached

    Example::

        class MyAgent(PipelineAgent):
            @task
            def fetch_data(self) -> dict:
                return search("AI news", max_results=5)

            @task
            def process(self, fetch_data: dict) -> str:
                # fetch_data parameter auto-wired from fetch_data task
                return llm(f"Summarize: {fetch_data}", model="gpt-4o")
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    # Attach task metadata
    wrapper._is_pipeline_task = True  # type: ignore[attr-defined]
    wrapper._task_name = func.__name__  # type: ignore[attr-defined]

    return wrapper  # type: ignore[return-value]


class Depends:
    """Specify an explicit dependency path for a task parameter.

    Use with typing.Annotated to specify the exact path to resolve
    from the task context, rather than using the default inference.

    Attributes:
        path: Dot-notation path to the dependency value (e.g., "task_name.field")

    Example:
        from typing import Annotated

        @task
        def write_article(
            self,
            # Explicitly get .content field instead of full task output
            outline: Annotated[str, Depends("generate_outline.content")],
            summary: Annotated[str, Depends("summarize.content")]
        ) -> str:
            return llm(f"Write from: {outline}", model="gpt-4o")
    """

    __slots__ = ("path",)

    def __init__(self, path: str) -> None:
        """Initialize with a dependency path.

        Args:
            path: Dot-notation path like "task_name.field.nested"

        Raises:
            ValueError: If path is empty or doesn't contain a dot
        """
        if not path or "." not in path:
            raise ValueError(
                f"Depends path must be in 'task.field' format, got: '{path}'"
            )
        self.path = path

    @override
    def __repr__(self) -> str:
        """Return string representation."""
        return f"Depends({self.path!r})"

    def __class_getitem__(cls, path: str) -> "Depends":
        """Allow Depends["path"] syntax as alternative to Depends("path").

        Args:
            path: Dot-notation path like "task_name.field"

        Returns:
            Depends instance with the specified path

        Example:
            param: Annotated[str, Depends["task.field"]]
        """
        return cls(path)
