"""Task result classes for standardized response format."""

from dataclasses import dataclass, field
from typing import Optional, Any, Dict
from enum import Enum

from typing import override


class TaskStatus(Enum):
    """Enumeration for task execution status."""

    # Task lifecycle statuses
    NOT_STARTED = "not_started"
    STARTED = "started"

    # Task completion statuses
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"
    PARTIAL = "partial"


@dataclass
class TaskResult:
    """Standardized result format for all task executions.

    This class provides a consistent interface for task results across
    the framework, enabling better error handling, performance tracking,
    and result processing.
    """

    status: TaskStatus
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=lambda: {})
    execution_time: float = 0.0

    @classmethod
    def success(
        cls,
        data: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        execution_time: float = 0.0,
    ) -> "TaskResult":
        """Create a successful task result.

        Args:
            data: The result data
            metadata: Optional metadata dictionary
            execution_time: Task execution time in seconds

        Returns:
            TaskResult with SUCCESS status
        """
        return cls(
            status=TaskStatus.SUCCESS,
            data=data,
            metadata=metadata or {},
            execution_time=execution_time,
        )

    @classmethod
    def failure(
        cls,
        error_message: str,
        metadata: Optional[Dict[str, Any]] = None,
        execution_time: float = 0.0,
    ) -> "TaskResult":
        """Create an error task result.

        Args:
            error_message: Description of the error
            metadata: Optional metadata dictionary
            execution_time: Task execution time in seconds

        Returns:
            TaskResult with ERROR status
        """
        return cls(
            status=TaskStatus.ERROR,
            error=error_message,
            metadata=metadata or {},
            execution_time=execution_time,
        )

    @classmethod
    def partial(
        cls,
        data: Any = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        execution_time: float = 0.0,
    ) -> "TaskResult":
        """Create a partial success task result.

        Args:
            data: The partial result data
            error: Optional description of what failed
            metadata: Optional metadata dictionary
            execution_time: Task execution time in seconds

        Returns:
            TaskResult with PARTIAL status
        """
        return cls(
            status=TaskStatus.PARTIAL,
            data=data,
            error=error,
            metadata=metadata or {},
            execution_time=execution_time,
        )

    @classmethod
    def skipped(
        cls,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
        execution_time: float = 0.0,
    ) -> "TaskResult":
        """Create a skipped task result.

        Args:
            reason: Reason why the task was skipped
            metadata: Optional metadata dictionary
            execution_time: Task execution time in seconds

        Returns:
            TaskResult with SKIPPED status
        """
        return cls(
            status=TaskStatus.SKIPPED,
            error=reason,
            metadata=metadata or {},
            execution_time=execution_time,
        )

    def is_success(self) -> bool:
        """Check if the task completed successfully.

        Returns:
            True if status is SUCCESS
        """
        return self.status == TaskStatus.SUCCESS

    def is_error(self) -> bool:
        """Check if the task failed with an error.

        Returns:
            True if status is ERROR
        """
        return self.status == TaskStatus.ERROR

    def is_partial(self) -> bool:
        """Check if the task completed with partial success.

        Returns:
            True if status is PARTIAL
        """
        return self.status == TaskStatus.PARTIAL

    def is_skipped(self) -> bool:
        """Check if the task was skipped.

        Returns:
            True if status is SKIPPED
        """
        return self.status == TaskStatus.SKIPPED

    def has_data(self) -> bool:
        """Check if the result contains data.

        Returns:
            True if data is not None
        """
        return self.data is not None

    def get_legacy_result(self) -> Any:
        """Get result in legacy format for backward compatibility.

        This method extracts the data field to maintain compatibility
        with existing code that expects raw return values.

        Returns:
            The data field, or raises RuntimeError if task failed
        """
        if self.status == TaskStatus.SUCCESS:
            return self.data
        elif self.status == TaskStatus.PARTIAL:
            # For partial results, return data but maybe log a warning
            return self.data
        elif self.status == TaskStatus.SKIPPED:
            return None
        else:
            # For errors, raise an exception to maintain existing error behavior
            raise RuntimeError(self.error or "Task failed")

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata entry.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value.

        Args:
            key: Metadata key
            default: Default value if key not found

        Returns:
            Metadata value or default
        """
        return self.metadata.get(key, default)

    @override
    def __str__(self) -> str:
        """String representation of TaskResult."""
        status_str = self.status.value

        if self.execution_time > 0:
            status_str += f" ({self.execution_time:.3f}s)"

        if self.error:
            return f"TaskResult({status_str}, error='{self.error}')"
        elif self.has_data():
            data_type = type(self.data).__name__
            return f"TaskResult({status_str}, data={data_type})"
        else:
            return f"TaskResult({status_str})"


# Utility functions for working with TaskResult


def wrap_legacy_result(result: Any, execution_time: float = 0.0) -> TaskResult:
    """Wrap a legacy task result in TaskResult format.

    This utility function helps with gradual migration by wrapping
    existing task results in the new standardized format.

    Args:
        result: Legacy task result (any type)
        execution_time: Task execution time in seconds

    Returns:
        TaskResult with SUCCESS status containing the legacy result
    """
    return TaskResult.success(data=result, execution_time=execution_time)


def unwrap_to_legacy(task_result: TaskResult) -> Any:
    """Unwrap TaskResult to legacy format for backward compatibility.

    Args:
        task_result: TaskResult instance

    Returns:
        The data field from the TaskResult

    Raises:
        RuntimeError: If the task failed
    """
    return task_result.get_legacy_result()
