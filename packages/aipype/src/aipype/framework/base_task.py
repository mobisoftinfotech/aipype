"""Base task class for agent tasks."""
# pyright: reportImportCycles=false

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from typing import override
from datetime import datetime
from .utils.common import setup_logger
from .task_result import TaskStatus, TaskResult

if TYPE_CHECKING:
    from .task_dependencies import TaskDependency
    from .task_context import TaskContext


class BaseTask(ABC):
    """Abstract base class for all agent tasks."""

    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List["TaskDependency"]] = None,
    ) -> None:
        """Initialize the task with a name, optional configuration, and dependencies.

        Args:
            name: Task name
            config: Task configuration dictionary
            dependencies: List of TaskDependency objects for this task
        """
        self.name = name
        self.config = config or {}
        self.dependencies = dependencies or []
        self.validation_rules: Optional[Dict[str, Any]] = None
        self.agent_name: Optional[str] = None
        self.logger = setup_logger(f"task.{name}")
        self._status = TaskStatus.NOT_STARTED
        self._status_changed_at = datetime.now()
        self._result: Optional[Any] = None
        self._error: Optional[str] = None
        self._execution_start: Optional[datetime] = None
        self._execution_time: float = 0.0

    def _validate(self) -> Optional[str]:
        """Validate task configuration using instance validation rules.

        Returns:
            Error message string if validation fails, None if valid
        """
        if not self.validation_rules:
            return None  # No validation rules defined

        from .utils.common import validate_task_config

        return validate_task_config(self.name, self.config, self.validation_rules)

    def _validate_or_fail(self, start_time: datetime) -> Optional[TaskResult]:
        """Validate configuration and return TaskResult.failure() if validation fails.

        Args:
            start_time: Task execution start time for calculating execution_time

        Returns:
            TaskResult.failure() if validation fails, None if validation passes
        """
        validation_error = self._validate()
        if validation_error:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(validation_error)
            return TaskResult.failure(
                error_message=validation_error,
                execution_time=execution_time,
                metadata={"task_name": self.name, "error_type": "ValidationError"},
            )
        return None

    @abstractmethod
    def run(self) -> TaskResult:
        """Execute the task and return a TaskResult."""
        pass

    def get_status(self) -> TaskStatus:
        """Get the current status of the task."""
        return self._status

    def is_completed(self) -> bool:
        """Check if the task has been completed successfully."""
        return self._status == TaskStatus.SUCCESS

    def has_error(self) -> bool:
        """Check if the task has encountered an error."""
        return self._status == TaskStatus.ERROR

    def get_result(self) -> Optional[Any]:
        """Get the result of the task if completed successfully."""
        return self._result if self._status == TaskStatus.SUCCESS else None

    def get_error(self) -> Optional[str]:
        """Get the error message if task failed."""
        return self._error if self._status == TaskStatus.ERROR else None

    @property
    def status_changed_at(self) -> datetime:
        """Get the timestamp when the status was last changed."""
        return self._status_changed_at

    def _change_status(
        self,
        new_status: TaskStatus,
        result: Optional[Any] = None,
        error: Optional[str] = None,
    ) -> None:
        """Change the task status and update timestamp."""
        old_status = self._status
        self._status = new_status
        self._status_changed_at = datetime.now()

        if new_status == TaskStatus.SUCCESS:
            self._result = result
            self._error = None
        elif new_status == TaskStatus.ERROR:
            self._error = error
            self._result = None
        elif new_status == TaskStatus.NOT_STARTED:
            self._result = None
            self._error = None

        self.logger.info(
            f"Task '{self.name}' status changed from {old_status.value} to {new_status.value}"
        )

    def _calculate_execution_time(self) -> None:
        """Calculate execution time if task was started."""
        if self._execution_start is not None:
            end_time = datetime.now()
            self._execution_time = (end_time - self._execution_start).total_seconds()

    def mark_started(self) -> None:
        """Mark the task as started."""
        self._execution_start = datetime.now()
        self._change_status(TaskStatus.STARTED)

    def mark_success(self, result: Any = None) -> None:
        """Mark the task as successfully completed with an optional result."""
        self._calculate_execution_time()
        self._change_status(TaskStatus.SUCCESS, result=result)

    def mark_error(self, error: str) -> None:
        """Mark the task as failed with an error message."""
        self._calculate_execution_time()
        self._change_status(TaskStatus.ERROR, error=error)

    def reset(self) -> None:
        """Reset the task to its initial state."""
        self._change_status(TaskStatus.NOT_STARTED)
        self._execution_start = None
        self._execution_time = 0.0
        self.logger.info(f"Task '{self.name}' reset")

    def get_execution_time(self) -> float:
        """Get the task execution time in seconds."""
        return self._execution_time

    def create_task_result_from_current_state(self) -> TaskResult:
        """Create a TaskResult object from current task state.

        This method is useful for backward compatibility and migration purposes.
        """
        if self._status == TaskStatus.SUCCESS:
            return TaskResult.success(
                data=self._result,
                execution_time=self._execution_time,
                metadata={"task_name": self.name, "agent_name": self.agent_name},
            )
        elif self._status == TaskStatus.ERROR:
            return TaskResult.failure(
                error_message=self._error or "Unknown error",
                execution_time=self._execution_time,
                metadata={"task_name": self.name, "agent_name": self.agent_name},
            )
        elif self._status == TaskStatus.SKIPPED:
            return TaskResult.skipped(
                reason=self._error or "Task was skipped",
                execution_time=self._execution_time,
                metadata={"task_name": self.name, "agent_name": self.agent_name},
            )
        else:
            # Task not completed yet or in unknown state
            return TaskResult.skipped(
                reason=f"Task in {self._status.value} state",
                execution_time=self._execution_time,
                metadata={"task_name": self.name, "agent_name": self.agent_name},
            )

    def get_dependencies(self) -> List["TaskDependency"]:
        """Get the list of dependencies for this task.

        Returns:
            List of TaskDependency objects for this task.
        """
        return self.dependencies

    def set_context(self, context: "TaskContext") -> None:
        """Set the task context for dependency resolution.

        Args:
            context: TaskContext instance for resolving dependencies

        Note:
            Default implementation does nothing. Override in subclasses that use context.
        """
        pass

    def set_agent_name(self, agent_name: str) -> None:
        """Set the name of the agent that owns this task.

        Args:
            agent_name: Name of the agent that owns this task
        """
        self.agent_name = agent_name
        self.logger.debug(f"Task '{self.name}' assigned to agent '{agent_name}'")

    @override
    def __str__(self) -> str:
        """String representation of the task."""
        return f"Task(name='{self.name}', status='{self._status.value}', changed_at='{self._status_changed_at.isoformat()}')"
