"""Agent run result classes for standardized pipeline execution response format."""

from dataclasses import dataclass, field
from typing import Optional, Any, Dict
from enum import Enum

from typing import override


class AgentRunStatus(Enum):
    """Enumeration for agent execution status."""

    SUCCESS = "success"
    PARTIAL = "partial"
    ERROR = "error"
    RUNNING = "running"


@dataclass
class AgentRunResult:
    """Standardized result format for agent pipeline executions.

    This class provides a consistent interface for agent execution results,
    enabling better error handling, status checking, and result processing.

    Note: Individual task results are accessible via agent.context.get_result(task_name)
    """

    status: AgentRunStatus
    agent_name: str
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_phases: int = 0
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=lambda: {})
    error_message: Optional[str] = None

    @classmethod
    def success(
        cls,
        agent_name: str,
        total_tasks: int,
        completed_tasks: int,
        total_phases: int,
        execution_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AgentRunResult":
        """Create a successful agent execution result.

        Args:
            agent_name: Name of the agent
            total_tasks: Total number of tasks in the pipeline
            completed_tasks: Number of tasks that completed successfully
            total_phases: Number of execution phases
            execution_time: Total execution time in seconds
            metadata: Optional metadata dictionary

        Returns:
            AgentRunResult with SUCCESS status
        """
        return cls(
            status=AgentRunStatus.SUCCESS,
            agent_name=agent_name,
            total_tasks=total_tasks,
            completed_tasks=completed_tasks,
            failed_tasks=total_tasks - completed_tasks,
            total_phases=total_phases,
            execution_time=execution_time,
            metadata=metadata or {},
        )

    @classmethod
    def partial(
        cls,
        agent_name: str,
        total_tasks: int,
        completed_tasks: int,
        failed_tasks: int,
        total_phases: int,
        execution_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AgentRunResult":
        """Create a partial success agent execution result.

        Args:
            agent_name: Name of the agent
            total_tasks: Total number of tasks in the pipeline
            completed_tasks: Number of tasks that completed successfully
            failed_tasks: Number of tasks that failed
            total_phases: Number of execution phases
            execution_time: Total execution time in seconds
            metadata: Optional metadata dictionary

        Returns:
            AgentRunResult with PARTIAL status
        """
        return cls(
            status=AgentRunStatus.PARTIAL,
            agent_name=agent_name,
            total_tasks=total_tasks,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            total_phases=total_phases,
            execution_time=execution_time,
            metadata=metadata or {},
        )

    @classmethod
    def failure(
        cls,
        agent_name: str,
        total_tasks: int,
        failed_tasks: int,
        error_message: str,
        total_phases: int = 0,
        execution_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AgentRunResult":
        """Create a failed agent execution result.

        Args:
            agent_name: Name of the agent
            total_tasks: Total number of tasks in the pipeline
            failed_tasks: Number of tasks that failed
            error_message: Description of the error
            total_phases: Number of execution phases
            execution_time: Total execution time in seconds
            metadata: Optional metadata dictionary

        Returns:
            AgentRunResult with ERROR status
        """
        return cls(
            status=AgentRunStatus.ERROR,
            agent_name=agent_name,
            total_tasks=total_tasks,
            completed_tasks=total_tasks - failed_tasks,
            failed_tasks=failed_tasks,
            total_phases=total_phases,
            execution_time=execution_time,
            metadata=metadata or {},
            error_message=error_message,
        )

    @classmethod
    def running(
        cls,
        agent_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AgentRunResult":
        """Create a running agent execution result.

        Args:
            agent_name: Name of the agent
            metadata: Optional metadata dictionary

        Returns:
            AgentRunResult with RUNNING status
        """
        return cls(
            status=AgentRunStatus.RUNNING,
            agent_name=agent_name,
            metadata=metadata or {},
        )

    def is_success(self) -> bool:
        """Check if the agent execution completed successfully.

        Returns:
            True if status is SUCCESS
        """
        return self.status == AgentRunStatus.SUCCESS

    def is_error(self) -> bool:
        """Check if the agent execution failed with an error.

        Returns:
            True if status is ERROR
        """
        return self.status == AgentRunStatus.ERROR

    def is_partial(self) -> bool:
        """Check if the agent execution completed with partial success.

        Returns:
            True if status is PARTIAL
        """
        return self.status == AgentRunStatus.PARTIAL

    def is_running(self) -> bool:
        """Check if the agent is currently running.

        Returns:
            True if status is RUNNING
        """
        return self.status == AgentRunStatus.RUNNING

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
        """String representation of AgentRunResult."""
        status_str = self.status.value

        if self.execution_time > 0:
            status_str += f" ({self.execution_time:.3f}s)"

        task_info = f"{self.completed_tasks}/{self.total_tasks} tasks"

        if self.error_message:
            return f"AgentRunResult({self.agent_name}: {status_str}, {task_info}, error='{self.error_message}')"
        else:
            return f"AgentRunResult({self.agent_name}: {status_str}, {task_info})"
