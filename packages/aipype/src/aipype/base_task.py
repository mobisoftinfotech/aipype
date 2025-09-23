"""Base task class and interfaces for implementing custom AI automation tasks.

This module provides the foundation for all tasks in the aipype framework.
BaseTask is the abstract base class that all task implementations must inherit
from. It provides standardized interfaces for task execution, validation,
status tracking, and error handling.

Key Features

#. **TaskResult Pattern**: All tasks return structured results instead of raising exceptions
#. **Validation Framework**: Declarative configuration validation with detailed error messages
#. **Status Tracking**: Automatic status management throughout task lifecycle
#. **Dependency Support**: Built-in support for task dependencies and data flow
#. **Error Handling**: Graceful error capture and propagation without exceptions

**Task Lifecycle**

#. **Initialization**: Task created with name, config, and dependencies
#. **Validation**: Configuration validated against task-specific rules
#. **Execution**: run() method performs the actual work
#. **Result**: TaskResult returned with success/failure status and data
#. **Context Update**: Results stored in shared TaskContext for other tasks

**Implementation Pattern**

.. code-block:: python

    class MyCustomTask(BaseTask):
        def __init__(self, name, config, dependencies=None):
            super().__init__(name, config, dependencies)
            self.validation_rules = {
                "required": ["input_file"],
                "optional": ["output_format"],
                "defaults": {"output_format": "json"},
                "types": {"input_file": str, "output_format": str}
            }

        def run(self) -> TaskResult:
            start_time = datetime.now()

            # Validate configuration
            validation_failure = self._validate_or_fail(start_time)
            if validation_failure:
                return validation_failure

            try:
                # Perform task work
                result_data = self.process_file(self.config["input_file"])
                execution_time = (datetime.now() - start_time).total_seconds()

                return TaskResult.success(
                    data=result_data,
                    execution_time=execution_time,
                    metadata={"task_type": "file_processor"}
                )
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                return TaskResult.failure(
                    error_message=f"Processing failed: {str(e)}",
                    execution_time=execution_time,
                    metadata={"task_type": "file_processor", "error_type": type(e).__name__}
                )

**Error Handling Philosophy**

Tasks should NEVER raise exceptions for operational errors. Instead:

* Return `TaskResult.failure()` for expected operational failures
* Return `TaskResult.partial()` for partial success scenarios
* Only raise exceptions for programming errors or invalid task construction
* Use detailed error messages that help users understand what went wrong

**Validation Rules Format**

The validation_rules dictionary supports:

* **required**: List of required configuration keys
* **optional**: List of optional configuration keys
* **defaults**: Default values for missing optional keys
* **types**: Expected types for configuration values
* **ranges**: Valid ranges for numeric values (min, max)
* **custom**: Custom validation functions for complex rules

See Also:

* TaskResult: Standardized return format for task execution
* TaskDependency: For creating dependencies between tasks
* PipelineAgent: For orchestrating multiple tasks together
* Task implementations: LLMTask, SearchTask, ConditionalTask, etc.
"""
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
    """Abstract base class for all task implementations in the aipype framework.

BaseTask provides the foundation for creating custom tasks that can be
orchestrated by PipelineAgent. It handles common concerns like validation,
status tracking, error handling, and dependency management, allowing
subclasses to focus on their specific functionality.

All tasks follow the TaskResult pattern where operational errors are
returned as structured results rather than raised as exceptions. This
enables graceful error handling and pipeline continuation.

Attributes:
    name: Unique identifier for this task instance
    config: Configuration dictionary with task-specific parameters
    dependencies: List of TaskDependency objects for data flow
    validation_rules: Optional validation rules for configuration
    agent_name: Name of the parent agent (set automatically)
    logger: Task-specific logger instance

Abstract Methods:
    run(): Must be implemented by subclasses to perform task work

**Common Patterns**

**Configuration validation:**

.. code-block:: python

    class MyTask(BaseTask):
        def __init__(self, name, config, dependencies=None):
            super().__init__(name, config, dependencies)
            self.validation_rules = {
                "required": ["api_key", "endpoint"],
                "optional": ["timeout", "retries"],
                "defaults": {"timeout": 30, "retries": 3},
                "types": {
                    "api_key": str,
                    "endpoint": str,
                    "timeout": int,
                    "retries": int
                },
                "ranges": {
                    "timeout": (1, 300),
                    "retries": (0, 10)
                }
            }

**Error handling with TaskResult:**

.. code-block:: python

    def run(self) -> TaskResult:
        start_time = datetime.now()

        # Always validate first
        validation_failure = self._validate_or_fail(start_time)
        if validation_failure:
            return validation_failure

        try:
            # Perform task work
            result = self.do_work()
            execution_time = (datetime.now() - start_time).total_seconds()

            return TaskResult.success(
                data=result,
                execution_time=execution_time,
                metadata={"processed_items": len(result)}
            )

        except ApiError as e:
            # Expected operational error
            execution_time = (datetime.now() - start_time).total_seconds()
            return TaskResult.failure(
                error_message=f"API call failed: {str(e)}",
                execution_time=execution_time,
                metadata={"error_type": "ApiError", "retry_recommended": True}
            )

**Working with dependencies:**

.. code-block:: python

    class ProcessDataTask(BaseTask):
        def __init__(self, name, config, dependencies=None):
            super().__init__(name, config, dependencies)
            # Dependencies will be resolved automatically before run()

        def run(self) -> TaskResult:
            # Access resolved dependency data
            input_data = self.config.get("input_data")  # From dependency
            processing_mode = self.config.get("mode", "default")  # From config

            # Process the data...

**Validation Rules**

Tasks can define validation_rules to automatically validate their
configuration. The validation system supports:

* **required**: Keys that must be present
* **optional**: Keys that are optional
* **defaults**: Default values for missing optional keys
* **types**: Expected Python types for values
* **ranges**: (min, max) tuples for numeric validation
* **custom**: Custom validation functions

**Status Management**

Task status is managed automatically:

* **NOT_STARTED**: Initial state
* **RUNNING**: During execution (set by framework)
* **SUCCESS**: Task completed successfully
* **ERROR**: Task failed with error
* **PARTIAL**: Task completed with partial success

**Thread Safety**

Individual task instances are not thread-safe and should not be
executed concurrently. However, different task instances can run
in parallel safely.

    See Also:
        * TaskResult: Return format for task execution
        * TaskDependency: For creating task dependencies
        * PipelineAgent: For orchestrating multiple tasks
        * Validation documentation in utils.common module
    """

    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List["TaskDependency"]] = None,
    ) -> None:
        """Initialize a new task instance with configuration and dependencies.

Creates a new task with the specified configuration. The task will
be initialized with default status and logging setup. Dependencies
will be resolved automatically by the PipelineAgent before execution.

Args:
    name: Unique identifier for this task within the agent. Must be
        unique across all tasks in the same pipeline. Used for:
        * Logging identification
        * Dependency references (other_task.field)
        * Result storage in TaskContext
        * Status tracking and error reporting
    config: Configuration dictionary containing task-specific parameters.
        The exact keys depend on the task implementation. Common patterns:
        * API credentials and endpoints
        * File paths and processing options
        * Behavioral flags and timeout values
        * Data transformation parameters
        Will be validated against validation_rules if defined.
    dependencies: List of TaskDependency objects specifying how this
        task receives data from other tasks. Dependencies are resolved
        automatically before run() is called, with resolved data added
        to the config dictionary. Use TaskDependency.create_required()
        or TaskDependency.create_optional() for convenience.

Attributes Initialized:
    * name: Task identifier
    * config: Configuration dictionary (may be updated by dependencies)
    * dependencies: List of dependency specifications
    * validation_rules: Set by subclasses for configuration validation
    * agent_name: Set automatically by PipelineAgent
    * logger: Task-specific logger (task.{name})
    * Status tracking fields for execution monitoring

**Example**

**Basic task creation:**

.. code-block:: python

    task = MyTask(
        name="process_data",
        config={
            "input_file": "data.csv",
            "output_format": "json",
            "batch_size": 1000
        }
    )

**Task with dependencies:**

.. code-block:: python

    task = TransformTask(
        name="process_results",
        config={
            "transform_function": my_transform_func,
            "output_name": "processed_data"
        },
        dependencies=[
            TaskDependency("input_data", "search_task.results", REQUIRED),
            TaskDependency("options", "config_task.settings", OPTIONAL)
        ]
    )

        Note:
            * Task names should be descriptive and use snake_case
            * Config validation occurs during run(), not during initialization
            * Dependencies are resolved by PipelineAgent, not by the task itself
            * Subclasses should call super().__init__() and set validation_rules

        See Also:
            * TaskDependency: For creating task dependencies
            * TaskResult: Return format from run() method
            * Validation rules documentation for config validation format
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
        """Execute the task and return a structured result.

        This is the main method that subclasses must implement to perform
        their specific functionality. The method should follow the TaskResult
        pattern for error handling and return structured results.

        Returns:
            TaskResult containing:
            * status: SUCCESS, PARTIAL, ERROR, or SKIPPED
            * data: Result data (dict) if successful, None if failed
            * error: Error message if failed, None if successful
            * execution_time: Time taken to execute in seconds
            * metadata: Additional information about execution

**Implementation Guidelines**

#. **Validation First**: Always validate configuration before work:

   .. code-block:: python

       def run(self) -> TaskResult:
           start_time = datetime.now()

           validation_failure = self._validate_or_fail(start_time)
           if validation_failure:
               return validation_failure

#. **Error Handling**: Use TaskResult pattern, not exceptions:

   .. code-block:: python

       try:
           result = self.do_work()
           return TaskResult.success(data=result, execution_time=elapsed)
       except ExpectedError as e:
           return TaskResult.failure(error_message=str(e), execution_time=elapsed)

#. **Timing**: Always include execution timing:

   .. code-block:: python

       start_time = datetime.now()
       # ... do work ...
       execution_time = (datetime.now() - start_time).total_seconds()

#. **Metadata**: Include useful execution information:

   .. code-block:: python

       return TaskResult.success(
           data=result,
           execution_time=execution_time,
           metadata={
               "processed_items": len(items),
               "api_calls": call_count,
               "cache_hits": cache_hits
           }
       )

**Return Patterns**

* **Success**: All work completed successfully
* **Partial**: Some work completed, some failed (recoverable)
* **Failure**: Task could not complete due to errors
* **Skipped**: Task was skipped due to conditions not being met

**Example Implementation**

.. code-block:: python

    def run(self) -> TaskResult:
        start_time = datetime.now()

        # Validate configuration
        validation_failure = self._validate_or_fail(start_time)
        if validation_failure:
            return validation_failure

        try:
            # Extract configuration
            input_file = self.config["input_file"]
            output_format = self.config.get("output_format", "json")

            # Perform work
            data = self.process_file(input_file)
            formatted_data = self.format_output(data, output_format)

            execution_time = (datetime.now() - start_time).total_seconds()

            return TaskResult.success(
                data={"processed_data": formatted_data, "format": output_format},
                execution_time=execution_time,
                metadata={"records_processed": len(data)}
            )

        except FileNotFoundError:
            execution_time = (datetime.now() - start_time).total_seconds()
            return TaskResult.failure(
                error_message=f"Input file not found: {input_file}",
                execution_time=execution_time,
                metadata={"error_type": "FileNotFoundError"}
            )

        Note:
            * Never raise exceptions for operational errors
            * Always calculate and include execution_time
            * Use descriptive error messages that help users understand issues
            * Include relevant metadata for debugging and monitoring

        See Also:
            * TaskResult: For understanding return value structure
            * _validate_or_fail(): For configuration validation pattern
            * Task-specific implementations for concrete examples
        """
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
