"""Task wrapper for decorated @task methods.

This module provides TaskWrapper, a BaseTask subclass that wraps
decorated @task methods and enables them to work with the existing
pipeline execution engine.

Example::

    # TaskWrapper is created automatically by PipelineAgent
    # You typically don't create it directly

    wrapper = TaskWrapper(
        name="process_data",
        method=agent.process_data,  # Bound method decorated with @task
        agent=agent,
        dependencies=[TaskDependency("fetch_data", "fetch_data.data", REQUIRED)]
    )

    result = wrapper.run()  # Executes the method and processes return value
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, List, Optional, cast, override

from pydantic import BaseModel

from .base_task import BaseTask
from .task_result import TaskResult

if TYPE_CHECKING:
    from .task_context import TaskContext
    from .task_dependencies import TaskDependency


class TaskWrapper(BaseTask):
    """Wraps a decorated @task method as a BaseTask.

    TaskWrapper bridges the gap between the new decorator-based syntax
    and the existing execution engine. It:

    1. Stores a reference to the decorated method
    2. Builds kwargs from resolved dependencies
    3. Calls the method and processes return values
    4. Handles delegation to returned BaseTask instances

    This allows the existing TaskExecutionPlan and DependencyResolver
    to work without modification.

    Attributes:
        method: The bound method to call
        agent: Reference to the parent agent
        context_instance: TaskContext for dependency resolution

    Example::

        class MyAgent(PipelineAgent):
            @task
            def fetch_data(self) -> dict:
                return {"data": "value"}

            @task
            def process(self, fetch_data: dict) -> str:
                return llm(f"Process: {fetch_data}")

        # TaskWrapper is created automatically for each @task method
        # and handles the execution and return value processing
    """

    def __init__(
        self,
        name: str,
        method: Callable[..., Any],
        agent: Any,  # PipelineAgent, but avoid circular import
        dependencies: List["TaskDependency"],
    ) -> None:
        """Initialize a task wrapper.

        Args:
            name: Task name (usually the method name)
            method: The bound method to call
            agent: Reference to the parent agent
            dependencies: Inferred TaskDependency objects
        """
        super().__init__(name, {}, dependencies)
        self.method = method
        self.agent = agent
        self.context_instance: Optional["TaskContext"] = None

    @override
    def set_context(self, context: "TaskContext") -> None:
        """Set the task context for dependency resolution.

        Args:
            context: TaskContext instance
        """
        self.context_instance = context

    @override
    def run(self) -> TaskResult:
        """Execute the wrapped method and process its return value.

        This method:
        1. Builds kwargs from resolved dependencies in self.config
        2. Calls the decorated method with those kwargs
        3. Processes the return value based on its type

        Returns:
            TaskResult with the execution result
        """
        start_time = datetime.now()

        try:
            # Build kwargs from resolved dependencies (already in self.config)
            kwargs = self._build_kwargs()

            # Call the decorated method
            result = self.method(**kwargs)

            # Process the return value
            return self._process_return_value(result, start_time)

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = (
                f"TaskWrapper execution failed: Task '{self.name}' failed: {str(e)}"
            )
            self.logger.error(error_msg)

            return TaskResult.failure(
                error_message=error_msg,
                execution_time=execution_time,
                metadata={
                    "task_name": self.name,
                    "error_type": type(e).__name__,
                },
            )

    def _build_kwargs(self) -> dict[str, Any]:
        """Build kwargs dict from resolved dependencies.

        Dependencies are resolved by DependencyResolver and stored in
        self.config before run() is called.

        Returns:
            Dictionary of parameter names to resolved values
        """
        kwargs: dict[str, Any] = {}

        for dep in self.dependencies:
            # Value should already be resolved into self.config by DependencyResolver
            if dep.name in self.config:
                kwargs[dep.name] = self.config[dep.name]
            elif dep.is_optional() and dep.default_value is not None:
                kwargs[dep.name] = dep.default_value
            # Required deps missing will have already caused an error in resolver

        return kwargs

    def _process_return_value(
        self,
        result: Any,
        start_time: datetime,
    ) -> TaskResult:
        """Process the return value from the wrapped method.

        Handles different return types:
        - TaskResult: Return as-is
        - BaseTask: Execute and return its result (delegation)
        - Pydantic BaseModel: Convert to dict
        - dict: Wrap in TaskResult.success
        - Other (str, list, etc.): Wrap with standard structure

        Args:
            result: Return value from the method
            start_time: When execution started

        Returns:
            TaskResult with appropriate data
        """
        execution_time = (datetime.now() - start_time).total_seconds()

        # Already a TaskResult - return as-is
        if isinstance(result, TaskResult):
            return result

        # BaseTask returned - delegate execution
        if isinstance(result, BaseTask):
            return self._delegate_to_task(result, start_time)

        # Pydantic model - convert to dict with 'data' key for consistent access
        if isinstance(result, BaseModel):
            model_dict = result.model_dump()
            # Start with model fields, then add 'data' to ensure it's always present
            wrapped_data = {**model_dict, "data": model_dict}
            return TaskResult.success(
                data=wrapped_data,
                execution_time=execution_time,
                metadata={"task_name": self.name, "return_type": "pydantic"},
            )

        # Dict - wrap with 'data' key for consistent access via task_name.data
        # Also preserve original keys for explicit path access (task_name.key)
        # Note: If the result dict has a 'data' key, it will be accessible
        # via task_name.data.data for explicit access
        if isinstance(result, dict):
            # Start with original keys, then add 'data' to ensure it's always present
            wrapped_data: dict[str, Any] = {**result, "data": result}
            return TaskResult.success(
                data=wrapped_data,
                execution_time=execution_time,
                metadata={"task_name": self.name, "return_type": "dict"},
            )

        # Primitive types (str, int, list, etc.) - wrap with standard structure
        return TaskResult.success(
            data={"output": result, "data": result},
            execution_time=execution_time,
            metadata={"task_name": self.name, "return_type": type(result).__name__},
        )

    def _delegate_to_task(
        self,
        delegated_task: BaseTask,
        start_time: datetime,
    ) -> TaskResult:
        """Execute a delegated BaseTask and return its result.

        When a @task method returns a BaseTask (like LLMTask or SearchTask),
        we execute it and return the combined result.

        Args:
            delegated_task: The BaseTask to execute
            start_time: Original start time

        Returns:
            TaskResult from the delegated task
        """
        # Set context on delegated task
        if self.context_instance:
            delegated_task.set_context(self.context_instance)

        # Copy agent name
        if self.agent_name:
            delegated_task.set_agent_name(self.agent_name)

        # Execute the delegated task
        delegated_result = delegated_task.run()

        # Add wrapper overhead to execution time
        total_time = (datetime.now() - start_time).total_seconds()

        # Wrap result data with 'data' key for consistent path access (like dict returns)
        # This ensures task_name.data path works for delegated BaseTask results
        if delegated_result.is_success() and isinstance(delegated_result.data, dict):
            # Cast to dict[str, Any] - isinstance confirms dict, but pyright narrows
            # to dict[Unknown, Unknown] since TaskResult.data is typed as Any
            result_data = cast(
                dict[str, Any],
                delegated_result.data,  # pyright: ignore[reportUnknownMemberType]
            )
            wrapped_data: dict[str, Any] = {
                **result_data,
                "data": result_data,
            }
            return TaskResult.success(
                data=wrapped_data,
                execution_time=total_time,
                metadata={
                    **delegated_result.metadata,
                    "delegated_from": self.name,
                    "delegated_to": delegated_task.name,
                },
            )

        # For non-dict or non-success results, return as-is with updated time
        delegated_result.execution_time = total_time
        delegated_result.add_metadata("delegated_from", self.name)
        delegated_result.add_metadata("delegated_to", delegated_task.name)

        return delegated_result

    @override
    def __str__(self) -> str:
        """String representation."""
        dep_count = len(self.dependencies)
        return f"TaskWrapper(name='{self.name}', dependencies={dep_count})"
