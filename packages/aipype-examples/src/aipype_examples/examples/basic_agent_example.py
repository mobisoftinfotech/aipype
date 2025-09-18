"""Example agent implementation using modern pipeline architecture."""

import time
from typing import List, Dict, Any, Optional, override

from aipype import (
    PipelineAgent,
    BaseTask,
    TransformTask,
    ConditionalTask,
    TaskDependency,
    DependencyType,
    TaskResult,
    print_message_box,
)


class PrintTask(BaseTask):
    """A simple task that prints a message."""

    @override
    def run(self) -> TaskResult:
        message = self.config.get("message", f"Executing task: {self.name}")
        print(message)
        self.logger.info(f"Printed message: {message}")
        return TaskResult.success(data={"message": message})


class DelayTask(BaseTask):
    """A task that waits for a specified number of seconds."""

    @override
    def run(self) -> TaskResult:
        delay = self.config.get("delay", 1.0)
        self.logger.info(f"Waiting for {delay} seconds...")
        time.sleep(delay)
        return TaskResult.success(data={"delay": delay})


class CalculationTask(BaseTask):
    """A task that performs a simple calculation."""

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        dependencies: Optional[List[TaskDependency]] = None,
    ):
        """Initialize calculation task with optional dependencies."""
        super().__init__(name, config)
        self.dependencies = dependencies or []
        self.validation_rules = {
            "defaults": {
                "operation": "add",
                "numbers": [1, 2, 3],
            },
            "types": {
                "operation": str,
                "numbers": list,
            },
            "custom": {
                "operation": lambda x: x in ["add", "multiply"],  # pyright: ignore[reportUnknownLambdaType]
                # Validate that numbers list is non-empty and contains only numeric types
                "numbers": lambda x: len(x) > 0  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
                and all(isinstance(n, (int, float)) for n in x),  # pyright: ignore[reportUnknownVariableType]
            },
        }

    @override
    def get_dependencies(self) -> List[TaskDependency]:
        """Get task dependencies."""
        return self.dependencies

    @override
    def run(self) -> TaskResult:
        from datetime import datetime

        start_time = datetime.now()

        # Validate configuration using new pattern
        validation_error = self._validate_or_fail(start_time)
        if validation_error:
            return validation_error

        # Try to get configuration from dependencies first, fall back to config
        calc_config = self.config.get("calc_config")
        if calc_config and isinstance(calc_config, dict):
            operation = str(
                calc_config.get(  # pyright: ignore
                    "operation", self.config.get("operation", "add")
                )
            )
            numbers_raw = calc_config.get(  # pyright: ignore
                "numbers", self.config.get("numbers", [1, 2, 3])
            )
            numbers = list(numbers_raw) if numbers_raw else []  # pyright: ignore
        else:
            operation = str(self.config.get("operation", "add"))
            numbers = list(self.config.get("numbers", [1, 2, 3]))

        # Perform calculation
        try:
            if operation == "add":
                result = sum(numbers)  # type: ignore
            elif operation == "multiply":
                result = 1
                for num in numbers:  # type: ignore
                    result *= num  # type: ignore
            else:
                # This should not happen due to validation, but keep as safety
                execution_time = (datetime.now() - start_time).total_seconds()
                error_msg = (
                    f"CalculationTask operation failed: Unknown operation: {operation}"
                )
                self.logger.error(error_msg)
                return TaskResult.failure(
                    error_message=error_msg,
                    execution_time=execution_time,
                    metadata={
                        "task_name": self.name,
                        "error_type": "ValueError",
                        "operation": operation,
                    },
                )

            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Calculated {operation} of {numbers} = {result}")
            return TaskResult.success(
                data={"result": result, "operation": operation, "numbers": numbers},
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"CalculationTask operation failed: {str(e)}"
            self.logger.error(error_msg)
            return TaskResult.failure(
                error_message=error_msg,
                execution_time=execution_time,
                metadata={
                    "task_name": self.name,
                    "error_type": type(e).__name__,
                },
            )


class ExampleAgent(PipelineAgent):
    """Example agent that demonstrates the modern pipeline system."""

    @override
    def setup_tasks(self) -> List[BaseTask]:
        """Set up example tasks with dependencies to demonstrate pipeline capabilities."""

        # Create and return all tasks
        return [
            self._create_welcome_task(),
            self._create_numbers_task(),
            self._create_prepare_calc_task(),
            self._create_calc_task(),
            self._create_double_task(),
            self._create_conditional_task(),
            self._create_farewell_task(),
        ]

    def _create_welcome_task(self) -> PrintTask:
        """Create the welcome message task.

        Returns:
            Configured PrintTask for welcome message
        """
        return PrintTask(
            "welcome", {"message": "[START] Starting Modern Example Agent!"}
        )

    def _create_numbers_task(self) -> PrintTask:
        """Create the numbers setup task.

        Returns:
            Configured PrintTask for numbers setup message
        """
        return PrintTask(
            "set_numbers", {"message": "Setting up calculation numbers..."}
        )

    def _create_prepare_calc_task(self) -> TransformTask:
        """Create the calculation preparation task.

        Returns:
            Configured TransformTask for preparing calculation data
        """
        return TransformTask(
            "prepare_calculation",
            {
                "transform_function": lambda _: {  # type: ignore
                    "operation": "add",
                    "numbers": [10, 20, 30],
                },
                "output_name": "calc_config",
                "validate_input": False,  # No input validation needed for static data generator
            },
        )

    def _create_calc_task(self) -> CalculationTask:
        """Create the calculation task.

        Returns:
            Configured CalculationTask for performing calculation
        """
        return CalculationTask(
            "add_numbers",
            {},
            [
                TaskDependency(
                    "calc_config",
                    "prepare_calculation.calc_config",
                    DependencyType.REQUIRED,
                )
            ],
        )

    def _create_double_task(self) -> TransformTask:
        """Create the double result task.

        Returns:
            Configured TransformTask for doubling the calculation result
        """
        return TransformTask(
            "double_result",
            {
                "transform_function": lambda result: int(result) * 2,  # type: ignore
                "output_name": "doubled_value",
            },
            [TaskDependency("result", "add_numbers.result", DependencyType.REQUIRED)],
        )

    def _create_conditional_task(self) -> ConditionalTask:
        """Create the conditional result check task.

        Returns:
            Configured ConditionalTask for checking result size
        """
        return ConditionalTask(
            "check_result",
            {
                "condition_function": lambda doubled_value: int(doubled_value) > 100,  # type: ignore
                "condition_inputs": [
                    "doubled_value"
                ],  # Specify inputs for the condition function
                "action_function": lambda: "Result is large! [SUCCESS]",
                "else_function": lambda: "Result is small. [STATS]",
                "action_field": "doubled_value",
            },
            [
                TaskDependency(
                    "doubled_value",
                    "double_result.doubled_value",
                    DependencyType.REQUIRED,
                )
            ],
        )

    def _create_farewell_task(self) -> PrintTask:
        """Create the farewell message task.

        Returns:
            Configured PrintTask for farewell message
        """
        return PrintTask(
            "farewell",
            {"message": "[SUCCESS] Modern Example Agent completed successfully!"},
        )


def main() -> None:
    """Demonstrate the modern pipeline agent with dependency resolution."""
    print_message_box(
        "[DEMO] MODERN EXAMPLE AGENT DEMONSTRATION",
        [
            "This example shows the new pipeline architecture where:",
            "- Tasks declare dependencies instead of manual sequencing",
            "- Framework automatically resolves execution order",
            "- Data flows between tasks via dependency injection",
            "- Tasks can transform and process results from other tasks",
        ],
    )

    agent = ExampleAgent(
        "modern_example_agent",
        {
            "enable_parallel": False,  # Sequential for clear demonstration
            "stop_on_failure": True,
        },
    )

    agent.setup_tasks()
    agent.run()
    agent.display_results()

    print_message_box(
        "[COMPLETE] DEMONSTRATION COMPLETE",
        [
            "The pipeline automatically:",
            "[OK] Resolved task dependencies",
            "[OK] Executed tasks in optimal order",
            "[OK] Passed data between tasks",
            "[OK] Applied transformations and conditions",
        ],
        newline_before=True,
    )


if __name__ == "__main__":
    main()
