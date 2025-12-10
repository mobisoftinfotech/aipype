"""Example agent implementation using modern declarative @task syntax."""

from typing import Annotated, Any, Dict, List

from aipype import (
    PipelineAgent,
    task,
    Depends,
    print_message_box,
)


class ExampleAgent(PipelineAgent):
    """Example agent demonstrating the declarative @task syntax.

    This agent shows:
    - Tasks defined with @task decorator
    - Automatic dependency inference from parameter names
    - Data transformation between tasks
    - Conditional logic using Python if/else
    """

    @task
    def welcome(self) -> Dict[str, str]:
        """Display welcome message - no dependencies, runs first."""
        message = "[START] Starting Modern Example Agent!"
        print(message)
        return {"message": message}

    @task
    def prepare_calculation(self, welcome: Dict[str, str]) -> Dict[str, Any]:
        """Prepare calculation configuration.

        Depends on welcome task (via parameter name) to ensure ordering.
        """
        print("Setting up calculation numbers...")
        return {
            "operation": "add",
            "numbers": [10, 20, 30],
        }

    @task
    def add_numbers(
        self,
        prepare_calculation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Perform the calculation based on prepared config.

        Demonstrates accessing dependency data directly.
        """
        operation = prepare_calculation.get("operation", "add")
        numbers: List[int] = prepare_calculation.get("numbers", [1, 2, 3])

        if operation == "add":
            result = sum(numbers)
        elif operation == "multiply":
            result = 1
            for num in numbers:
                result *= num
        else:
            raise ValueError(f"Unknown operation: {operation}")

        print(f"Calculated {operation} of {numbers} = {result}")
        return {"result": result, "operation": operation, "numbers": numbers}

    @task
    def double_result(
        self,
        add_numbers: Annotated[int, Depends("add_numbers.result")],
    ) -> Dict[str, int]:
        """Double the calculation result.

        Demonstrates Depends() for extracting specific fields.
        """
        doubled = add_numbers * 2
        print(f"Doubled result: {add_numbers} * 2 = {doubled}")
        return {"doubled_value": doubled}

    @task
    def check_result(
        self,
        double_result: Annotated[int, Depends("double_result.doubled_value")],
    ) -> Dict[str, Any]:
        """Check if the result meets criteria using Python conditionals.

        Demonstrates replacing ConditionalTask with simple Python logic.
        """
        if double_result > 100:
            message = "Result is large! [SUCCESS]"
        else:
            message = "Result is small. [STATS]"

        print(message)
        return {"message": message, "is_large": double_result > 100}

    @task
    def farewell(self, check_result: Dict[str, Any]) -> Dict[str, str]:
        """Display farewell message - runs last due to dependency chain."""
        message = "[SUCCESS] Modern Example Agent completed successfully!"
        print(message)
        return {"message": message}


def main() -> None:
    """Demonstrate the modern pipeline agent with @task decorators."""
    print_message_box(
        "[DEMO] MODERN EXAMPLE AGENT DEMONSTRATION",
        [
            "This example shows the declarative @task syntax where:",
            "- Tasks are defined as methods with @task decorator",
            "- Dependencies inferred from parameter names",
            "- Depends() extracts specific fields from results",
            "- Python conditionals replace ConditionalTask",
        ],
    )

    agent = ExampleAgent(
        "modern_example_agent",
        {
            "enable_parallel": False,  # Sequential for clear demonstration
            "stop_on_failure": True,
        },
    )

    agent.run()
    agent.display_results()

    print_message_box(
        "[COMPLETE] DEMONSTRATION COMPLETE",
        [
            "The pipeline automatically:",
            "[OK] Discovered @task methods",
            "[OK] Inferred dependencies from parameters",
            "[OK] Executed tasks in dependency order",
            "[OK] Passed data between tasks",
        ],
        newline_before=True,
    )


if __name__ == "__main__":
    main()
