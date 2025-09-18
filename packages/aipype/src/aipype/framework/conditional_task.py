"""ConditionalTask - Task that executes based on context conditions."""

from typing import Any, Callable, Dict, List, Optional

from typing import override
from .base_task import BaseTask
from .task_result import TaskResult
from .task_dependencies import TaskDependency
from .task_context import TaskContext


class ConditionalTask(BaseTask):
    """Task that executes actions based on conditional logic using context data."""

    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[TaskDependency]] = None,
    ):
        """Initialize conditional task.

        Args:
            name: Task name
            config: Task configuration
            dependencies: List of task dependencies

        Config parameters:
        - condition_function: Function that evaluates to True/False
        - condition_inputs: List of input field names for condition function
        - action_function: Function to execute when condition is True
        - action_inputs: List of input field names for action function
        - else_function: Optional function to execute when condition is False
        - else_inputs: List of input field names for else function
        - skip_reason: Custom reason message when condition is False
        """
        super().__init__(name, config, dependencies)
        self.validation_rules = {
            "required": ["condition_function"],
            "defaults": {
                "condition_inputs": [],
                "action_inputs": [],
                "else_inputs": [],
                "skip_reason": "Condition evaluated to False",
            },
            "types": {
                "condition_inputs": list,
                "action_inputs": list,
                "else_inputs": list,
                "skip_reason": str,
            },
            # Validation lambdas intentionally use dynamic typing for flexibility across field types
            "custom": {
                "condition_function": lambda x: callable(x),  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
                "action_function": lambda x: x is None or callable(x),  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
                "else_function": lambda x: x is None or callable(x),  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
            },
        }
        self.context_instance: Optional[TaskContext] = None

    @override
    def set_context(self, context: TaskContext) -> None:
        """Set the task context.

        Args:
            context: TaskContext instance
        """
        self.context_instance = context

    @override
    def run(self) -> TaskResult:
        """Execute the conditional task."""
        from datetime import datetime

        start_time = datetime.now()

        # Validate configuration using instance validation rules
        validation_failure = self._validate_or_fail(start_time)
        if validation_failure:
            return validation_failure

        # Validation has already ensured condition_function is callable
        condition_func = self.config.get("condition_function")
        assert callable(condition_func), (
            "condition_function should be callable after validation"
        )

        # Evaluate condition
        try:
            condition_result = self._evaluate_condition(condition_func)
        except Exception as e:
            error_msg = f"ConditionalTask execution failed: {str(e)}"
            self.logger.error(error_msg)
            return TaskResult.failure(
                error_message=error_msg,
                metadata={
                    "task_name": self.name,
                    "error_type": type(e).__name__,
                },
            )

        result: Dict[str, Any] = {
            "condition_result": condition_result,
            "executed": False,
            "action_result": None,
            "else_result": None,
            "skip_reason": None,
        }

        if condition_result:
            # Execute main action
            action_func = self.config.get("action_function")
            if action_func:
                try:
                    action_result = self._execute_action(action_func, "action_inputs")
                    result["executed"] = True
                    result["action_result"] = action_result
                except Exception as e:
                    error_msg = f"ConditionalTask action execution failed: {str(e)}"
                    self.logger.error(error_msg)
                    return TaskResult.failure(
                        error_message=error_msg,
                        metadata={
                            "task_name": self.name,
                            "error_type": type(e).__name__,
                        },
                    )

                self.logger.info(
                    f"Condition true for task '{self.name}' - action executed"
                )
            else:
                self.logger.warning(
                    f"Condition true for task '{self.name}' but no action_function specified"
                )

        else:
            # Execute else action if provided
            else_func = self.config.get("else_function")
            if else_func:
                try:
                    else_result = self._execute_action(else_func, "else_inputs")
                    result["executed"] = True
                    result["else_result"] = else_result
                except Exception as e:
                    error_msg = (
                        f"ConditionalTask else action execution failed: {str(e)}"
                    )
                    self.logger.error(error_msg)
                    return TaskResult.failure(
                        error_message=error_msg,
                        metadata={
                            "task_name": self.name,
                            "error_type": type(e).__name__,
                        },
                    )

                self.logger.info(
                    f"Condition false for task '{self.name}' - else action executed"
                )
            else:
                # No else action, task is skipped
                skip_reason = self.config.get(
                    "skip_reason", "Condition evaluated to False"
                )
                result["skip_reason"] = skip_reason

                self.logger.info(
                    f"Condition false for task '{self.name}' - task skipped: {skip_reason}"
                )

        # Wrap result in TaskResult for standardized response format
        return TaskResult.success(
            data=result,
            execution_time=0.0,  # ConditionalTask is typically fast
            metadata={
                "task_name": self.name,
                "condition_result": result["condition_result"],
                "executed": result["executed"],
            },
        )

    def _evaluate_condition(self, condition_func: Callable[..., Any]) -> bool:
        """Evaluate the condition function.

        Args:
            condition_func: Function to evaluate

        Returns:
            Boolean result of condition evaluation
        """
        try:
            # Get condition inputs
            condition_inputs = self.config.get("condition_inputs", [])

            if not condition_inputs:
                # No inputs required - call function with no arguments
                return bool(condition_func())

            elif len(condition_inputs) == 1:
                # Single input
                input_name = condition_inputs[0]
                if input_name not in self.config:
                    raise ValueError(
                        f"Condition input '{input_name}' not found in resolved config"
                    )

                input_value = self.config[input_name]
                return bool(condition_func(input_value))

            else:
                # Multiple inputs - pass as keyword arguments
                input_values = {}
                for input_name in condition_inputs:
                    if input_name not in self.config:
                        raise ValueError(
                            f"Condition input '{input_name}' not found in resolved config"
                        )
                    input_values[input_name] = self.config[input_name]

                return bool(condition_func(**input_values))

        except Exception as e:
            error_msg = f"ConditionalTask condition evaluation operation failed: Condition evaluation failed: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _execute_action(self, action_func: Callable[..., Any], inputs_key: str) -> Any:
        """Execute an action function.

        Args:
            action_func: Function to execute
            inputs_key: Config key for input field names

        Returns:
            Result of action execution
        """
        try:
            # Get action inputs
            action_inputs = self.config.get(inputs_key, [])

            if not action_inputs:
                # No inputs required - call function with no arguments
                return action_func()

            elif len(action_inputs) == 1:
                # Single input
                input_name = action_inputs[0]
                if input_name not in self.config:
                    raise ValueError(
                        f"Action input '{input_name}' not found in resolved config"
                    )

                input_value = self.config[input_name]
                return action_func(input_value)

            else:
                # Multiple inputs - pass as keyword arguments
                input_values = {}
                for input_name in action_inputs:
                    if input_name not in self.config:
                        raise ValueError(
                            f"Action input '{input_name}' not found in resolved config"
                        )
                    input_values[input_name] = self.config[input_name]

                return action_func(**input_values)

        except Exception as e:
            error_msg = f"ConditionalTask action execution operation failed: Action execution failed: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def preview_condition(self) -> Dict[str, Any]:
        """Preview what the condition would evaluate to with current input.

        Returns:
            Preview information about the condition
        """
        try:
            condition_func = self.config.get("condition_function")
            if not condition_func:
                return {"error": "No condition_function specified"}

            # Try to evaluate condition
            condition_result = self._evaluate_condition(condition_func)

            preview: Dict[str, Any] = {
                "task_name": self.name,
                "condition_result": condition_result,
                "condition_inputs": self.config.get("condition_inputs", []),
                "would_execute": "action"
                if condition_result
                else ("else" if self.config.get("else_function") else "skip"),
                "skip_reason": self.config.get(
                    "skip_reason", "Condition evaluated to False"
                )
                if not condition_result
                else None,
            }

            # Add input values for reference
            condition_inputs = self.config.get("condition_inputs", [])
            if condition_inputs:
                preview["input_values"] = {}
                for input_name in condition_inputs:
                    if input_name in self.config:
                        value = self.config[input_name]
                        preview["input_values"][input_name] = {
                            "type": type(value).__name__,
                            "value": str(value)[:100] + "..."
                            if len(str(value)) > 100
                            else str(value),
                        }

            return preview

        except Exception as e:
            return {
                "task_name": self.name,
                "error": f"Condition preview failed: {str(e)}",
            }

    def get_execution_summary(self, context: TaskContext) -> Optional[Dict[str, Any]]:
        """Get a formatted summary of the conditional task execution results.

        Args:
            context: TaskContext to retrieve results from

        Returns:
            Dictionary containing execution summary with formatted info, or None if no results found
        """
        # Get task results using ultra-safe convenience methods
        condition_result, action_result, else_result = context.get_result_fields(
            self.name, "condition_result", "action_result", "else_result"
        )
        condition_result = condition_result or False

        if condition_result is None and action_result is None and else_result is None:
            return None

        # Prepare quality check summary
        quality_info = [f"Condition Met: {'Yes' if condition_result else 'No'}"]

        # Show which execution path was taken
        if condition_result:
            quality_info.append("Execution Path: main action (condition passed)")
            quality_info.append(
                f"Main Action Executed: {'Yes' if action_result else 'No'}"
            )
            result_data = action_result
        else:
            quality_info.append("Execution Path: else action (condition failed)")
            quality_info.append(
                f"Else Action Executed: {'Yes' if else_result else 'No'}"
            )
            result_data = else_result

        # Extract structured data from result if available
        if isinstance(result_data, dict):
            # Dynamic data structure requires runtime type checking
            status: str = str(result_data.get("status", "unknown"))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            message: str = str(result_data.get("message", "No message"))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            action_taken: str = str(result_data.get("action_taken", "none"))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]

            quality_info.extend(
                [
                    f"Status: {status.upper()}",
                    f"Action: {action_taken}",
                    f"Message: {message}",
                ]
            )

        # Add skip reason if applicable
        skip_reason = context.get_result_field(self.name, "skip_reason", str)
        if skip_reason:
            quality_info.append(f"Skip Reason: {skip_reason}")

        return {
            "condition_result": condition_result,
            "action_result": action_result,
            "else_result": else_result,
            "result_data": result_data,
            "skip_reason": skip_reason,
            "quality_info": quality_info,
        }

    @override
    def __str__(self) -> str:
        """String representation of the conditional task."""
        dep_count = len(self.dependencies)
        has_condition = bool(self.config.get("condition_function"))
        has_action = bool(self.config.get("action_function"))
        has_else = bool(self.config.get("else_function"))

        return f"ConditionalTask(name='{self.name}', dependencies={dep_count}, condition={has_condition}, action={has_action}, else={has_else})"


# Common condition functions


def threshold_condition(
    threshold: float, operator: str = ">="
) -> Callable[[float], bool]:
    """Create a condition function that checks if a value meets a threshold.

    Args:
        threshold: Threshold value to compare against
        operator: Comparison operator (>=, >, <=, <, ==, !=)

    Returns:
        Condition function
    """

    def condition(value: float) -> bool:
        if not isinstance(value, (int, float)):  # pyright: ignore
            raise ValueError("Value must be numeric")

        if operator == ">=":
            return value >= threshold
        elif operator == ">":
            return value > threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "<":
            return value < threshold
        elif operator == "==":
            return value == threshold
        elif operator == "!=":
            return value != threshold
        else:
            raise ValueError(f"Unknown operator: {operator}")

    return condition


def contains_condition(
    search_term: str, case_sensitive: bool = False
) -> Callable[[str], bool]:
    """Create a condition function that checks if text contains a term.

    Args:
        search_term: Term to search for
        case_sensitive: Whether search is case sensitive

    Returns:
        Condition function
    """

    def condition(text: str) -> bool:
        if not isinstance(text, str):  # pyright: ignore
            raise ValueError("Input must be a string")

        if case_sensitive:
            return search_term in text
        else:
            return search_term.lower() in text.lower()

    return condition


def list_size_condition(
    min_size: Optional[int] = None, max_size: Optional[int] = None
) -> Callable[[List[Any]], bool]:
    """Create a condition function that checks list size.

    Args:
        min_size: Minimum required size (inclusive)
        max_size: Maximum allowed size (inclusive)

    Returns:
        Condition function
    """

    def condition(items: List[Any]) -> bool:
        if not isinstance(items, list):  # pyright: ignore
            raise ValueError("Input must be a list")

        size = len(items)

        if min_size is not None and size < min_size:
            return False

        if max_size is not None and size > max_size:
            return False

        return True

    return condition


def success_rate_condition(min_success_rate: float) -> Callable[[Dict[str, Any]], bool]:
    """Create a condition function that checks success rate from a results dict.

    Args:
        min_success_rate: Minimum success rate (0.0 to 1.0)

    Returns:
        Condition function
    """

    def condition(results: Dict[str, Any]) -> bool:
        if not isinstance(results, dict):  # pyright: ignore
            raise ValueError("Input must be a dictionary")

        successful: Any = results.get(
            "successful_fetches", results.get("successful", 0)
        )
        total = results.get(
            "total_urls",
            results.get(
                "total",
                successful + results.get("failed_fetches", results.get("failed", 0)),
            ),
        )

        if total == 0:
            return False

        success_rate = successful / total
        return bool(success_rate >= min_success_rate)

    return condition


def quality_gate_condition(
    min_score: float, score_field: str = "score"
) -> Callable[[Dict[str, Any]], bool]:
    """Create a condition function that checks quality score.

    Args:
        min_score: Minimum quality score
        score_field: Field name containing the score

    Returns:
        Condition function
    """

    def condition(data: Dict[str, Any]) -> bool:
        if not isinstance(data, dict):  # pyright: ignore
            raise ValueError("Input must be a dictionary")

        if score_field not in data:
            raise ValueError(f"Score field '{score_field}' not found in data")

        score = data[score_field]
        if not isinstance(score, (int, float)):
            raise ValueError(f"Score must be numeric, got {type(score)}")

        return score >= min_score

    return condition


# Common action functions


def log_action(message: str, level: str = "info") -> Callable[[], Dict[str, Any]]:
    """Create an action function that logs a message.

    Args:
        message: Message to log
        level: Log level (info, warning, error)

    Returns:
        Action function
    """

    def action() -> Dict[str, Any]:
        import logging

        logger = logging.getLogger("conditional_task")

        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)

        return {"action": "log", "message": message, "level": level}

    return action


def increment_counter_action(
    counter_name: str = "counter",
) -> Callable[[TaskContext], Dict[str, Any]]:
    """Create an action function that increments a counter in context.

    Args:
        counter_name: Name of the counter in context

    Returns:
        Action function
    """

    def action(context: TaskContext) -> Dict[str, Any]:
        current_result = context.get_result(counter_name)
        current_value: int = current_result.get("value", 0) if current_result else 0
        new_value: int = current_value + 1
        context.store_result(counter_name, {"value": new_value})

        return {"action": "increment", "counter": counter_name, "new_value": new_value}

    return action


def set_flag_action(
    flag_name: str, flag_value: Any = True
) -> Callable[[], Dict[str, Any]]:
    """Create an action function that sets a flag value.

    Args:
        flag_name: Name of the flag
        flag_value: Value to set

    Returns:
        Action function
    """

    def action() -> Dict[str, Any]:
        return {"action": "set_flag", "flag": flag_name, "value": flag_value}

    return action
