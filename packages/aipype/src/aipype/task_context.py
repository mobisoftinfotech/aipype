"""TaskContext - Simplified shared data store for inter-task communication."""

import threading
from typing import Dict, List, Optional, Any, TypeVar, Type

from typing import override
from datetime import datetime
from .utils.common import setup_logger
from .utils.display import print_message_box

T = TypeVar("T")


class TaskContext:
    """Simplified context for storing and accessing task results across pipeline execution.

    Focuses on string-based data storage and simple dot notation path access.
    Designed for text-based automation workflows.
    """

    def __init__(self) -> None:
        """Initialize empty task context."""
        self.logger = setup_logger("task_context")
        self._results: Dict[str, Dict[str, Any]] = {}
        self._execution_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()  # Reentrant lock for thread safety

    def store_result(self, task_name: str, result: Dict[str, Any]) -> None:
        """Store a task result in the context.

        Args:
            task_name: Name of the task that produced the result
            result: The result data to store (dictionary with string keys)
        """
        with self._lock:
            self._results[task_name] = result
            self.logger.debug(f"Stored result for task '{task_name}'")

    def get_result(self, task_name: str) -> Optional[Dict[str, Any]]:
        """Get a task result from the context.

        Args:
            task_name: Name of the task to get result for

        Returns:
            The stored result dictionary or None if not found
        """
        with self._lock:
            return self._results.get(task_name)

    def has_result(self, task_name: str) -> bool:
        """Check if a task result exists in the context.

        Args:
            task_name: Name of the task to check

        Returns:
            True if result exists, False otherwise
        """
        with self._lock:
            return task_name in self._results

    def get_path_value(self, path: str) -> Optional[Any]:
        """Get a value from context using paths like 'task.field', 'task.array[0]', or 'task.array[].field'.

        Args:
            path: The path to resolve (e.g., 'search.query', 'search.results[0].title', 'search.results[].url')

        Returns:
            The resolved value or None if path cannot be resolved
        """
        try:
            # Validate path format
            if not path or "." not in path:
                self.logger.warning(
                    f"Invalid path format: '{path}' - must be 'task.field'"
                )
                return None

            # Split path into task name and remaining path
            task_name, remaining_path = path.split(".", 1)
            if not task_name or not remaining_path:
                self.logger.warning(
                    f"Invalid path format: '{path}' - must be 'task.field'"
                )
                return None

            # Get the task result
            task_result = self.get_result(task_name)
            if task_result is None:
                self.logger.debug(f"Task '{task_name}' not found in context")
                return None

            # Navigate through the remaining path
            return self._navigate_path(task_result, remaining_path.split("."))

        except Exception as e:
            self.logger.warning(f"Failed to resolve path '{path}': {str(e)}")
            return None

    def _navigate_path(
        self, current_value: Any, path_parts: List[str]
    ) -> Optional[Any]:
        """Navigate through path parts to resolve the final value."""
        for part in path_parts:
            if current_value is None:
                return None

            # Handle array extraction like 'results[]'
            if part.endswith("[]"):
                field_name = part[:-2]  # Remove '[]'
                if isinstance(current_value, dict) and field_name in current_value:
                    array_data: Any = current_value[field_name]  # type: ignore
                    if isinstance(array_data, list):
                        # This is array extraction - we expect more path parts to specify what to extract
                        if len(path_parts) > path_parts.index(part) + 1:
                            next_part = path_parts[path_parts.index(part) + 1]
                            # Extract the specified field from each item in the array
                            extracted_values: List[Any] = []
                            for item in array_data:  # type: ignore
                                if isinstance(item, dict) and next_part in item:
                                    extracted_values.append(item[next_part])
                            return extracted_values
                        else:
                            return array_data  # type: ignore
                    else:
                        return None
                else:
                    return None

            # Handle array index like 'results[0]'
            elif "[" in part and "]" in part and not part.endswith("[]"):
                try:
                    field_name = part[: part.index("[")]
                    index_str = part[part.index("[") + 1 : part.index("]")]
                    index = int(index_str)

                    if isinstance(current_value, dict) and field_name in current_value:
                        indexed_array: Any = current_value[field_name]  # type: ignore
                        if isinstance(indexed_array, list) and 0 <= index < len(
                            indexed_array  # type: ignore
                        ):
                            current_value = indexed_array[index]  # type: ignore
                        else:
                            return None
                    else:
                        return None
                except (ValueError, IndexError):
                    return None

            # Handle regular field access
            else:
                if isinstance(current_value, dict) and part in current_value:
                    current_value = current_value[part]  # type: ignore
                elif isinstance(current_value, list):
                    # If we have a list and the next part is a field name,
                    # extract that field from each item in the list
                    field_values: List[Any] = []
                    for item in current_value:  # type: ignore
                        if isinstance(item, dict) and part in item:
                            field_values.append(item[part])

                    if field_values:
                        current_value = field_values
                    else:
                        return None
                else:
                    return None

        return current_value  # type: ignore

    def set_data(self, task_name: str, data: Dict[str, Any]) -> None:
        """Set data for a task (alias for store_result for compatibility).

        Args:
            task_name: Name of the task
            data: Data dictionary to store
        """
        self.store_result(task_name, data)

    def get_data(self, task_name: str) -> Optional[Dict[str, Any]]:
        """Get data for a task (alias for get_result for compatibility).

        Args:
            task_name: Name of the task

        Returns:
            Task data dictionary or None if not found
        """
        return self.get_result(task_name)

    def record_task_started(self, task_name: str) -> None:
        """Record that a task has started execution.

        Args:
            task_name: Name of the task that started
        """
        with self._lock:
            entry: Dict[str, Any] = {
                "task_name": task_name,
                "status": "started",
                "start_time": datetime.now(),
                "end_time": None,
                "result": None,
                "error": None,
            }

            # Remove any existing entry for this task
            self._execution_history = [
                h for h in self._execution_history if h["task_name"] != task_name
            ]
            self._execution_history.append(entry)

            self.logger.debug(f"Recorded task '{task_name}' as started")

    def record_task_completed(self, task_name: str, result: Dict[str, Any]) -> None:
        """Record that a task has completed successfully.

        Args:
            task_name: Name of the task that completed
            result: The result produced by the task
        """
        with self._lock:
            # Find the existing entry
            for entry in self._execution_history:
                if entry["task_name"] == task_name:
                    entry["status"] = "completed"
                    entry["end_time"] = datetime.now()
                    entry["result"] = result
                    entry["error"] = None
                    break
            else:
                # No existing entry, create one
                new_entry: Dict[str, Any] = {
                    "task_name": task_name,
                    "status": "completed",
                    "start_time": datetime.now(),
                    "end_time": datetime.now(),
                    "result": result,
                    "error": None,
                }
                self._execution_history.append(new_entry)

            self.logger.debug(f"Recorded task '{task_name}' as completed")

    def record_task_failed(self, task_name: str, error: str) -> None:
        """Record that a task has failed.

        Args:
            task_name: Name of the task that failed
            error: Error message describing the failure
        """
        with self._lock:
            # Find the existing entry
            for entry in self._execution_history:
                if entry["task_name"] == task_name:
                    entry["status"] = "failed"
                    entry["end_time"] = datetime.now()
                    entry["result"] = None
                    entry["error"] = error
                    break
            else:
                # No existing entry, create one
                failed_entry: Dict[str, Any] = {
                    "task_name": task_name,
                    "status": "failed",
                    "start_time": datetime.now(),
                    "end_time": datetime.now(),
                    "result": None,
                    "error": error,
                }
                self._execution_history.append(failed_entry)

            self.logger.debug(f"Recorded task '{task_name}' as failed: {error}")

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get the execution history of all tasks.

        Returns:
            List of execution history entries
        """
        with self._lock:
            return self._execution_history.copy()

    def get_completed_tasks(self) -> List[str]:
        """Get names of all completed tasks.

        Returns:
            List of task names that completed successfully
        """
        with self._lock:
            return [
                entry["task_name"]
                for entry in self._execution_history
                if entry["status"] == "completed"
            ]

    def get_failed_tasks(self) -> List[str]:
        """Get names of all failed tasks.

        Returns:
            List of task names that failed
        """
        with self._lock:
            return [
                entry["task_name"]
                for entry in self._execution_history
                if entry["status"] == "failed"
            ]

    def clear(self) -> None:
        """Clear all stored results and execution history."""
        with self._lock:
            self._results.clear()
            self._execution_history.clear()
            self.logger.debug("Cleared all context data")

    def get_all_results(self) -> Dict[str, Dict[str, Any]]:
        """Get all stored task results.

        Returns:
            Dictionary mapping task names to their result dictionaries
        """
        with self._lock:
            return self._results.copy()

    def get_task_count(self) -> int:
        """Get the number of tasks with stored results.

        Returns:
            Number of tasks with results
        """
        with self._lock:
            return len(self._results)

    @override
    def __str__(self) -> str:
        """String representation of the context."""
        with self._lock:
            completed = len(self.get_completed_tasks())
            failed = len(self.get_failed_tasks())
            total_results = len(self._results)

            return f"TaskContext(results={total_results}, completed={completed}, failed={failed})"

    def get_result_field(
        self,
        task_name: str,
        field_name: str,
        expected_type: Type[T] = str,
        default: Optional[T] = None,
    ) -> Optional[T]:
        """Get a specific field from a task result with type checking.

        Ultra-safe: handles all edge cases internally, never throws errors.

        Args:
            task_name: Name of the task to get result from
            field_name: Name of the field to extract
            expected_type: Expected type of the field value (default: str)
            default: Default value to return if field not found or wrong type

        Returns:
            The field value if found and correct type, otherwise default
        """
        try:
            with self._lock:
                task_result = self.get_result(task_name)
                if not isinstance(task_result, dict):
                    return default

                field_value = task_result.get(field_name)
                if isinstance(field_value, expected_type):
                    return field_value

                return default
        except Exception:
            # Ultra-safe: never throw errors from convenience methods
            return default

    def get_result_content(self, task_name: str) -> Optional[str]:
        """Get the 'content' field from a task result as a string.

        Ultra-safe: handles all edge cases internally, never throws errors.

        Args:
            task_name: Name of the task to get content from

        Returns:
            The content string if found and valid, otherwise None
        """
        return self.get_result_field(task_name, "content", str)

    def has_result_content(self, task_name: str) -> bool:
        """Check if a task has non-empty content.

        Ultra-safe: handles all edge cases internally, never throws errors.

        Args:
            task_name: Name of the task to check

        Returns:
            True if task has non-empty content field, False otherwise
        """
        try:
            content = self.get_result_content(task_name)
            return content is not None and bool(content.strip())
        except Exception:
            # Ultra-safe: never throw errors from convenience methods
            return False

    def display_result_content(
        self, task_name: str, title: str, field_name: str = "content"
    ) -> None:
        """Display task result content in a message box if present.

        Ultra-safe: handles all edge cases internally including:
        - Task not completed
        - Missing or empty content
        - Any errors during display

        No return value needed - this is a pure convenience method.

        Args:
            task_name: Name of the task to display content from
            title: Title for the message box
            field_name: Name of the field to display (default: "content")
        """
        try:
            # Check if task completed successfully
            completed_tasks = self.get_completed_tasks()
            if task_name not in completed_tasks:
                return

            content = self.get_result_field(task_name, field_name, str)
            if content is not None and bool(content.strip()):
                print_message_box(title, [content])
        except Exception:
            # Ultra-safe: never throw errors from convenience methods
            pass

    def get_result_fields(self, task_name: str, *field_names: str) -> tuple[Any, ...]:
        """Get multiple fields from a task result at once.

        Ultra-safe: handles all edge cases internally, never throws errors.

        Args:
            task_name: Name of the task to get fields from
            *field_names: Names of the fields to extract

        Returns:
            Tuple of field values in the same order as field_names
        """
        try:
            with self._lock:
                task_result = self.get_result(task_name)
                if not isinstance(task_result, dict):
                    return tuple(None for _ in field_names)

                return tuple(task_result.get(field_name) for field_name in field_names)
        except Exception:
            # Ultra-safe: never throw errors from convenience methods
            return tuple(None for _ in field_names)

    def display_completed_results(
        self, task_content_pairs: List[tuple[str, str]]
    ) -> None:
        """Display content for multiple completed tasks in one call.

        Ultra-safe: handles all edge cases internally, only displays content
        for tasks that completed successfully and have valid content.

        Args:
            task_content_pairs: List of (task_name, title) pairs to display
        """
        try:
            for task_name, title in task_content_pairs:
                self.display_result_content(task_name, title)
        except Exception:
            # Ultra-safe: never throw errors from convenience methods
            pass

    @override
    def __repr__(self) -> str:
        """Detailed representation of the context."""
        return self.__str__()
