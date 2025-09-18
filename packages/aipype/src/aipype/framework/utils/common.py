"""Common utilities for mi-agents."""

from typing import Any, Dict, List, Optional
import logging
from datetime import datetime


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with the specified name and level."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def timestamp() -> str:
    """Return current timestamp as ISO format string."""
    return datetime.now().isoformat()


def safe_dict_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary."""
    return data.get(key, default)


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """Flatten a list of lists into a single list."""
    return [item for sublist in nested_list for item in sublist]


def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> bool:
    """Validate that all required fields are present in the data."""
    return all(field in data and data[field] is not None for field in required_fields)


def validate_task_config(
    task_name: str, config: Dict[str, Any], validation_rules: Dict[str, Any]
) -> Optional[str]:
    """Simple configuration validation that returns error message or None.

    Args:
        task_name: Name of the task (for error messages)
        config: Configuration dictionary to validate
        validation_rules: Dictionary defining validation rules

    Returns:
        Error message string if validation fails, None if valid

    Example:
        rules = {
            "required": ["llm_provider", "llm_model"],
            "defaults": {"temperature": 0.7, "max_tokens": 1000},
            "types": {"temperature": (int, float), "max_tokens": int},
            "ranges": {"temperature": (0, 2), "max_tokens": (1, 32000)},
            "custom": {"llm_provider": lambda x: x.strip() != ""}
        }
        error = validate_task_config("my_task", config, rules)
        if error:
            return TaskResult.failure(error_message=error)
    """
    # Check required fields
    if "required" in validation_rules:
        missing = [
            field
            for field in validation_rules["required"]
            if field not in config or config[field] is None
        ]
        if missing:
            return f"{task_name} validation failed: Missing required fields: {missing}"

    # Apply defaults (modifies config in-place)
    if "defaults" in validation_rules:
        for field, default_value in validation_rules["defaults"].items():
            if field not in config:
                config[field] = default_value

    # Check types
    if "types" in validation_rules:
        for field, expected_types in validation_rules["types"].items():
            if field in config and config[field] is not None:
                if not isinstance(config[field], expected_types):
                    return f"{task_name} validation failed: {field} must be of type {expected_types}"

    # Check ranges for numeric values
    if "ranges" in validation_rules:
        for field, (min_val, max_val) in validation_rules["ranges"].items():
            if field in config and config[field] is not None:
                value = config[field]
                # Handle None bounds (None means no limit)
                min_ok = min_val is None or min_val <= value
                max_ok = max_val is None or value <= max_val
                if not (min_ok and max_ok):
                    min_str = str(min_val) if min_val is not None else "no minimum"
                    max_str = str(max_val) if max_val is not None else "no maximum"
                    return f"{task_name} validation failed: {field} must be between {min_str} and {max_str}"

    # Check custom validators
    if "custom" in validation_rules:
        for field, validator_func in validation_rules["custom"].items():
            if field in config and config[field] is not None:
                try:
                    if not validator_func(config[field]):
                        return f"{task_name} validation failed: {field} failed custom validation"
                except Exception as e:
                    return f"{task_name} validation failed: {field} validation error: {str(e)}"

    return None  # All validations passed
