"""Tests for common utilities."""

import pytest
import logging
from typing import Any, Dict, List
from unittest.mock import patch
from aipype import (
    setup_logger,
    timestamp,
    safe_dict_get,
    flatten_list,
    validate_required_fields,
)


class TestSetupLogger:
    """Test logger setup functionality."""

    def test_setup_logger_default_level(self) -> None:
        """Test logger setup with default level."""
        logger = setup_logger("test_logger")

        assert logger.name == "test_logger"
        assert logger.level == logging.INFO
        assert len(logger.handlers) >= 1

    def test_setup_logger_custom_level(self) -> None:
        """Test logger setup with custom level."""
        logger = setup_logger("debug_logger", logging.DEBUG)

        assert logger.name == "debug_logger"
        assert logger.level == logging.DEBUG

    def test_setup_logger_warning_level(self) -> None:
        """Test logger setup with warning level."""
        logger = setup_logger("warning_logger", logging.WARNING)

        assert logger.name == "warning_logger"
        assert logger.level == logging.WARNING

    def test_setup_logger_error_level(self) -> None:
        """Test logger setup with error level."""
        logger = setup_logger("error_logger", logging.ERROR)

        assert logger.name == "error_logger"
        assert logger.level == logging.ERROR

    def test_setup_logger_handler_format(self) -> None:
        """Test logger handler has correct formatter."""
        logger = setup_logger("format_test_logger")

        # Should have at least one handler
        assert len(logger.handlers) >= 1

        # Check the first handler has a formatter
        handler = logger.handlers[0]
        assert handler.formatter is not None

        # Test the format includes expected components
        formatter = handler.formatter
        assert formatter is not None
        assert formatter._fmt is not None
        assert "%(asctime)s" in formatter._fmt
        assert "%(name)s" in formatter._fmt
        assert "%(levelname)s" in formatter._fmt
        assert "%(message)s" in formatter._fmt

    def test_setup_logger_no_duplicate_handlers(self) -> None:
        """Test that calling setup_logger twice doesn't create duplicate handlers."""
        logger_name = "no_duplicate_test"

        # Setup logger first time
        logger1 = setup_logger(logger_name)
        handler_count_1 = len(logger1.handlers)

        # Setup logger second time
        logger2 = setup_logger(logger_name)
        handler_count_2 = len(logger2.handlers)

        # Should be the same logger instance with same handler count
        assert logger1 is logger2
        assert handler_count_1 == handler_count_2

    def test_setup_logger_stream_handler(self) -> None:
        """Test that logger uses StreamHandler."""
        logger = setup_logger("stream_test_logger")

        # Should have at least one StreamHandler
        stream_handlers: List[logging.StreamHandler[Any]] = [
            h for h in logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert len(stream_handlers) >= 1


class TestTimestamp:
    """Test timestamp utility function."""

    def test_timestamp_format(self) -> None:
        """Test timestamp returns ISO format string."""
        ts = timestamp()

        # Should be a string
        assert isinstance(ts, str)

        # Should contain expected ISO format components
        assert "T" in ts  # ISO format separator
        assert "-" in ts  # Date separators
        assert ":" in ts  # Time separators

    def test_timestamp_valid_datetime(self) -> None:
        """Test timestamp can be parsed back to datetime."""
        from datetime import datetime

        ts = timestamp()

        # Should be parseable back to datetime (may have microseconds)
        try:
            parsed = datetime.fromisoformat(ts)
            assert isinstance(parsed, datetime)
        except ValueError:
            # Try without microseconds if parsing fails
            ts_no_micro = ts.split(".")[0] if "." in ts else ts
            parsed = datetime.fromisoformat(ts_no_micro)
            assert isinstance(parsed, datetime)

    def test_timestamp_uniqueness(self) -> None:
        """Test that consecutive timestamps are different."""
        ts1 = timestamp()

        # Small delay to ensure different timestamps
        import time

        time.sleep(0.001)

        ts2 = timestamp()

        assert ts1 != ts2

    def test_timestamp_no_parameters(self) -> None:
        """Test timestamp function doesn't accept parameters."""
        # Function should work with no arguments
        ts = timestamp()
        assert isinstance(ts, str)

        # Should not accept any arguments (this would raise TypeError)
        with pytest.raises(TypeError):
            timestamp("invalid_arg")  # pyright: ignore


class TestSafeDictGet:
    """Test safe dictionary get utility."""

    def test_safe_dict_get_existing_key(self) -> None:
        """Test getting existing key from dictionary."""
        data = {"key1": "value1", "key2": 42, "key3": None}

        assert safe_dict_get(data, "key1") == "value1"
        assert safe_dict_get(data, "key2") == 42
        assert safe_dict_get(data, "key3") is None

    def test_safe_dict_get_missing_key_default_none(self) -> None:
        """Test getting missing key returns None by default."""
        data = {"existing": "value"}

        assert safe_dict_get(data, "missing") is None

    def test_safe_dict_get_missing_key_custom_default(self) -> None:
        """Test getting missing key returns custom default."""
        data = {"existing": "value"}

        assert safe_dict_get(data, "missing", "default") == "default"
        assert safe_dict_get(data, "missing", 0) == 0
        assert safe_dict_get(data, "missing", []) == []
        assert safe_dict_get(data, "missing", {}) == {}

    def test_safe_dict_get_empty_dict(self) -> None:
        """Test safe_dict_get with empty dictionary."""
        data: Dict[str, Any] = {}

        assert safe_dict_get(data, "any_key") is None
        assert safe_dict_get(data, "any_key", "default") == "default"

    def test_safe_dict_get_none_value_vs_missing_key(self) -> None:
        """Test distinction between None value and missing key."""
        data = {"none_value": None}

        # Key exists but has None value
        assert safe_dict_get(data, "none_value") is None
        assert safe_dict_get(data, "none_value", "default") is None

        # Key doesn't exist
        assert safe_dict_get(data, "missing_key") is None
        assert safe_dict_get(data, "missing_key", "default") == "default"

    def test_safe_dict_get_complex_values(self) -> None:
        """Test safe_dict_get with complex data types."""
        complex_data = {
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "tuple": (1, 2, 3),
            "bool": True,
        }

        assert safe_dict_get(complex_data, "list") == [1, 2, 3]
        assert safe_dict_get(complex_data, "dict") == {"nested": "value"}
        assert safe_dict_get(complex_data, "tuple") == (1, 2, 3)
        assert safe_dict_get(complex_data, "bool") is True

    def test_safe_dict_get_type_consistency(self) -> None:
        """Test that safe_dict_get maintains type consistency."""
        data = {"string": "text", "number": 123}

        result = safe_dict_get(data, "string", "default")
        assert isinstance(result, str)
        assert result == "text"

        result = safe_dict_get(data, "number", 0)
        assert isinstance(result, int)
        assert result == 123


class TestFlattenList:
    """Test list flattening utility."""

    def test_flatten_list_basic(self) -> None:
        """Test basic list flattening."""
        nested = [[1, 2], [3, 4], [5]]
        flattened = flatten_list(nested)

        assert flattened == [1, 2, 3, 4, 5]

    def test_flatten_list_empty_lists(self) -> None:
        """Test flattening with empty sublists."""
        nested = [[], [1, 2], [], [3], []]
        flattened = flatten_list(nested)

        assert flattened == [1, 2, 3]

    def test_flatten_list_empty_input(self) -> None:
        """Test flattening empty list."""
        nested: List[List[Any]] = []
        flattened = flatten_list(nested)

        assert flattened == []

    def test_flatten_list_single_sublist(self) -> None:
        """Test flattening with single sublist."""
        nested = [[1, 2, 3, 4, 5]]
        flattened = flatten_list(nested)

        assert flattened == [1, 2, 3, 4, 5]

    def test_flatten_list_mixed_types(self) -> None:
        """Test flattening with mixed data types."""
        nested: List[List[Any]] = [["a", "b"], [1, 2], [True, False]]
        flattened = flatten_list(nested)

        assert flattened == ["a", "b", 1, 2, True, False]

    def test_flatten_list_preserves_order(self) -> None:
        """Test that flattening preserves order of elements."""
        nested = [["first", "second"], ["third"], ["fourth", "fifth"]]
        flattened = flatten_list(nested)

        assert flattened == ["first", "second", "third", "fourth", "fifth"]

    def test_flatten_list_with_none_values(self) -> None:
        """Test flattening with None values."""
        nested: List[List[Any]] = [[None, 1], [2, None], [None]]
        flattened = flatten_list(nested)

        assert flattened == [None, 1, 2, None, None]

    def test_flatten_list_single_elements(self) -> None:
        """Test flattening with single-element sublists."""
        nested = [[1], [2], [3], [4]]
        flattened = flatten_list(nested)

        assert flattened == [1, 2, 3, 4]

    def test_flatten_list_complex_objects(self) -> None:
        """Test flattening with complex objects."""
        obj1 = {"key": "value1"}
        obj2 = {"key": "value2"}
        nested = [[obj1], [obj2]]
        flattened = flatten_list(nested)

        assert flattened == [obj1, obj2]
        assert flattened[0]["key"] == "value1"
        assert flattened[1]["key"] == "value2"


class TestValidateRequiredFields:
    """Test required fields validation utility."""

    def test_validate_required_fields_all_present(self) -> None:
        """Test validation when all required fields are present."""
        data = {"field1": "value1", "field2": "value2", "field3": 42}
        required = ["field1", "field2", "field3"]

        assert validate_required_fields(data, required) is True

    def test_validate_required_fields_missing_field(self) -> None:
        """Test validation when required field is missing."""
        data = {"field1": "value1", "field2": "value2"}
        required = ["field1", "field2", "field3"]

        assert validate_required_fields(data, required) is False

    def test_validate_required_fields_none_value(self) -> None:
        """Test validation when required field has None value."""
        data = {"field1": "value1", "field2": None, "field3": "value3"}
        required = ["field1", "field2", "field3"]

        assert validate_required_fields(data, required) is False

    def test_validate_required_fields_empty_required_list(self) -> None:
        """Test validation with empty required fields list."""
        data = {"field1": "value1", "field2": "value2"}
        required: List[str] = []

        assert validate_required_fields(data, required) is True

    def test_validate_required_fields_empty_data(self) -> None:
        """Test validation with empty data dictionary."""
        data: Dict[str, Any] = {}
        required = ["field1"]

        assert validate_required_fields(data, required) is False

    def test_validate_required_fields_empty_data_empty_required(self) -> None:
        """Test validation with both empty data and empty required fields."""
        data: Dict[str, Any] = {}
        required: List[str] = []

        assert validate_required_fields(data, required) is True

    def test_validate_required_fields_extra_fields_ignored(self) -> None:
        """Test validation ignores extra fields not in required list."""
        data = {
            "required1": "value1",
            "required2": "value2",
            "extra1": "extra_value1",
            "extra2": "extra_value2",
        }
        required = ["required1", "required2"]

        assert validate_required_fields(data, required) is True

    def test_validate_required_fields_various_data_types(self) -> None:
        """Test validation with various data types as values."""
        data = {
            "string_field": "text",
            "number_field": 42,
            "boolean_field": True,
            "list_field": [1, 2, 3],
            "dict_field": {"nested": "value"},
        }
        required = [
            "string_field",
            "number_field",
            "boolean_field",
            "list_field",
            "dict_field",
        ]

        assert validate_required_fields(data, required) is True

    def test_validate_required_fields_zero_and_false_values(self) -> None:
        """Test validation treats 0 and False as valid (not None)."""
        data: Dict[str, Any] = {
            "zero_field": 0,
            "false_field": False,
            "empty_string": "",
            "empty_list": [],
            "empty_dict": {},
        }
        required = [
            "zero_field",
            "false_field",
            "empty_string",
            "empty_list",
            "empty_dict",
        ]

        # All these values are falsy but not None, so should be valid
        assert validate_required_fields(data, required) is True

    def test_validate_required_fields_partial_match(self) -> None:
        """Test validation with partial field matches."""
        data = {"field1": "value1", "field3": "value3"}
        required = ["field1", "field2", "field3"]

        # Missing field2
        assert validate_required_fields(data, required) is False

    def test_validate_required_fields_case_sensitive(self) -> None:
        """Test validation is case sensitive for field names."""
        data = {"Field1": "value1", "FIELD2": "value2"}
        required = ["field1", "field2"]  # lowercase

        # Case doesn't match
        assert validate_required_fields(data, required) is False

    def test_validate_required_fields_single_field(self) -> None:
        """Test validation with single required field."""
        data = {"single_field": "value"}
        required = ["single_field"]

        assert validate_required_fields(data, required) is True

    def test_validate_required_fields_duplicate_required(self) -> None:
        """Test validation with duplicate field names in required list."""
        data = {"field1": "value1"}
        required = ["field1", "field1", "field1"]  # Duplicates

        # Should still work correctly
        assert validate_required_fields(data, required) is True


class TestCommonUtilitiesIntegration:
    """Integration tests for common utilities working together."""

    def test_logger_with_timestamp(self) -> None:
        """Test using logger with timestamp utility."""
        logger = setup_logger("integration_test_unique")

        # Capture log output - patch the specific logger's handler
        with patch.object(logger.handlers[0], "emit") as mock_emit:
            current_time = timestamp()
            logger.info(f"Integration test at {current_time}")

            # Verify emit was called
            mock_emit.assert_called_once()

    def test_safe_dict_get_with_validation(self) -> None:
        """Test combining safe_dict_get with validation."""
        data = {"field1": "value1", "field2": None}

        # Get values safely - need to handle None values separately from missing keys
        field1 = safe_dict_get(data, "field1", "default")
        field2 = safe_dict_get(
            data, "field2", "default"
        )  # This returns None, not default
        field3 = safe_dict_get(data, "field3", "default")  # This returns default

        # Handle None values explicitly for validation
        processed_data = {
            "field1": field1,
            "field2": field2 if field2 is not None else "default_for_none",
            "field3": field3,
        }

        # Validate the processed data
        required = ["field1", "field2", "field3"]
        is_valid = validate_required_fields(processed_data, required)

        # Should be valid since we handled None values
        assert is_valid is True
        assert processed_data["field2"] == "default_for_none"  # None was replaced
        assert processed_data["field3"] == "default"  # Missing was filled

    def test_flatten_list_with_dict_values(self) -> None:
        """Test flattening lists containing dictionary values."""
        dict1 = {"timestamp": timestamp(), "value": 1}
        dict2 = {"timestamp": timestamp(), "value": 2}

        nested = [[dict1], [dict2]]
        flattened = flatten_list(nested)

        assert len(flattened) == 2
        assert flattened[0]["value"] == 1
        assert flattened[1]["value"] == 2
        assert "timestamp" in flattened[0]
        assert "timestamp" in flattened[1]
