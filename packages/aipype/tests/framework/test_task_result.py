"""Tests for TaskResult standardized response format."""

import pytest

from aipype import TaskResult, TaskStatus, wrap_legacy_result, unwrap_to_legacy


class TestTaskResult:
    """Test TaskResult class functionality."""

    def test_success_creation(self) -> None:
        """Test creating successful task result."""
        data = {"key": "value"}
        metadata = {"source": "test"}
        result = TaskResult.success(data=data, metadata=metadata, execution_time=1.5)

        assert result.status == TaskStatus.SUCCESS
        assert result.data == data
        assert result.error is None
        assert result.metadata == metadata
        assert result.execution_time == 1.5
        assert result.is_success()
        assert not result.is_error()
        assert not result.is_partial()
        assert not result.is_skipped()
        assert result.has_data()

    def test_error_creation(self) -> None:
        """Test creating error task result."""
        error_msg = "Something went wrong"
        metadata = {"error_code": 500}
        result = TaskResult.failure(
            error_message=error_msg, metadata=metadata, execution_time=0.5
        )

        assert result.status == TaskStatus.ERROR
        assert result.data is None
        assert result.error == error_msg
        assert result.metadata == metadata
        assert result.execution_time == 0.5
        assert not result.is_success()
        assert result.is_error()
        assert not result.is_partial()
        assert not result.is_skipped()
        assert not result.has_data()

    def test_partial_creation(self) -> None:
        """Test creating partial success task result."""
        data = [1, 2, 3]
        error_msg = "Some items failed"
        result = TaskResult.partial(data=data, error=error_msg, execution_time=2.0)

        assert result.status == TaskStatus.PARTIAL
        assert result.data == data
        assert result.error == error_msg
        assert result.execution_time == 2.0
        assert not result.is_success()
        assert not result.is_error()
        assert result.is_partial()
        assert not result.is_skipped()
        assert result.has_data()

    def test_skipped_creation(self) -> None:
        """Test creating skipped task result."""
        reason = "Condition not met"
        result = TaskResult.skipped(reason=reason, execution_time=0.1)

        assert result.status == TaskStatus.SKIPPED
        assert result.data is None
        assert result.error == reason
        assert result.execution_time == 0.1
        assert not result.is_success()
        assert not result.is_error()
        assert not result.is_partial()
        assert result.is_skipped()
        assert not result.has_data()

    def test_metadata_operations(self) -> None:
        """Test metadata operations."""
        result = TaskResult.success(data="test")

        # Test adding metadata
        result.add_metadata("key1", "value1")
        result.add_metadata("key2", 42)

        assert result.get_metadata("key1") == "value1"
        assert result.get_metadata("key2") == 42
        assert result.get_metadata("nonexistent", "default") == "default"
        assert result.metadata["key1"] == "value1"

    def test_legacy_result_success(self) -> None:
        """Test legacy result extraction for success."""
        data = {"result": "success"}
        result = TaskResult.success(data=data)

        legacy_result = result.get_legacy_result()
        assert legacy_result == data

    def test_legacy_result_partial(self) -> None:
        """Test legacy result extraction for partial success."""
        data = [1, 2, 3]
        result = TaskResult.partial(data=data, error="Some failed")

        legacy_result = result.get_legacy_result()
        assert legacy_result == data

    def test_legacy_result_skipped(self) -> None:
        """Test legacy result extraction for skipped."""
        result = TaskResult.skipped(reason="Condition not met")

        legacy_result = result.get_legacy_result()
        assert legacy_result is None

    def test_legacy_result_error(self) -> None:
        """Test legacy result extraction for error raises exception."""
        result = TaskResult.failure(error_message="Task failed")

        with pytest.raises(RuntimeError, match="Task failed"):
            result.get_legacy_result()

    def test_string_representation(self) -> None:
        """Test string representation of TaskResult."""
        # Success with data
        result = TaskResult.success(data={"key": "value"}, execution_time=1.5)
        str_repr = str(result)
        assert "success (1.500s)" in str_repr
        assert "data=dict" in str_repr

        # Error with message
        result = TaskResult.failure(error_message="Failed")
        str_repr = str(result)
        assert "error" in str_repr
        assert "Failed" in str_repr

        # No execution time
        result = TaskResult.success(data="test")
        str_repr = str(result)
        assert "(0.000s)" not in str_repr

    def test_default_metadata(self) -> None:
        """Test that metadata defaults to empty dict."""
        result = TaskResult(status=TaskStatus.SUCCESS)
        assert isinstance(result.metadata, dict)
        assert len(result.metadata) == 0

    def test_default_metadata_factory(self) -> None:
        """Test that default factory creates empty dict for metadata."""
        result = TaskResult(status=TaskStatus.SUCCESS)
        assert isinstance(result.metadata, dict)
        assert len(result.metadata) == 0

        # Test that each instance gets its own metadata dict
        result2 = TaskResult(status=TaskStatus.ERROR, error="test")
        result.add_metadata("key", "value")
        assert len(result.metadata) == 1
        assert len(result2.metadata) == 0  # Should not share metadata


class TestTaskResultUtilities:
    """Test utility functions for TaskResult."""

    def test_wrap_legacy_result(self) -> None:
        """Test wrapping legacy result."""
        legacy_data = {"old": "format"}
        execution_time = 2.5

        result = wrap_legacy_result(legacy_data, execution_time)

        assert result.status == TaskStatus.SUCCESS
        assert result.data == legacy_data
        assert result.execution_time == execution_time
        assert result.is_success()

    def test_unwrap_to_legacy_success(self) -> None:
        """Test unwrapping successful result to legacy format."""
        data = {"new": "format"}
        result = TaskResult.success(data=data)

        unwrapped = unwrap_to_legacy(result)
        assert unwrapped == data

    def test_unwrap_to_legacy_error(self) -> None:
        """Test unwrapping error result raises exception."""
        result = TaskResult.failure(error_message="Task failed")

        with pytest.raises(RuntimeError, match="Task failed"):
            unwrap_to_legacy(result)

    def test_wrap_unwrap_roundtrip(self) -> None:
        """Test wrapping and unwrapping maintains data."""
        original_data = {"test": "data", "numbers": [1, 2, 3]}

        # Wrap legacy result
        wrapped = wrap_legacy_result(original_data, 1.0)

        # Unwrap back to legacy
        unwrapped = unwrap_to_legacy(wrapped)

        assert unwrapped == original_data


class TestTaskStatus:
    """Test TaskStatus enum."""

    def test_status_values(self) -> None:
        """Test TaskStatus enum values."""
        # Lifecycle statuses
        assert TaskStatus.NOT_STARTED.value == "not_started"
        assert TaskStatus.STARTED.value == "started"

        # Completion statuses
        assert TaskStatus.SUCCESS.value == "success"
        assert TaskStatus.ERROR.value == "error"
        assert TaskStatus.PARTIAL.value == "partial"
        assert TaskStatus.SKIPPED.value == "skipped"

    def test_status_comparison(self) -> None:
        """Test TaskStatus comparison."""
        assert TaskStatus.SUCCESS == TaskStatus.SUCCESS

        # These comparisons are intentionally testing that different enum values are not equal.
        # The type checker flags them as "unnecessary" since it knows they'll never be equal,
        # but these are valid test assertions to verify enum comparison behavior.
        assert TaskStatus.SUCCESS != TaskStatus.ERROR
        assert TaskStatus.ERROR != TaskStatus.PARTIAL
        assert TaskStatus.SKIPPED != TaskStatus.SUCCESS
