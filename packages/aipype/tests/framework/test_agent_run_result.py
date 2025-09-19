"""Tests for AgentRunResult - standardized agent execution response format."""

from aipype import AgentRunResult, AgentRunStatus


class TestAgentRunResult:
    """Test AgentRunResult functionality."""

    def test_success_creation(self) -> None:
        """Test creating a successful AgentRunResult."""
        agent_result = AgentRunResult.success(
            agent_name="test_agent",
            total_tasks=2,
            completed_tasks=2,
            total_phases=1,
            execution_time=1.5,
            metadata={"test": "value"},
        )

        assert agent_result.status == AgentRunStatus.SUCCESS
        assert agent_result.agent_name == "test_agent"
        assert agent_result.total_tasks == 2
        assert agent_result.completed_tasks == 2
        assert agent_result.failed_tasks == 0
        assert agent_result.total_phases == 1
        assert agent_result.execution_time == 1.5
        assert agent_result.metadata == {"test": "value"}
        assert agent_result.error_message is None

    def test_partial_creation(self) -> None:
        """Test creating a partial success AgentRunResult."""
        agent_result = AgentRunResult.partial(
            agent_name="test_agent",
            total_tasks=2,
            completed_tasks=1,
            failed_tasks=1,
            total_phases=1,
            execution_time=0.8,
        )

        assert agent_result.status == AgentRunStatus.PARTIAL
        assert agent_result.agent_name == "test_agent"
        assert agent_result.total_tasks == 2
        assert agent_result.completed_tasks == 1
        assert agent_result.failed_tasks == 1
        assert agent_result.total_phases == 1
        assert agent_result.execution_time == 0.8

    def test_failure_creation(self) -> None:
        """Test creating a failed AgentRunResult."""
        agent_result = AgentRunResult.failure(
            agent_name="test_agent",
            total_tasks=2,
            failed_tasks=2,
            error_message="All tasks failed",
            total_phases=1,
            execution_time=0.2,
        )

        assert agent_result.status == AgentRunStatus.ERROR
        assert agent_result.agent_name == "test_agent"
        assert agent_result.total_tasks == 2
        assert agent_result.completed_tasks == 0
        assert agent_result.failed_tasks == 2
        assert agent_result.total_phases == 1
        assert agent_result.execution_time == 0.2
        assert agent_result.error_message == "All tasks failed"

    def test_running_creation(self) -> None:
        """Test creating a running AgentRunResult."""
        agent_result = AgentRunResult.running(
            agent_name="test_agent",
            metadata={"start_time": "2023-01-01T00:00:00"},
        )

        assert agent_result.status == AgentRunStatus.RUNNING
        assert agent_result.agent_name == "test_agent"
        assert agent_result.total_tasks == 0
        assert agent_result.completed_tasks == 0
        assert agent_result.failed_tasks == 0
        assert agent_result.total_phases == 0
        assert agent_result.execution_time == 0.0
        assert agent_result.metadata == {"start_time": "2023-01-01T00:00:00"}

    def test_status_helper_methods(self) -> None:
        """Test status checking helper methods."""
        success_result = AgentRunResult.success("agent", 1, 1, 1)
        partial_result = AgentRunResult.partial("agent", 2, 1, 1, 1)
        error_result = AgentRunResult.failure("agent", 1, 1, "Error")
        running_result = AgentRunResult.running("agent")

        # Test is_success()
        assert success_result.is_success()
        assert not partial_result.is_success()
        assert not error_result.is_success()
        assert not running_result.is_success()

        # Test is_partial()
        assert not success_result.is_partial()
        assert partial_result.is_partial()
        assert not error_result.is_partial()
        assert not running_result.is_partial()

        # Test is_error()
        assert not success_result.is_error()
        assert not partial_result.is_error()
        assert error_result.is_error()
        assert not running_result.is_error()

        # Test is_running()
        assert not success_result.is_running()
        assert not partial_result.is_running()
        assert not error_result.is_running()
        assert running_result.is_running()

    def test_metadata_operations(self) -> None:
        """Test metadata add and get operations."""
        agent_result = AgentRunResult.success("agent", 1, 1, 1)

        # Test adding metadata
        agent_result.add_metadata("key1", "value1")
        agent_result.add_metadata("key2", 42)

        # Test getting metadata
        assert agent_result.get_metadata("key1") == "value1"
        assert agent_result.get_metadata("key2") == 42
        assert agent_result.get_metadata("missing", "default") == "default"
        assert agent_result.get_metadata("missing") is None

    def test_string_representation(self) -> None:
        """Test string representation of AgentRunResult."""
        # Test success without execution time
        success_result = AgentRunResult.success("test_agent", 2, 2, 1)
        success_str = str(success_result)
        assert "test_agent" in success_str
        assert "success" in success_str
        assert "2/2 tasks" in success_str

        # Test with execution time
        timed_result = AgentRunResult.success(
            "test_agent", 1, 1, 1, execution_time=1.234
        )
        timed_str = str(timed_result)
        assert "(1.234s)" in timed_str

        # Test with error message
        error_result = AgentRunResult.failure("test_agent", 1, 1, "Test error")
        error_str = str(error_result)
        assert "error='Test error'" in error_str
        assert "test_agent" in error_str

    def test_default_values(self) -> None:
        """Test that default values are properly set."""
        # Create with minimal parameters
        agent_result = AgentRunResult(
            status=AgentRunStatus.SUCCESS, agent_name="test_agent"
        )

        assert agent_result.total_tasks == 0
        assert agent_result.completed_tasks == 0
        assert agent_result.failed_tasks == 0
        assert agent_result.total_phases == 0
        assert agent_result.execution_time == 0.0
        assert agent_result.metadata == {}
        assert agent_result.error_message is None
