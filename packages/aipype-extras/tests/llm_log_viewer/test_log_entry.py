"""Tests for LogEntry dataclass."""

from aipype_extras.llm_log_viewer.llm_log_viewer import LogEntry


class TestLogEntry:
    """Test cases for LogEntry dataclass."""

    def test_log_entry_initialization(self) -> None:
        """Test LogEntry can be initialized with all fields."""
        entry = LogEntry(
            timestamp="2024-01-15T10:30:45Z",
            agent_name="TestAgent",
            task_name="test_task",
            provider="openai",
            model="gpt-3.5-turbo",
            input={"prompt": "Hello world"},
            output={"response": "Hi there!"},
            raw_line='{"timestamp": "2024-01-15T10:30:45Z"}',
            index=0,
        )

        assert entry.timestamp == "2024-01-15T10:30:45Z"
        assert entry.agent_name == "TestAgent"
        assert entry.task_name == "test_task"
        assert entry.provider == "openai"
        assert entry.model == "gpt-3.5-turbo"
        assert entry.input == {"prompt": "Hello world"}
        assert entry.output == {"response": "Hi there!"}
        assert entry.raw_line == '{"timestamp": "2024-01-15T10:30:45Z"}'
        assert entry.index == 0

    def test_log_entry_with_none_agent_name(self) -> None:
        """Test LogEntry with None agent_name (optional field)."""
        entry = LogEntry(
            timestamp="2024-01-15T10:30:45Z",
            agent_name=None,
            task_name="test_task",
            provider="openai",
            model="gpt-3.5-turbo",
            input={},
            output={},
            raw_line="{}",
            index=1,
        )

        assert entry.agent_name is None
        assert entry.task_name == "test_task"

    def test_log_entry_with_empty_dicts(self) -> None:
        """Test LogEntry with empty input/output dictionaries."""
        entry = LogEntry(
            timestamp="",
            agent_name=None,
            task_name="",
            provider="",
            model="",
            input={},
            output={},
            raw_line="",
            index=0,
        )

        assert entry.input == {}
        assert entry.output == {}
        assert entry.timestamp == ""
        assert entry.task_name == ""

    def test_log_entry_with_complex_data(self) -> None:
        """Test LogEntry with complex nested data structures."""
        complex_input = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            "temperature": 0.7,
            "max_tokens": 100,
        }

        complex_output = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8},
        }

        entry = LogEntry(
            timestamp="2024-01-15T10:30:45Z",
            agent_name="ComplexAgent",
            task_name="complex_task",
            provider="openai",
            model="gpt-4",
            input=complex_input,
            output=complex_output,
            raw_line='{"complex": "data"}',
            index=5,
        )

        assert entry.input == complex_input
        assert entry.output == complex_output
        assert entry.input["messages"][0]["role"] == "user"
        assert entry.output["choices"][0]["finish_reason"] == "stop"

    def test_log_entry_field_types(self) -> None:
        """Test that LogEntry fields maintain their expected types."""
        entry = LogEntry(
            timestamp="2024-01-15T10:30:45Z",
            agent_name="TestAgent",
            task_name="test_task",
            provider="openai",
            model="gpt-3.5-turbo",
            input={"key": "value"},
            output={"result": 42},
            raw_line="test_line",
            index=99,
        )

        assert isinstance(entry.timestamp, str)
        assert isinstance(entry.agent_name, str)
        assert isinstance(entry.task_name, str)
        assert isinstance(entry.provider, str)
        assert isinstance(entry.model, str)
        assert isinstance(entry.input, dict)
        assert isinstance(entry.output, dict)
        assert isinstance(entry.raw_line, str)
        assert isinstance(entry.index, int)

    def test_log_entry_equality(self) -> None:
        """Test LogEntry equality comparison."""
        entry1 = LogEntry(
            timestamp="2024-01-15T10:30:45Z",
            agent_name="TestAgent",
            task_name="test_task",
            provider="openai",
            model="gpt-3.5-turbo",
            input={"prompt": "Hello"},
            output={"response": "Hi"},
            raw_line="line1",
            index=0,
        )

        entry2 = LogEntry(
            timestamp="2024-01-15T10:30:45Z",
            agent_name="TestAgent",
            task_name="test_task",
            provider="openai",
            model="gpt-3.5-turbo",
            input={"prompt": "Hello"},
            output={"response": "Hi"},
            raw_line="line1",
            index=0,
        )

        entry3 = LogEntry(
            timestamp="2024-01-15T10:30:45Z",
            agent_name="DifferentAgent",
            task_name="test_task",
            provider="openai",
            model="gpt-3.5-turbo",
            input={"prompt": "Hello"},
            output={"response": "Hi"},
            raw_line="line1",
            index=0,
        )

        assert entry1 == entry2
        assert entry1 != entry3
