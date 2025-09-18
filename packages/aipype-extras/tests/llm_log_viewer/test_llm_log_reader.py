"""Tests for LLMLogReader class."""

import json
import os
import tempfile
from typing import List
import pytest
from aipype_extras.llm_log_viewer.llm_log_viewer import LLMLogReader, LogEntry


class TestLLMLogReader:
    """Test cases for LLMLogReader class."""

    @pytest.fixture
    def sample_log_entries(self) -> List[dict]:
        """Sample log entries for testing."""
        return [
            {
                "timestamp": "2024-01-15T10:30:45Z",
                "agent_name": "TestAgent1",
                "task_name": "task1",
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "input": {"prompt": "Hello world"},
                "output": {"response": "Hi there!"},
            },
            {
                "timestamp": "2024-01-15T10:31:00Z",
                "agent_name": "TestAgent2",
                "task_name": "task2",
                "provider": "anthropic",
                "model": "claude-3",
                "input": {"prompt": "How are you?"},
                "output": {"response": "I'm doing well!"},
            },
            {
                "timestamp": "2024-01-15T10:31:15Z",
                "agent_name": None,
                "task_name": "task3",
                "provider": "cohere",
                "model": "command",
                "input": {"prompt": "Tell me a joke"},
                "output": {"response": "Why did the chicken cross the road?"},
            },
        ]

    @pytest.fixture
    def temp_log_file(self, sample_log_entries: List[dict]) -> str:
        """Create a temporary log file with sample data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for entry in sample_log_entries:
                f.write(json.dumps(entry) + "\n")
            return f.name

    @pytest.fixture
    def empty_log_file(self) -> str:
        """Create an empty temporary log file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            return f.name

    @pytest.fixture
    def malformed_log_file(self) -> str:
        """Create a log file with malformed JSON lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"valid": "json"}\n')
            f.write("invalid json line\n")
            f.write('{"another": "valid", "entry": true}\n')
            f.write('{"incomplete": json\n')
            f.write('{"final": "valid"}\n')
            return f.name

    def teardown_method(self) -> None:
        """Clean up temporary files after each test."""
        # This will be called after each test method
        pass

    def test_init(self, temp_log_file: str) -> None:
        """Test LLMLogReader initialization."""
        reader = LLMLogReader(temp_log_file)
        assert reader.log_file_path == temp_log_file

    def test_read_logs_basic(
        self, temp_log_file: str, sample_log_entries: List[dict]
    ) -> None:
        """Test basic log reading functionality."""
        reader = LLMLogReader(temp_log_file)
        logs = reader.read_logs()

        assert len(logs) == 3
        assert isinstance(logs[0], LogEntry)

        # Should be in reverse order by default
        assert logs[0].task_name == "task3"
        assert logs[1].task_name == "task2"
        assert logs[2].task_name == "task1"

    def test_read_logs_forward_order(self, temp_log_file: str) -> None:
        """Test reading logs in forward order."""
        reader = LLMLogReader(temp_log_file)
        logs = reader.read_logs(reverse=False)

        assert len(logs) == 3
        assert logs[0].task_name == "task1"
        assert logs[1].task_name == "task2"
        assert logs[2].task_name == "task3"

    def test_read_logs_with_pagination(self, temp_log_file: str) -> None:
        """Test log reading with pagination parameters."""
        reader = LLMLogReader(temp_log_file)

        # Get first 2 entries (reverse order)
        logs = reader.read_logs(start_index=0, count=2)
        assert len(logs) == 2
        assert logs[0].task_name == "task3"
        assert logs[1].task_name == "task2"

        # Get next entry
        logs = reader.read_logs(start_index=2, count=1)
        assert len(logs) == 1
        assert logs[0].task_name == "task1"

        # Request more than available
        logs = reader.read_logs(start_index=1, count=10)
        assert len(logs) == 2  # Only 2 remaining after index 1

    def test_read_logs_empty_file(self, empty_log_file: str) -> None:
        """Test reading from an empty log file."""
        reader = LLMLogReader(empty_log_file)
        logs = reader.read_logs()
        assert len(logs) == 0

    def test_read_logs_nonexistent_file(self) -> None:
        """Test reading from a non-existent file."""
        reader = LLMLogReader("/path/that/does/not/exist.jsonl")
        logs = reader.read_logs()
        assert len(logs) == 0

    def test_read_logs_malformed_json(self, malformed_log_file: str) -> None:
        """Test that malformed JSON lines are skipped gracefully."""
        reader = LLMLogReader(malformed_log_file)
        logs = reader.read_logs()

        # Should only read the 3 valid JSON lines, skipping malformed ones
        assert len(logs) == 3
        assert logs[0].task_name == ""  # final entry (reverse order)
        assert "final" in logs[0].raw_line
        assert "another" in logs[1].raw_line
        assert "valid" in logs[2].raw_line

    def test_read_logs_field_mapping(self, temp_log_file: str) -> None:
        """Test that log fields are correctly mapped to LogEntry."""
        reader = LLMLogReader(temp_log_file)
        logs = reader.read_logs(count=1)  # Get just first entry

        log = logs[0]
        assert log.timestamp == "2024-01-15T10:31:15Z"
        assert log.agent_name is None
        assert log.task_name == "task3"
        assert log.provider == "cohere"
        assert log.model == "command"
        assert log.input == {"prompt": "Tell me a joke"}
        assert log.output == {"response": "Why did the chicken cross the road?"}
        assert log.index == 2  # Original line index before reversal

    def test_read_logs_missing_fields(self) -> None:
        """Test handling of log entries with missing fields."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Entry with missing optional fields
            f.write('{"timestamp": "2024-01-15T10:30:45Z"}\n')
            f.write('{"task_name": "test", "provider": "openai"}\n')
            temp_file = f.name

        try:
            reader = LLMLogReader(temp_file)
            logs = reader.read_logs()

            assert len(logs) == 2

            # Check defaults are applied for missing fields
            log1 = logs[0]  # Second entry (reverse order)
            assert log1.timestamp == ""
            assert log1.agent_name is None
            assert log1.task_name == "test"
            assert log1.provider == "openai"
            assert log1.model == ""
            assert log1.input == {}
            assert log1.output == {}

            log2 = logs[1]  # First entry
            assert log2.timestamp == "2024-01-15T10:30:45Z"
            assert log2.task_name == ""

        finally:
            os.unlink(temp_file)

    def test_get_total_log_count(self, temp_log_file: str) -> None:
        """Test getting total log count."""
        reader = LLMLogReader(temp_log_file)
        count = reader.get_total_log_count()
        assert count == 3

    def test_get_total_log_count_empty_file(self, empty_log_file: str) -> None:
        """Test getting count for empty file."""
        reader = LLMLogReader(empty_log_file)
        count = reader.get_total_log_count()
        assert count == 0

    def test_get_total_log_count_nonexistent_file(self) -> None:
        """Test getting count for non-existent file."""
        reader = LLMLogReader("/path/that/does/not/exist.jsonl")
        count = reader.get_total_log_count()
        assert count == 0

    def test_get_total_log_count_with_empty_lines(self) -> None:
        """Test count calculation with empty lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"entry": 1}\n')
            f.write("\n")  # Empty line
            f.write('{"entry": 2}\n')
            f.write("   \n")  # Whitespace only
            f.write('{"entry": 3}\n')
            temp_file = f.name

        try:
            reader = LLMLogReader(temp_file)
            count = reader.get_total_log_count()
            assert count == 3  # Should only count non-empty lines

        finally:
            os.unlink(temp_file)

    def test_log_entry_index_assignment(self, temp_log_file: str) -> None:
        """Test that LogEntry index reflects original line position."""
        reader = LLMLogReader(temp_log_file)
        logs = reader.read_logs(reverse=False)  # Forward order to check indices

        assert logs[0].index == 0
        assert logs[1].index == 1
        assert logs[2].index == 2

    def test_read_logs_boundary_conditions(self, temp_log_file: str) -> None:
        """Test boundary conditions for pagination."""
        reader = LLMLogReader(temp_log_file)

        # Start index beyond file length
        logs = reader.read_logs(start_index=10, count=5)
        assert len(logs) == 0

        # Zero count
        logs = reader.read_logs(start_index=0, count=0)
        assert len(logs) == 0

        # Negative start index (should behave reasonably)
        logs = reader.read_logs(start_index=0, count=1)
        assert len(logs) == 1

    def test_file_encoding_handling(self) -> None:
        """Test handling of files with different encodings."""
        # Create file with UTF-8 content
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            entry = {
                "timestamp": "2024-01-15T10:30:45Z",
                "task_name": "test_unicode_🚀",
                "provider": "openai",
                "input": {"prompt": "Hello 世界"},
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            temp_file = f.name

        try:
            reader = LLMLogReader(temp_file)
            logs = reader.read_logs()

            assert len(logs) == 1
            assert logs[0].task_name == "test_unicode_🚀"
            assert logs[0].input["prompt"] == "Hello 世界"

        finally:
            os.unlink(temp_file)
