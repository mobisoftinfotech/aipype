"""Tests for Flask app creation and routes."""

import json
from typing import Generator, List
from unittest.mock import Mock, patch
import pytest
from flask import Flask
from flask.testing import FlaskClient
from aipype_extras.llm_log_viewer.llm_log_viewer import (
    create_app,
    LLMLogReader,
    LogEntry,
)


class TestFlaskApp:
    """Test cases for Flask app functionality."""

    @pytest.fixture
    def mock_log_reader(self) -> Mock:
        """Create a mock LLMLogReader for testing."""
        mock_reader = Mock(spec=LLMLogReader)
        return mock_reader

    @pytest.fixture
    def sample_log_entries(self) -> List[LogEntry]:
        """Sample LogEntry objects for testing."""
        return [
            LogEntry(
                timestamp="2024-01-15T10:31:15Z",
                agent_name="TestAgent1",
                task_name="task1",
                provider="openai",
                model="gpt-3.5-turbo",
                input={"prompt": "Hello world"},
                output={"response": "Hi there!"},
                raw_line='{"timestamp": "2024-01-15T10:31:15Z"}',
                index=0,
            ),
            LogEntry(
                timestamp="2024-01-15T10:30:45Z",
                agent_name=None,
                task_name="task2",
                provider="anthropic",
                model="claude-3",
                input={"prompt": "How are you?"},
                output={"response": "I'm doing well!"},
                raw_line='{"timestamp": "2024-01-15T10:30:45Z"}',
                index=1,
            ),
        ]

    @pytest.fixture
    def app(self, mock_log_reader: Mock) -> Flask:
        """Create Flask app for testing."""
        return create_app(mock_log_reader)

    @pytest.fixture
    def client(self, app: Flask) -> Generator[FlaskClient, None, None]:
        """Create test client."""
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_create_app_returns_flask_instance(self, mock_log_reader: Mock) -> None:
        """Test that create_app returns a Flask instance."""
        app = create_app(mock_log_reader)
        assert isinstance(app, Flask)

    def test_create_app_sets_security_headers(
        self, client, mock_log_reader: Mock
    ) -> None:
        """Test that security headers are properly set."""
        mock_log_reader.read_logs.return_value = []
        mock_log_reader.get_total_log_count.return_value = 0

        response = client.get("/api/logs")

        assert response.headers.get("Access-Control-Allow-Origin") == "*"
        assert "GET, POST, PUT, DELETE, OPTIONS" in response.headers.get(
            "Access-Control-Allow-Methods", ""
        )
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "SAMEORIGIN"

    @patch("aipype_extras.llm_log_viewer.llm_log_viewer.render_template")
    def test_index_route(self, mock_render_template, client) -> None:
        """Test the index route."""
        mock_render_template.return_value = "<html>Test Page</html>"

        response = client.get("/")

        assert response.status_code == 200
        mock_render_template.assert_called_once_with("index.html")

    def test_api_logs_route_basic(
        self, client, mock_log_reader: Mock, sample_log_entries: List[LogEntry]
    ) -> None:
        """Test the /api/logs route with basic functionality."""
        mock_log_reader.read_logs.return_value = sample_log_entries
        mock_log_reader.get_total_log_count.return_value = 2

        response = client.get("/api/logs")

        assert response.status_code == 200
        assert response.content_type == "application/json"

        data = json.loads(response.data)
        assert "logs" in data
        assert "total" in data
        assert "offset" in data
        assert "count" in data

        assert data["total"] == 2
        assert data["offset"] == 0
        assert data["count"] == 2
        assert len(data["logs"]) == 2

        # Verify log data structure
        log1 = data["logs"][0]
        assert log1["timestamp"] == "2024-01-15T10:31:15Z"
        assert log1["agent_name"] == "TestAgent1"
        assert log1["task_name"] == "task1"
        assert log1["provider"] == "openai"
        assert log1["model"] == "gpt-3.5-turbo"
        assert log1["input"] == {"prompt": "Hello world"}
        assert log1["output"] == {"response": "Hi there!"}
        assert log1["index"] == 0

        # Verify None agent_name handling
        log2 = data["logs"][1]
        assert log2["agent_name"] is None
        assert log2["task_name"] == "task2"

    def test_api_logs_route_with_pagination(
        self, client, mock_log_reader: Mock
    ) -> None:
        """Test the /api/logs route with pagination parameters."""
        mock_log_reader.read_logs.return_value = []
        mock_log_reader.get_total_log_count.return_value = 100

        response = client.get("/api/logs?offset=10&count=25")

        assert response.status_code == 200
        data = json.loads(response.data)

        assert data["offset"] == 10
        assert data["total"] == 100
        assert data["count"] == 0  # No logs returned from mock

        # Verify the mock was called with correct parameters
        mock_log_reader.read_logs.assert_called_once_with(
            start_index=10, count=25, reverse=True
        )

    def test_api_logs_route_default_parameters(
        self, client, mock_log_reader: Mock
    ) -> None:
        """Test the /api/logs route uses default parameters when none provided."""
        mock_log_reader.read_logs.return_value = []
        mock_log_reader.get_total_log_count.return_value = 0

        response = client.get("/api/logs")

        assert response.status_code == 200

        # Verify default parameters were used
        mock_log_reader.read_logs.assert_called_once_with(
            start_index=0, count=50, reverse=True
        )

    def test_api_logs_route_parameter_conversion(
        self, client, mock_log_reader: Mock
    ) -> None:
        """Test that query parameters are properly converted to integers."""
        mock_log_reader.read_logs.return_value = []
        mock_log_reader.get_total_log_count.return_value = 0

        response = client.get("/api/logs?offset=5&count=15")

        assert response.status_code == 200

        # Verify parameters were converted to integers
        mock_log_reader.read_logs.assert_called_once_with(
            start_index=5, count=15, reverse=True
        )

    def test_api_logs_route_empty_results(self, client, mock_log_reader: Mock) -> None:
        """Test the /api/logs route with empty results."""
        mock_log_reader.read_logs.return_value = []
        mock_log_reader.get_total_log_count.return_value = 0

        response = client.get("/api/logs")

        assert response.status_code == 200
        data = json.loads(response.data)

        assert data["logs"] == []
        assert data["total"] == 0
        assert data["count"] == 0

    def test_api_logs_route_with_complex_data(
        self, client, mock_log_reader: Mock
    ) -> None:
        """Test the /api/logs route with complex nested data."""
        complex_entry = LogEntry(
            timestamp="2024-01-15T10:31:15Z",
            agent_name="ComplexAgent",
            task_name="complex_task",
            provider="openai",
            model="gpt-4",
            input={
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                ],
                "temperature": 0.7,
            },
            output={
                "choices": [{"message": {"role": "assistant", "content": "Response"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
            raw_line='{"complex": "data"}',
            index=0,
        )

        mock_log_reader.read_logs.return_value = [complex_entry]
        mock_log_reader.get_total_log_count.return_value = 1

        response = client.get("/api/logs")

        assert response.status_code == 200
        data = json.loads(response.data)

        log = data["logs"][0]
        assert log["input"]["messages"][0]["role"] == "user"
        assert log["input"]["temperature"] == 0.7
        assert log["output"]["usage"]["prompt_tokens"] == 10

    def test_api_logs_error_handling(self, client, mock_log_reader: Mock) -> None:
        """Test error handling in the /api/logs route."""
        # Mock an exception being raised by read_logs
        mock_log_reader.read_logs.side_effect = Exception("File read error")
        mock_log_reader.get_total_log_count.return_value = 0

        # Currently the Flask app doesn't have try/catch around log_reader calls
        # so exceptions will bubble up and Flask will return a 500 error
        with pytest.raises(Exception, match="File read error"):
            client.get("/api/logs")

    def test_invalid_route_404(self, client) -> None:
        """Test that invalid routes return 404."""
        response = client.get("/nonexistent-route")
        assert response.status_code == 404

    def test_api_logs_route_data_types(
        self, client, mock_log_reader: Mock, sample_log_entries: List[LogEntry]
    ) -> None:
        """Test that the API returns proper data types."""
        mock_log_reader.read_logs.return_value = sample_log_entries
        mock_log_reader.get_total_log_count.return_value = 2

        response = client.get("/api/logs")

        assert response.status_code == 200
        data = json.loads(response.data)

        # Verify top-level data types
        assert isinstance(data["logs"], list)
        assert isinstance(data["total"], int)
        assert isinstance(data["offset"], int)
        assert isinstance(data["count"], int)

        # Verify log entry data types
        if data["logs"]:
            log = data["logs"][0]
            assert isinstance(log["timestamp"], str)
            assert isinstance(log["task_name"], str)
            assert isinstance(log["provider"], str)
            assert isinstance(log["model"], str)
            assert isinstance(log["input"], dict)
            assert isinstance(log["output"], dict)
            assert isinstance(log["index"], int)
            # agent_name can be string or None

    def test_cors_preflight_request(self, client, mock_log_reader: Mock) -> None:
        """Test CORS preflight OPTIONS request handling."""
        response = client.options("/api/logs")

        # Should include CORS headers
        assert "Access-Control-Allow-Origin" in response.headers
        assert "Access-Control-Allow-Methods" in response.headers
        assert "Access-Control-Allow-Headers" in response.headers

    def test_multiple_requests_independence(
        self, client, mock_log_reader: Mock
    ) -> None:
        """Test that multiple requests are independent."""
        mock_log_reader.read_logs.return_value = []
        mock_log_reader.get_total_log_count.return_value = 0

        # Make multiple requests
        response1 = client.get("/api/logs?offset=0&count=10")
        response2 = client.get("/api/logs?offset=10&count=20")

        assert response1.status_code == 200
        assert response2.status_code == 200

        # Verify each request called the mock with different parameters
        assert mock_log_reader.read_logs.call_count == 2

        calls = mock_log_reader.read_logs.call_args_list
        assert calls[0].kwargs["start_index"] == 0
        assert calls[0].kwargs["count"] == 10
        assert calls[1].kwargs["start_index"] == 10
        assert calls[1].kwargs["count"] == 20
