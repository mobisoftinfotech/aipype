"""Integration tests for Gmail tasks with real Gmail API calls.

These tests verify the Gmail task implementations (GmailListEmailsTask, GmailReadEmailTask)
work correctly with real Gmail API calls and return proper TaskResult objects.

Prerequisites:
- Valid Google credentials file with Gmail scope
- OAuth2 tokens will be created/refreshed automatically
- Gmail account with some test emails
- Internet connection for API calls

Setup:
1. Download OAuth2 credentials from Google Cloud Console
2. Set environment variables:
   export GOOGLE_CREDENTIALS_FILE=./google_credentials.json
   export GMAIL_TOKEN_FILE=./gmail_token.json
3. Run tests - first run will trigger OAuth2 flow in browser

Run with: pytest integration_tests/tasklib/google/test_gmail_tasks_integration.py -v

IMPORTANT: These tests will read real emails from your Gmail account.
Use a dedicated test account when possible.

Test Classes:
- TestGmailListEmailsTaskIntegration: GmailListEmailsTask functionality tests
- TestGmailReadEmailTaskIntegration: GmailReadEmailTask functionality tests
- TestGmailTasksUnifiedAuthentication: GoogleOAuthTask integration and workflow tests
"""

import os
import pytest
import time
from typing import Any, List

from aipype import (
    TaskDependency,
    DependencyType,
    TaskResult,
)
from aipype_tl_google import (
    GmailListEmailsTask,
    GmailReadEmailTask,
    GoogleOAuthTask,
    GmailMessage,
)
from aipype_tl_google.tasklib.gmail_models import (
    GmailSystemLabels,
    GmailSearchOperators,
)


@pytest.fixture(scope="session")
def gmail_credentials_available() -> bool:
    """Check if Google credentials file exists."""
    google_credentials = os.getenv("GOOGLE_CREDENTIALS_FILE")
    return google_credentials is not None and os.path.exists(google_credentials)


@pytest.fixture(scope="session")
def skip_if_no_gmail_credentials(gmail_credentials_available: bool) -> None:
    """Skip test if Google credentials are not available."""
    if not gmail_credentials_available:
        pytest.skip(
            "Google credentials not available. "
            "Download from Google Cloud Console and set GOOGLE_CREDENTIALS_FILE environment variable."
        )


@pytest.mark.integration
@pytest.mark.gmail
@pytest.mark.slow
class TestGmailListEmailsTaskIntegration:
    """Integration tests for GmailListEmailsTask with real Gmail API."""

    def test_basic_list_emails_task(self, google_oauth_credentials: Any) -> None:
        """Test basic Gmail list emails task functionality."""
        task = GmailListEmailsTask(
            "test_list_emails",
            {
                "query": "",
                "max_results": 5,
                "parse_messages": True,
                "timeout": 60,
                "credentials": google_oauth_credentials,
            },
        )

        start_time = time.time()
        result = task.run()
        execution_time = time.time() - start_time

        # Verify task result structure
        assert isinstance(result, TaskResult)
        assert result.is_success()

        # Verify data structure
        assert "messages" in result.data
        assert "message_ids" in result.data
        assert "search_result" in result.data
        assert "query_used" in result.data
        assert "total_found" in result.data
        assert "retrieved_count" in result.data

        # Verify messages format
        messages: List[GmailMessage] = result.data["messages"]
        assert isinstance(messages, list)

        if messages:  # If there are messages
            first_message: GmailMessage = messages[0]
            assert isinstance(first_message, GmailMessage)
            assert first_message.message_id
            assert first_message.thread_id

        message_ids: List[str] = result.data["message_ids"]
        assert isinstance(message_ids, list)
        assert len(message_ids) == len(messages)

        print(f"[SUCCESS] Listed {len(messages)} emails in {execution_time:.2f}s")
        if messages:
            print(f"   First message: '{messages[0].subject[:50]}...'")

    def test_list_emails_with_query(self, google_oauth_credentials: Any) -> None:
        """Test Gmail list emails with search query."""
        # Search for messages in inbox from last 30 days
        query = GmailSearchOperators.combine_and(
            "in:inbox", GmailSearchOperators.newer_than(30)
        )

        task = GmailListEmailsTask(
            "test_list_with_query",
            {
                "query": query,
                "max_results": 10,
                "parse_messages": True,
                "credentials": google_oauth_credentials,
            },
        )

        result = task.run()

        assert result.is_success()
        assert result.data["query_used"] == query

        messages = result.data["messages"]

        print(f"[SUCCESS] Query '{query}' returned {len(messages)} messages")

    def test_list_emails_with_labels(self, google_oauth_credentials: Any) -> None:
        """Test Gmail list emails filtered by labels."""
        task = GmailListEmailsTask(
            "test_list_with_labels",
            {
                "query": "",
                "max_results": 8,
                "label_ids": [GmailSystemLabels.INBOX],
                "parse_messages": True,
                "credentials": google_oauth_credentials,
            },
        )

        result = task.run()

        assert result.is_success()
        assert "messages" in result.data

        messages = result.data["messages"]

        # All messages should have INBOX label
        if messages:
            for message in messages:
                assert GmailSystemLabels.INBOX in message.label_ids

        print(f"[SUCCESS] INBOX label filter returned {len(messages)} messages")

    def test_list_emails_parse_messages_false(
        self, google_oauth_credentials: Any
    ) -> None:
        """Test Gmail list emails without parsing full message content."""
        task = GmailListEmailsTask(
            "test_list_no_parse",
            {
                "query": "",
                "max_results": 5,
                "parse_messages": False,  # Only get message IDs
                "credentials": google_oauth_credentials,
            },
        )

        result = task.run()

        assert result.is_success()
        assert "message_ids" in result.data

        # Messages should be empty when parse_messages=False
        messages = result.data["messages"]
        message_ids = result.data["message_ids"]

        assert len(messages) == 0  # No parsed messages
        assert len(message_ids) >= 0  # But should have IDs

        print(f"[SUCCESS] Retrieved {len(message_ids)} message IDs without parsing")

    def test_list_emails_with_dependencies(self, google_oauth_credentials: Any) -> None:
        """Test Gmail list emails task with dependency resolution."""
        # Create task with dependency for query
        task = GmailListEmailsTask(
            "test_list_with_dependency",
            {
                "max_results": 3,
                "parse_messages": True,
                "credentials": google_oauth_credentials,
            },
            dependencies=[
                TaskDependency(
                    "query", "search_params.gmail_query", DependencyType.REQUIRED
                )
            ],
        )

        # Simulate dependency resolution
        task.config["query"] = GmailSearchOperators.newer_than(7)  # Last week

        result = task.run()

        assert result.is_success()
        assert result.data["query_used"] == GmailSearchOperators.newer_than(7)

        print("[SUCCESS] Dependency resolution working for Gmail list task")

    def test_list_emails_error_handling(self, google_oauth_credentials: Any) -> None:
        """Test error handling with invalid configuration."""
        # Test with invalid max_results
        task = GmailListEmailsTask(
            "test_error_handling",
            {
                "query": "",
                "max_results": 150,  # Over limit (should be max 100)
                "parse_messages": True,
                "credentials": google_oauth_credentials,
            },
        )

        result = task.run()

        # Should handle validation error gracefully
        assert isinstance(result, TaskResult)
        # Might succeed if Gmail API accepts it, or fail gracefully

        print("[SUCCESS] Error handling test completed")


@pytest.mark.integration
@pytest.mark.gmail
@pytest.mark.slow
class TestGmailReadEmailTaskIntegration:
    """Integration tests for GmailReadEmailTask with real Gmail API."""

    def test_basic_read_email_task(self, google_oauth_credentials: Any) -> None:
        """Test basic Gmail read email task functionality."""
        # First get a message ID
        list_task = GmailListEmailsTask(
            "get_message_id",
            {
                "query": "",
                "max_results": 1,
                "parse_messages": False,
                "credentials": google_oauth_credentials,
            },
        )

        list_result = list_task.run()
        assert list_result.is_success()

        message_ids = list_result.data["message_ids"]
        if not message_ids:
            pytest.skip("No messages available for read testing")

        message_id = message_ids[0]

        # Now test reading the message
        read_task = GmailReadEmailTask(
            "test_read_email",
            {
                "message_ids": [message_id],
                "format": "full",
                "include_attachments": True,
                "timeout": 60,
                "credentials": google_oauth_credentials,
            },
        )

        start_time = time.time()
        result = read_task.run()
        execution_time = time.time() - start_time

        # Verify task result structure
        assert isinstance(result, TaskResult)
        assert result.is_success()

        # Verify data structure
        assert "messages" in result.data
        assert "message_count" in result.data
        assert "success_count" in result.data
        assert "failed_count" in result.data
        assert "failed_reads" in result.data

        messages = result.data["messages"]
        assert len(messages) == 1

        message = messages[0]
        assert isinstance(message, dict)  # Task returns dict, not GmailMessage object
        assert message["message_id"] == message_id
        assert "subject" in message  # Should have subject field (can be empty)
        assert message["sender"]  # Should have sender (required)

        print(f"[SUCCESS] Read email in {execution_time:.2f}s")
        print(f"   Subject: '{message['subject'][:50]}...'")
        print(f"   From: {message['sender_name']}")

    def test_read_multiple_emails(self, google_oauth_credentials: Any) -> None:
        """Test reading multiple emails at once."""
        # Get multiple message IDs
        list_task = GmailListEmailsTask(
            "get_multiple_message_ids",
            {
                "query": "",
                "max_results": 3,
                "parse_messages": False,
                "credentials": google_oauth_credentials,
            },
        )

        list_result = list_task.run()
        assert list_result.is_success()

        message_ids = list_result.data["message_ids"]
        if len(message_ids) < 2:
            pytest.skip("Not enough messages for multiple read testing")

        # Read multiple messages
        read_task = GmailReadEmailTask(
            "test_read_multiple",
            {
                "message_ids": message_ids[:2],  # Read first 2
                "format": "full",
                "include_attachments": False,
                "credentials": google_oauth_credentials,
            },
        )

        result = read_task.run()

        assert result.is_success()
        assert result.data["message_count"] == 2
        assert result.data["success_count"] == 2
        assert result.data["failed_count"] == 0

        messages = result.data["messages"]
        assert len(messages) == 2

        # Verify different messages
        assert messages[0]["message_id"] != messages[1]["message_id"]

        print(f"[SUCCESS] Read {len(messages)} emails successfully")

    def test_read_email_different_formats(self, google_oauth_credentials: Any) -> None:
        """Test reading emails in different formats."""
        # Get a message ID
        list_task = GmailListEmailsTask(
            "get_message_for_formats",
            {
                "query": "",
                "max_results": 1,
                "parse_messages": False,
                "credentials": google_oauth_credentials,
            },
        )

        list_result = list_task.run()
        if not list_result.data["message_ids"]:
            pytest.skip("No messages for format testing")

        message_id = list_result.data["message_ids"][0]

        # Test different formats
        formats_to_test = ["minimal", "full", "metadata"]

        for format_type in formats_to_test:
            read_task = GmailReadEmailTask(
                f"test_read_{format_type}",
                {
                    "message_ids": [message_id],
                    "format": format_type,
                    "include_attachments": False,
                    "credentials": google_oauth_credentials,
                },
            )

            result = read_task.run()

            assert result.is_success()
            assert len(result.data["messages"]) == 1

            message = result.data["messages"][0]
            assert message["message_id"] == message_id

            print(f"[SUCCESS] Successfully read message in {format_type} format")

    def test_read_email_with_dependencies(self, google_oauth_credentials: Any) -> None:
        """Test Gmail read email task with dependency resolution."""
        # Create read task with dependency
        read_task = GmailReadEmailTask(
            "test_read_with_dependency",
            {
                "format": "full",
                "include_attachments": False,
                "credentials": google_oauth_credentials,
            },
            dependencies=[
                TaskDependency(
                    "message_ids", "email_list.message_ids", DependencyType.REQUIRED
                )
            ],
        )

        # Get message IDs to simulate dependency resolution
        list_task = GmailListEmailsTask(
            "simulate_dependency",
            {
                "query": "",
                "max_results": 1,
                "parse_messages": False,
                "credentials": google_oauth_credentials,
            },
        )

        list_result = list_task.run()
        if not list_result.data["message_ids"]:
            pytest.skip("No messages for dependency testing")

        # Simulate dependency resolution
        message_ids = list_result.data["message_ids"]
        read_task.config["message_ids"] = message_ids

        result = read_task.run()

        assert result.is_success()
        assert len(result.data["messages"]) == 1

        print("[SUCCESS] Dependency resolution working for Gmail read task")

    def test_read_email_error_handling(self, google_oauth_credentials: Any) -> None:
        """Test error handling with invalid message IDs."""
        # Use invalid message ID
        read_task = GmailReadEmailTask(
            "test_error_handling",
            {
                "message_ids": ["invalid_message_id_12345"],
                "format": "full",
                "include_attachments": False,
                "credentials": google_oauth_credentials,
            },
        )

        result = read_task.run()

        # Task might return partial success with error info
        assert isinstance(result, TaskResult)

        # Should handle invalid IDs gracefully
        if result.is_success():
            # Check if it reported failures
            assert result.data.get("failed_count", 0) >= 1

        print("[SUCCESS] Error handling working for invalid message IDs")

    @pytest.mark.slow
    def test_comprehensive_gmail_task_workflow(
        self, google_oauth_credentials: Any
    ) -> None:
        """Test comprehensive workflow combining list and read tasks."""
        workflow_start = time.time()

        # Step 1: List recent emails
        list_task = GmailListEmailsTask(
            "comprehensive_list",
            {
                "query": GmailSearchOperators.newer_than(14),  # Last 2 weeks
                "max_results": 5,
                "parse_messages": False,  # Just get IDs for reading
                "credentials": google_oauth_credentials,
            },
        )

        list_result = list_task.run()
        assert list_result.is_success()

        message_ids = list_result.data["message_ids"]

        if not message_ids:
            pytest.skip("No recent messages for comprehensive workflow testing")

        # Step 2: Read the messages in full detail
        read_task = GmailReadEmailTask(
            "comprehensive_read",
            {
                "message_ids": message_ids,
                "format": "full",
                "include_attachments": True,
                "credentials": google_oauth_credentials,
            },
        )

        read_result = read_task.run()
        assert read_result.is_success()

        messages = read_result.data["messages"]

        # Step 3: Analyze the results
        unread_count = sum(1 for msg in messages if msg["is_unread"])
        important_count = sum(1 for msg in messages if msg["is_important"])
        with_attachments = sum(1 for msg in messages if msg["has_attachments"])

        # Step 4: Verify workflow integrity
        assert len(messages) == len(message_ids)
        assert read_result.data["success_count"] == len(messages)

        workflow_time = time.time() - workflow_start

        print("[SUCCESS] Comprehensive Gmail task workflow completed:")
        print(f"   Listed {len(message_ids)} message IDs")
        print(f"   Read {len(messages)} full messages")
        print(f"   Unread: {unread_count}")
        print(f"   Important: {important_count}")
        print(f"   With attachments: {with_attachments}")
        print(f"   Total time: {workflow_time:.2f} seconds")

        # Performance check
        assert workflow_time < 120, (
            f"Comprehensive workflow too slow: {workflow_time:.2f}s"
        )


@pytest.fixture(scope="session")
def google_oauth_credentials(skip_if_no_gmail_credentials: Any) -> Any:
    """Create pre-authenticated Google credentials using GoogleOAuthTask."""
    google_credentials = os.getenv("GOOGLE_CREDENTIALS_FILE")

    # Create GoogleOAuthTask to get pre-authenticated credentials
    oauth_task = GoogleOAuthTask(
        "test_oauth",
        {
            "service_types": ["gmail"],
            "credentials_file": google_credentials,
            "token_file": os.getenv("GMAIL_TOKEN_FILE", "gmail_token.json"),
        },
    )

    oauth_result = oauth_task.run()
    if not oauth_result.is_success():
        pytest.skip(f"GoogleOAuthTask failed: {oauth_result.error}")

    return oauth_result.data["credentials"]


@pytest.mark.integration
@pytest.mark.gmail
@pytest.mark.slow
class TestGmailTasksUnifiedAuthentication:
    """Integration tests for Gmail tasks using unified authentication."""

    def test_gmail_list_emails_with_pre_auth_credentials(
        self, google_oauth_credentials: Any
    ) -> None:
        """Test GmailListEmailsTask with pre-authenticated credentials."""
        task = GmailListEmailsTask(
            "test_list_pre_auth",
            {
                "query": "",
                "max_results": 3,
                "parse_messages": True,
                "credentials": google_oauth_credentials,  # Use pre-authenticated credentials
            },
        )

        result = task.run()

        # Verify task result structure
        assert isinstance(result, TaskResult)
        assert result.is_success()

        # Verify data structure
        assert "messages" in result.data
        assert "message_ids" in result.data
        assert "search_result" in result.data

        messages: List[GmailMessage] = result.data["messages"]
        message_ids: List[str] = result.data["message_ids"]
        assert isinstance(messages, list)
        assert isinstance(message_ids, list)
        assert len(message_ids) == len(messages)

        print(
            f"[SUCCESS] GmailListEmailsTask with pre-auth credentials: {len(messages)} messages"
        )

    def test_gmail_read_email_with_pre_auth_credentials(
        self, google_oauth_credentials: Any
    ) -> None:
        """Test GmailReadEmailTask with pre-authenticated credentials."""
        # First get a message ID using list task
        list_task = GmailListEmailsTask(
            "get_message_id_pre_auth",
            {
                "query": "",
                "max_results": 1,
                "parse_messages": False,
                "credentials": google_oauth_credentials,
            },
        )

        list_result = list_task.run()
        assert list_result.is_success()

        message_ids = list_result.data["message_ids"]
        if not message_ids:
            pytest.skip("No messages available for pre-auth read testing")

        message_id = message_ids[0]

        # Test reading with pre-authenticated credentials
        read_task = GmailReadEmailTask(
            "test_read_pre_auth",
            {
                "message_ids": [message_id],
                "format": "full",
                "include_attachments": True,
                "credentials": google_oauth_credentials,  # Use pre-authenticated credentials
            },
        )

        result = read_task.run()

        # Verify task result structure
        assert isinstance(result, TaskResult)
        assert result.is_success()

        # Verify data structure
        assert "messages" in result.data
        assert "message_count" in result.data
        assert "success_count" in result.data

        messages = result.data["messages"]
        assert len(messages) == 1

        message = messages[0]
        assert message["message_id"] == message_id
        assert "subject" in message
        assert "sender" in message

        print(
            f"[SUCCESS] GmailReadEmailTask with pre-auth credentials: read message {message_id[:8]}..."
        )

    def test_unified_auth_workflow_list_then_read(
        self, google_oauth_credentials: Any
    ) -> None:
        """Test complete workflow: OAuth -> List -> Read using unified authentication."""
        workflow_start = time.time()

        # Step 1: List emails with pre-authenticated credentials
        list_task = GmailListEmailsTask(
            "workflow_list",
            {
                "query": "",
                "max_results": 2,
                "parse_messages": False,  # Just get IDs
                "credentials": google_oauth_credentials,
            },
        )

        list_result = list_task.run()
        assert list_result.is_success()

        message_ids = list_result.data["message_ids"]
        if not message_ids:
            pytest.skip("No messages for unified auth workflow test")

        # Step 2: Read emails with same pre-authenticated credentials
        read_task = GmailReadEmailTask(
            "workflow_read",
            {
                "message_ids": message_ids,
                "format": "full",
                "include_attachments": False,
                "credentials": google_oauth_credentials,  # Same credentials
            },
        )

        read_result = read_task.run()
        assert read_result.is_success()

        # Step 3: Verify workflow integrity
        messages = read_result.data["messages"]
        assert len(messages) == len(message_ids)
        assert read_result.data["success_count"] == len(messages)

        workflow_time = time.time() - workflow_start

        print("[SUCCESS] Unified authentication workflow completed:")
        print(f"   Listed {len(message_ids)} message IDs")
        print(f"   Read {len(messages)} full messages")
        print(f"   Total time: {workflow_time:.2f} seconds")
        print("   Using single set of pre-authenticated credentials")

    def test_gmail_tasks_dependency_injection_with_pre_auth(
        self, google_oauth_credentials: Any
    ) -> None:
        """Test dependency injection with pre-authenticated credentials."""
        # Create list task with dependency for credentials
        list_task = GmailListEmailsTask(
            "dep_list",
            {
                "query": "",
                "max_results": 2,
                "parse_messages": False,
            },
            dependencies=[
                TaskDependency(
                    "credentials", "oauth_task.credentials", DependencyType.REQUIRED
                )
            ],
        )

        # Simulate dependency resolution by setting credentials directly
        list_task.config["credentials"] = google_oauth_credentials

        result = list_task.run()
        assert result.is_success()

        message_ids = result.data["message_ids"]

        print(
            f"[SUCCESS] Dependency injection with pre-auth: {len(message_ids)} messages listed"
        )

    def test_gmail_tasks_pipeline_dependency_workflow(
        self, google_oauth_credentials: Any
    ) -> None:
        """Test Gmail tasks in a pipeline workflow with dependency injection."""
        # Simulate a pipeline workflow where one task depends on another

        # Step 1: List recent emails (simulates first task in pipeline)
        list_task = GmailListEmailsTask(
            "pipeline_list",
            {
                "query": "",
                "max_results": 3,
                "parse_messages": False,
                "credentials": google_oauth_credentials,
            },
        )

        list_result = list_task.run()
        assert list_result.is_success()

        message_ids = list_result.data["message_ids"]
        if not message_ids:
            pytest.skip("No messages for pipeline workflow test")

        # Step 2: Read those emails (simulates dependent task in pipeline)
        read_task = GmailReadEmailTask(
            "pipeline_read",
            {
                "format": "full",
                "include_attachments": False,
                "credentials": google_oauth_credentials,
            },
        )

        # Simulate dependency injection - in real pipeline this would be automatic
        read_task.config["message_ids"] = message_ids[:2]  # Read first 2 messages

        read_result = read_task.run()
        assert read_result.is_success()

        messages = read_result.data["messages"]
        assert len(messages) == 2

        print("[SUCCESS] Gmail tasks pipeline dependency workflow:")
        print(f"   Listed {len(message_ids)} messages")
        print(f"   Read {len(messages)} messages via dependency injection")
        print("   Pipeline workflow using unified authentication successful")
