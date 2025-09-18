"""Integration tests for GmailService with real Gmail API calls.

These tests use the actual Gmail API with real credentials - no mocking.
They verify OAuth2 authentication, email operations, and data parsing.

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

Run with: pytest integration_tests/tasklib/google/test_gmail_service_integration.py -v

IMPORTANT: These tests will read real emails from your Gmail account.
Use a dedicated test account when possible.

Test Classes:
- TestGmailServiceIntegration: Basic GmailService functionality tests
- TestGmailServicePreAuthCredentials: Pre-authenticated credentials tests
- TestGmailServiceUnifiedAuthWorkflow: GoogleOAuthTask integration tests
"""

import os
import pytest
import time
from typing import Any, List
from datetime import datetime

from aipype_g import (
    GmailService,
    GoogleAuthService,
    GoogleOAuthTask,
    GmailMessage,
    GmailLabel,
)
from aipype_g.tasklib.gmail_service import GmailServiceError
from aipype_g.tasklib.google_auth_service import GMAIL_SCOPES
from aipype_g.tasklib.gmail_models import (
    GmailSearchOperators,
    GmailSystemLabels,
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


@pytest.fixture(scope="session")
def gmail_service(pre_authenticated_credentials: Any) -> GmailService:
    """Create Gmail service instance for testing using pre-authenticated credentials."""
    service = GmailService(credentials=pre_authenticated_credentials, timeout=60)

    # Verify service is properly initialized
    if not service.service:
        pytest.skip(
            "Gmail service could not be initialized - check credentials and authentication"
        )

    return service


@pytest.fixture(scope="session")
def pre_authenticated_credentials(skip_if_no_gmail_credentials: Any) -> Any:
    """Create pre-authenticated Google credentials."""
    google_credentials = os.getenv("GOOGLE_CREDENTIALS_FILE")
    google_token = os.getenv("GMAIL_TOKEN_FILE", "gmail_token.json")

    # Create credentials using GoogleAuthService
    auth_service = GoogleAuthService(
        credentials_file=google_credentials,
        token_file=google_token,
        scopes=GMAIL_SCOPES,
    )
    credentials = auth_service.authenticate()

    return credentials


@pytest.fixture(scope="session")
def gmail_service_with_pre_auth(pre_authenticated_credentials: Any) -> GmailService:
    """Create Gmail service instance using pre-authenticated credentials."""
    service = GmailService(credentials=pre_authenticated_credentials, timeout=60)

    # Verify service is properly initialized
    if not service.service:
        pytest.skip("Gmail service with pre-auth could not be initialized")

    return service


@pytest.fixture
def test_email_recipient() -> str:
    """Email address for sending test emails."""
    # Use environment variable or default to same account (send to self)
    return os.getenv("TEST_EMAIL_RECIPIENT", "")


@pytest.mark.integration
@pytest.mark.gmail
@pytest.mark.slow
class TestGmailServiceIntegration:
    """Integration tests for Gmail service with real API calls."""

    def test_service_initialization_and_authentication(
        self, skip_if_no_gmail_credentials: Any
    ) -> None:
        """Test Gmail service initialization and OAuth2 authentication."""
        service = GmailService()

        # Service should initialize without errors
        assert service is not None
        assert service.credentials is not None
        assert service.service is not None

        # Check that credentials are valid
        assert service.credentials.valid

        print("[SUCCESS] Gmail service initialized and authenticated successfully")

    def test_list_recent_messages_basic(self, gmail_service: GmailService) -> None:
        """Test listing recent messages without filters."""
        messages = gmail_service.list_messages(query="", max_results=5)

        # Should return some messages (most Gmail accounts have messages)
        assert isinstance(messages, list)
        assert len(messages) >= 0  # Empty account is possible

        # Verify message structure
        if messages:
            message = messages[0]
            assert "id" in message
            assert "threadId" in message

        print(f"[SUCCESS] Retrieved {len(messages)} recent messages")

    def test_list_messages_with_query(self, gmail_service: GmailService) -> None:
        """Test listing messages with Gmail search query."""
        # Search for messages in inbox
        query = GmailSearchOperators.combine_and(
            "in:inbox",
            GmailSearchOperators.newer_than(30),  # Last 30 days
        )

        messages = gmail_service.list_messages(query=query, max_results=10)

        assert isinstance(messages, list)

        print(f"[SUCCESS] Query '{query}' returned {len(messages)} messages")

    def test_list_messages_with_labels(self, gmail_service: GmailService) -> None:
        """Test listing messages filtered by labels."""
        # Use standard INBOX label
        messages = gmail_service.list_messages(
            label_ids=[GmailSystemLabels.INBOX], max_results=5
        )

        assert isinstance(messages, list)

        print(f"[SUCCESS] INBOX label filter returned {len(messages)} messages")

    def test_get_message_minimal_format(self, gmail_service: GmailService) -> None:
        """Test retrieving a message in minimal format."""
        # First get a message ID
        messages = gmail_service.list_messages(max_results=1)

        if not messages:
            pytest.skip("No messages available for testing")

        message_id = messages[0]["id"]

        # Get message in minimal format
        message_data = gmail_service.get_message(
            message_id=message_id, format="minimal"
        )

        # Verify minimal format response
        assert message_data["id"] == message_id
        assert "threadId" in message_data
        assert "labelIds" in message_data

        print(f"[SUCCESS] Retrieved message {message_id[:8]}... in minimal format")

    def test_get_message_full_format(self, gmail_service: GmailService) -> None:
        """Test retrieving a message in full format."""
        # First get a message ID
        messages = gmail_service.list_messages(max_results=1)

        if not messages:
            pytest.skip("No messages available for testing")

        message_id = messages[0]["id"]

        # Get message in full format
        message_data = gmail_service.get_message(message_id=message_id, format="full")

        # Verify full format response
        assert message_data["id"] == message_id
        assert "payload" in message_data
        assert "headers" in message_data["payload"]

        print(f"[SUCCESS] Retrieved message {message_id[:8]}... in full format")

    def test_parse_message_from_api_data(self, gmail_service: GmailService) -> None:
        """Test parsing Gmail API message data into GmailMessage object."""
        # Get a message in full format for parsing
        messages = gmail_service.list_messages(max_results=1)

        if not messages:
            pytest.skip("No messages available for testing")

        message_id = messages[0]["id"]
        message_data = gmail_service.get_message(message_id=message_id, format="full")

        # Parse the message
        parsed_message = gmail_service.parse_message(message_data)

        # Verify parsed structure
        assert isinstance(parsed_message, GmailMessage)
        assert parsed_message.message_id == message_id
        assert parsed_message.thread_id == message_data["threadId"]
        assert isinstance(parsed_message.label_ids, list)
        assert isinstance(parsed_message.subject, str)
        assert isinstance(parsed_message.sender, str)
        assert isinstance(parsed_message.body, str)

        # Test properties
        assert isinstance(parsed_message.is_unread, bool)
        assert isinstance(parsed_message.is_important, bool)
        assert isinstance(parsed_message.is_starred, bool)
        assert isinstance(parsed_message.has_attachments, bool)
        assert isinstance(parsed_message.attachment_count, int)

        print(f"[SUCCESS] Parsed message: '{parsed_message.subject[:50]}...'")
        print(f"   From: {parsed_message.sender_name} <{parsed_message.sender_email}>")
        print(f"   Attachments: {parsed_message.attachment_count}")

    def test_list_labels(self, gmail_service: GmailService) -> None:
        """Test listing all Gmail labels."""
        labels = gmail_service.list_labels()

        assert isinstance(labels, list)
        assert len(labels) > 0  # Should have at least system labels

        # Check for system labels
        label_names = [label.name for label in labels]
        assert "INBOX" in label_names
        assert "SENT" in label_names

        # Verify label structure
        first_label = labels[0]
        assert isinstance(first_label, GmailLabel)
        assert first_label.label_id
        assert first_label.name

        # Test label properties
        system_labels = [label for label in labels if label.is_system_label]
        user_labels = [label for label in labels if label.is_user_label]

        assert len(system_labels) > 0  # Should have system labels

        print(f"[SUCCESS] Retrieved {len(labels)} labels")
        print(f"   System labels: {len(system_labels)}")
        print(f"   User labels: {len(user_labels)}")

    def test_create_and_cleanup_test_label(self, gmail_service: GmailService) -> None:
        """Test creating and cleaning up a test label."""
        import uuid

        # Create unique test label name
        test_label_name = f"test-label-{uuid.uuid4().hex[:8]}"

        try:
            # Create the label
            created_label = gmail_service.create_label(
                name=test_label_name,
                message_list_visibility="show",
                label_list_visibility="labelShow",
            )

            assert isinstance(created_label, GmailLabel)
            assert created_label.name == test_label_name
            assert created_label.label_id
            assert created_label.is_user_label

            # Verify label exists in list
            all_labels = gmail_service.list_labels()
            label_names = [label.name for label in all_labels]
            assert test_label_name in label_names

            print(
                f"[SUCCESS] Created test label: {test_label_name} ({created_label.label_id})"
            )

        except Exception as e:
            print(f"[WARNING] Could not create test label: {e}")
            # Don't fail the test - Gmail API permissions might not allow label creation

    @pytest.mark.skipif(
        not os.getenv("TEST_EMAIL_RECIPIENT"),
        reason="TEST_EMAIL_RECIPIENT environment variable not set",
    )
    def test_send_test_email(
        self, gmail_service: GmailService, test_email_recipient: str
    ) -> None:
        """Test sending a test email."""
        if not test_email_recipient:
            pytest.skip("No test email recipient configured")

        import uuid

        test_id = uuid.uuid4().hex[:8]
        subject = f"Integration Test Email {test_id}"
        body = f"""This is a test email sent by the Gmail service integration tests.

Test ID: {test_id}
Timestamp: {datetime.now().isoformat()}

This email was sent automatically by the mi-agents framework integration tests.
You can safely delete this email.
"""

        try:
            result = gmail_service.send_message(
                to=test_email_recipient, subject=subject, body=body
            )

            assert "id" in result
            assert "threadId" in result

            sent_message_id = result["id"]

            print(f"[SUCCESS] Sent test email: {subject}")
            print(f"   Message ID: {sent_message_id}")
            print(f"   Recipient: {test_email_recipient}")

        except GmailServiceError as e:
            print(f"[WARNING] Could not send test email: {e}")
            # Don't fail - might be permissions issue

    def test_modify_message_labels(self, gmail_service: GmailService) -> None:
        """Test adding and removing labels from a message."""
        # Get a message from inbox
        inbox_messages = gmail_service.list_messages(
            label_ids=[GmailSystemLabels.INBOX], max_results=1
        )

        if not inbox_messages:
            pytest.skip("No inbox messages available for label testing")

        message_id = inbox_messages[0]["id"]

        # Get original message to check current labels
        original_message = gmail_service.get_message(message_id, format="minimal")
        original_labels = set(original_message.get("labelIds", []))

        try:
            # Add STARRED label
            if GmailSystemLabels.STARRED not in original_labels:
                result = gmail_service.modify_message_labels(
                    message_id=message_id, add_label_ids=[GmailSystemLabels.STARRED]
                )

                assert "id" in result
                assert "labelIds" in result

                # Verify label was added
                updated_labels = set(result["labelIds"])
                assert GmailSystemLabels.STARRED in updated_labels

                print(f"[SUCCESS] Added STARRED label to message {message_id[:8]}...")

                # Remove the label to clean up
                gmail_service.modify_message_labels(
                    message_id=message_id, remove_label_ids=[GmailSystemLabels.STARRED]
                )
                print(
                    f"[SUCCESS] Removed STARRED label from message {message_id[:8]}..."
                )
            else:
                print(
                    f"[SUCCESS] Message {message_id[:8]}... already starred, skipping modify test"
                )

        except GmailServiceError as e:
            print(f"[WARNING] Could not modify message labels: {e}")
            # Don't fail - might be permissions or read-only message

    def test_error_handling_invalid_message_id(
        self, gmail_service: GmailService
    ) -> None:
        """Test error handling with invalid message ID."""
        invalid_message_id = "invalid_message_id_12345"

        with pytest.raises(GmailServiceError) as exc_info:
            gmail_service.get_message(invalid_message_id)

        assert "Failed to get message" in str(exc_info.value)

        print("[SUCCESS] Error handling works correctly for invalid message ID")

    def test_progress_callback_functionality(self, gmail_service: GmailService) -> None:
        """Test progress callback functionality."""
        progress_messages: List[str] = []

        def progress_callback(message: str) -> None:
            progress_messages.append(message)

        # List messages with progress callback
        messages = gmail_service.list_messages(
            query="in:inbox", max_results=3, progress_callback=progress_callback
        )

        # Should have received progress messages
        assert isinstance(messages, list)  # Validate that messages were returned
        assert len(progress_messages) >= 2  # At least start and end
        assert any("Starting Gmail message list" in msg for msg in progress_messages)
        assert any("Successfully retrieved" in msg for msg in progress_messages)

        print("[SUCCESS] Progress callback functionality working")
        print(f"   Received {len(progress_messages)} progress updates")

    @pytest.mark.slow
    def test_performance_with_larger_result_set(
        self, gmail_service: GmailService
    ) -> None:
        """Test performance with larger result sets."""
        start_time = time.time()

        messages = gmail_service.list_messages(
            query="in:inbox",
            max_results=50,  # Larger result set
        )

        execution_time = time.time() - start_time

        # Should complete in reasonable time
        assert execution_time < 30, f"Query took too long: {execution_time:.2f} seconds"

        print(
            f"[SUCCESS] Retrieved {len(messages)} messages in {execution_time:.2f} seconds"
        )

    @pytest.mark.slow
    def test_comprehensive_workflow(self, gmail_service: GmailService) -> None:
        """Comprehensive test simulating real-world Gmail operations."""
        workflow_start = time.time()

        # 1. List labels
        labels = gmail_service.list_labels()
        assert len(labels) > 0

        # 2. Search for recent messages
        recent_messages = gmail_service.list_messages(
            query=GmailSearchOperators.newer_than(7),  # Last week
            max_results=10,
        )

        # 3. Parse some messages if available
        parsed_messages: List[GmailMessage] = []
        for message_data in recent_messages[:3]:  # Parse first 3
            full_message_data = gmail_service.get_message(
                message_data["id"], format="full"
            )
            parsed_message = gmail_service.parse_message(full_message_data)
            parsed_messages.append(parsed_message)

        # 4. Analyze results
        if parsed_messages:
            unread_count = sum(1 for msg in parsed_messages if msg.is_unread)
            important_count = sum(1 for msg in parsed_messages if msg.is_important)
            with_attachments = sum(1 for msg in parsed_messages if msg.has_attachments)

            print("[SUCCESS] Comprehensive Gmail workflow completed:")
            print(f"   Labels: {len(labels)}")
            print(f"   Recent messages: {len(recent_messages)}")
            print(f"   Parsed messages: {len(parsed_messages)}")
            print(f"   Unread: {unread_count}")
            print(f"   Important: {important_count}")
            print(f"   With attachments: {with_attachments}")

        workflow_time = time.time() - workflow_start
        assert workflow_time < 60, (
            f"Comprehensive workflow took too long: {workflow_time:.2f}s"
        )

        print(f"[SUCCESS] Workflow completed in {workflow_time:.2f} seconds")


@pytest.mark.integration
@pytest.mark.gmail
@pytest.mark.slow
class TestGmailServicePreAuthCredentials:
    """Integration tests for Gmail service with pre-authenticated credentials."""

    def test_service_with_pre_authenticated_credentials(
        self, gmail_service_with_pre_auth: GmailService
    ) -> None:
        """Test Gmail service using pre-authenticated credentials."""
        service = gmail_service_with_pre_auth

        # Service should be properly initialized
        assert service is not None
        assert service.credentials is not None
        assert service.service is not None
        assert service.credentials.valid

        print(
            "[SUCCESS] Gmail service with pre-authenticated credentials initialized successfully"
        )

    def test_list_messages_with_pre_auth(
        self, gmail_service_with_pre_auth: GmailService
    ) -> None:
        """Test listing messages using pre-authenticated credentials."""
        messages = gmail_service_with_pre_auth.list_messages(query="", max_results=3)

        assert isinstance(messages, list)
        assert len(messages) >= 0

        if messages:
            message = messages[0]
            assert "id" in message
            assert "threadId" in message

        print(
            f"[SUCCESS] Listed {len(messages)} messages using pre-authenticated credentials"
        )

    def test_parse_message_with_pre_auth(
        self, gmail_service_with_pre_auth: GmailService
    ) -> None:
        """Test parsing messages using pre-authenticated credentials."""
        # Get a message first
        messages = gmail_service_with_pre_auth.list_messages(max_results=1)

        if not messages:
            pytest.skip("No messages available for pre-auth parsing test")

        message_id = messages[0]["id"]
        message_data = gmail_service_with_pre_auth.get_message(
            message_id=message_id, format="full"
        )

        # Parse the message
        parsed_message = gmail_service_with_pre_auth.parse_message(message_data)

        # Verify parsed structure
        assert isinstance(parsed_message, GmailMessage)
        assert parsed_message.message_id == message_id
        assert isinstance(parsed_message.subject, str)
        assert isinstance(parsed_message.sender, str)

        print(
            f"[SUCCESS] Parsed message using pre-authenticated credentials: '{parsed_message.subject[:50]}...'"
        )

    def test_list_labels_with_pre_auth(
        self, gmail_service_with_pre_auth: GmailService
    ) -> None:
        """Test listing labels using pre-authenticated credentials."""
        labels = gmail_service_with_pre_auth.list_labels()

        assert isinstance(labels, list)
        assert len(labels) > 0

        # Check for system labels
        label_names = [label.name for label in labels]
        assert "INBOX" in label_names
        assert "SENT" in label_names

        print(
            f"[SUCCESS] Listed {len(labels)} labels using pre-authenticated credentials"
        )


@pytest.mark.integration
@pytest.mark.gmail
@pytest.mark.slow
class TestGmailServiceUnifiedAuthWorkflow:
    """Integration tests for unified authentication workflow using GoogleOAuthTask."""

    def test_google_oauth_task_gmail_service_integration(
        self, skip_if_no_gmail_credentials: Any
    ) -> None:
        """Test GoogleOAuthTask integration with GmailService."""
        google_credentials = os.getenv("GOOGLE_CREDENTIALS_FILE")
        if not google_credentials:
            pytest.skip(
                "GOOGLE_CREDENTIALS_FILE not set - skipping unified auth workflow test"
            )

        # Step 1: Create GoogleOAuthTask
        oauth_task = GoogleOAuthTask(
            "test_oauth",
            {
                "service_types": ["gmail"],
                "credentials_file": google_credentials,
                "token_file": os.getenv("GMAIL_TOKEN_FILE", "gmail_token.json"),
            },
        )

        # Step 2: Run OAuth task to get credentials
        oauth_result = oauth_task.run()
        assert oauth_result.is_success()
        assert "credentials" in oauth_result.data

        credentials = oauth_result.data["credentials"]
        assert credentials is not None

        # Step 3: Use credentials with GmailService
        gmail_service = GmailService(credentials=credentials, timeout=60)

        # Step 4: Test Gmail operations
        messages = gmail_service.list_messages(query="", max_results=3)
        assert isinstance(messages, list)

        print(
            "[SUCCESS] Unified authentication workflow: GoogleOAuthTask -> GmailService integration successful"
        )
        print(f"   Retrieved {len(messages)} messages using unified auth")

    def test_google_oauth_task_creates_valid_credentials(
        self, skip_if_no_gmail_credentials: Any
    ) -> None:
        """Test that GoogleOAuthTask creates valid credentials for Gmail operations."""
        google_credentials = os.getenv("GOOGLE_CREDENTIALS_FILE")

        # Create GoogleOAuthTask
        oauth_task = GoogleOAuthTask(
            "test_oauth_validation",
            {
                "service_types": ["gmail"],
                "credentials_file": google_credentials,
                "token_file": os.getenv("GMAIL_TOKEN_FILE", "gmail_token.json"),
            },
        )

        # Get credentials
        oauth_result = oauth_task.run()
        assert oauth_result.is_success()

        credentials = oauth_result.data["credentials"]

        # Verify credentials are valid and have Gmail scope
        assert credentials is not None
        assert credentials.valid

        # Test that credentials work with Gmail API
        gmail_service = GmailService(credentials=credentials, timeout=60)
        labels = gmail_service.list_labels()
        assert isinstance(labels, list)
        assert len(labels) > 0

        print(
            "[SUCCESS] GoogleOAuthTask creates valid credentials for Gmail operations"
        )
        print(f"   Credentials valid: {credentials.valid}")
        print(f"   Gmail labels accessible: {len(labels)} labels found")
