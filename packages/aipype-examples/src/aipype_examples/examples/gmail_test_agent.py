"""Gmail Test Agent - Modern agent using unified Google authentication for Gmail API testing.

This agent demonstrates the modern MI-Agents approach using GoogleOAuthTask for unified
Google service authentication, supporting Gmail and future Google services like Sheets.

Features:
1. Unified Google OAuth2 authentication (supports Gmail + Sheets + other services)
2. Testing Gmail API connectivity and basic operations
3. Listing recent emails with configurable queries
4. Counting unread emails
5. Comprehensive test result reporting

To run this example:
1. Set up Google API credentials (supports Gmail and Sheets APIs)
2. Set environment variables for Google authentication
3. Run: python examples/gmail_test_agent.py

Environment Variables:
- GOOGLE_CREDENTIALS_FILE: Path to Google credentials JSON file
- GMAIL_TOKEN_FILE: Path to Gmail OAuth2 token file (auto-generated)
- GMAIL_TEST_QUERY: Gmail search query (default: newer_than:7d)
- GMAIL_TEST_MAX_EMAILS: Max emails to test (default: 5)
- GMAIL_TEST_TIMEOUT: API timeout in seconds (default: 30)

Architecture:
This agent uses GoogleOAuthTask for authentication, which provides:
- Unified authentication for multiple Google services
- Credential sharing across tasks via dependency injection
- Support for future expansion to Google Sheets, Drive, etc.
- Modern OAuth2 flow with proper error handling

For detailed logging, configure Python logging level to DEBUG.

This is a read-only test agent - it won't modify any emails.
"""

import os
import sys
from typing import Dict, Any, List, Optional, cast, override

from aipype import (
    PipelineAgent,
    BaseTask,
    TaskDependency,
    DependencyType,
    TransformTask,
    ConditionalTask,
    TaskResult,
)
from aipype_g import GoogleOAuthTask, GmailListEmailsTask
from aipype import print_header, print_message_box


class GmailTestReportTask(BaseTask):
    """Task that generates a test report based on Gmail operations."""

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        dependencies: Optional[List[TaskDependency]] = None,
    ):
        """Initialize Gmail test report task."""
        super().__init__(name, config)
        self.dependencies = dependencies or []

    @override
    def get_dependencies(self) -> List[TaskDependency]:
        """Get the list of task dependencies."""
        return self.dependencies

    @override
    def run(self) -> TaskResult:
        """Generate and display test report."""
        from datetime import datetime

        start_time = datetime.now()

        self.logger.debug("Starting test report generation...")
        self.logger.debug(f"Config received: {list(self.config.keys())}")

        # Get data from dependencies - reconstruct email_data from individual fields
        email_data = {
            "retrieved_count": self.config.get("email_data", 0),
            "query_used": self.config.get("query_used", ""),
            "messages": self.config.get("messages", []),
            "message_ids": self.config.get("message_ids", []),
            "parse_messages": self.config.get("parse_messages", False),
        }
        unread_count = self.config.get("unread_count", 0)
        error_info = self.config.get("error_info")
        check_result = self.config.get("check_result", "Status unknown")

        self.logger.debug(f"Email data keys: {list(email_data.keys())}")
        self.logger.debug(f"Email data type: {type(email_data)}")
        self.logger.debug(f"Unread count: {unread_count} (type: {type(unread_count)})")
        self.logger.debug(f"Error info: {error_info}")
        self.logger.debug(f"Check result: {check_result}")

        print()  # Add newline before
        print_header("GMAIL API TEST RESULTS")

        if error_info:
            print_message_box(
                "AUTHENTICATION/API ERROR",
                [
                    f"Error: {error_info}",
                    "",
                    "Troubleshooting:",
                    "1. Check your Google credentials file exists (supports Gmail + Sheets)",
                    "2. Verify OAuth2 setup in Google Cloud Console",
                    "3. Ensure Gmail API is enabled",
                    "4. Check environment variable GOOGLE_CREDENTIALS_FILE",
                    "5. GoogleOAuthTask provides unified authentication for Google services",
                ],
            )

            execution_time = (datetime.now() - start_time).total_seconds()
            return TaskResult.failure(
                error_message=f"Gmail test failed: {error_info}",
                execution_time=execution_time,
                metadata={"test_status": "failed", "error": error_info},
            )

        # Display test results
        total_found = email_data.get("retrieved_count", 0)
        query_used = email_data.get("query_used", "")

        self.logger.debug(
            f"Processing results: total_found={total_found}, query='{query_used}'"
        )

        print_message_box(
            "GMAIL API TEST RESULTS",
            [
                "GMAIL AUTHENTICATION: SUCCESS",
                f"Query used: '{query_used}'",
                f"Total emails found: {total_found}",
                f"Unread emails: {unread_count}",
            ],
        )

        # Email details (only if messages were parsed)
        messages = email_data.get("messages", [])
        message_ids = email_data.get("message_ids", [])

        # Log debug information
        self.logger.debug(f"Messages parsed: {len(messages)}")
        self.logger.debug(f"Message IDs found: {len(message_ids)}")
        self.logger.debug(
            f"Parse messages was: {email_data.get('parse_messages', 'unknown')}"
        )

        # Debug info display
        debug_content = [
            f"Messages parsed: {len(messages)}",
            f"Message IDs found: {len(message_ids)}",
            f"Parse messages was: {email_data.get('parse_messages', 'unknown')}",
        ]

        # Sample emails display
        sample_content: list[str] = []
        if messages and len(messages) > 0:
            sample_content.append(
                f"SAMPLE EMAILS (showing first {min(3, len(messages))}):"
            )
            for i, msg in enumerate(messages[:3], 1):
                subject = msg.get("subject", "No Subject")[:50]
                sender_email = msg.get("sender_email", "Unknown")
                is_unread = msg.get("is_unread", False)
                status = "UNREAD" if is_unread else "READ"
                sample_content.append(f"{i}. [{status}] {subject}...")
                sample_content.append(f"   From: {sender_email}")
        elif message_ids and len(message_ids) > 0:
            sample_content.append(
                f"MESSAGE IDS FOUND (first {min(3, len(message_ids))}):"
            )
            for i, msg_id in enumerate(message_ids[:3], 1):
                sample_content.append(f"{i}. {msg_id}")
        else:
            sample_content.append(
                "No message details available (messages not parsed for speed)"
            )

        print_message_box("DEBUG INFO", debug_content)
        if sample_content:
            print_message_box("EMAIL DETAILS", sample_content)

        # Test summary with detailed status
        summary_content = [
            "Authentication: SUCCESS",
            f"Email listing: SUCCESS ({total_found} found)",
            f"Unread counting: SUCCESS ({unread_count} unread)",
            f"API Status: {check_result}",
        ]

        # Log detailed debug information
        self.logger.debug(f"Email data structure: {type(email_data)}")
        self.logger.debug(f"Data keys: {list(email_data.keys())}")
        self.logger.debug(
            f"Parse messages mode: {email_data.get('parse_messages', 'unknown')}"
        )
        self.logger.debug(
            f"Search metadata: {email_data.get('search_metadata', 'none')}"
        )

        verbose_content = [
            f"Email data structure: {type(email_data)}",
            f"Data keys: {list(email_data.keys())}",
            f"Parse messages mode: {email_data.get('parse_messages', 'unknown')}",
            f"Search metadata: {email_data.get('search_metadata', 'none')}",
        ]

        # Recommendations
        recommendations: list[str] = []
        if unread_count == 0:
            recommendations.append(
                "No unread emails found - your inbox is clean! [CLEAN]"
            )
        elif unread_count > 20:
            recommendations.append("High unread count - consider email automation")
        else:
            recommendations.append("Normal unread count - email management is good")

        if total_found == 0:
            recommendations.append(
                "No recent emails found - try adjusting the search query"
            )

        print_message_box("TEST SUMMARY", summary_content)
        print_message_box("VERBOSE DEBUG INFO", verbose_content)
        print_message_box("RECOMMENDATIONS", recommendations)

        execution_time = (datetime.now() - start_time).total_seconds()
        return TaskResult.success(
            data={
                "test_passed": True,
                "total_emails": total_found,
                "unread_count": unread_count,
                "authentication_success": True,
                "sample_emails": len(messages),
            },
            execution_time=execution_time,
            metadata={
                "test_status": "passed",
                "total_found": total_found,
                "unread_count": unread_count,
            },
        )


class GmailTestAgent(PipelineAgent):
    """Modern Gmail test agent demonstrating unified Google authentication.

    This agent showcases the MI-Agents framework's unified Google authentication
    approach using GoogleOAuthTask for credential management across multiple
    Google services (Gmail, Sheets, Drive, etc.).

    Pipeline Flow:
    1. GoogleOAuthTask - Unified OAuth2 authentication
    2. GmailListEmailsTask - List recent emails using shared credentials
    3. GmailListEmailsTask - Count unread emails using shared credentials
    4. TransformTask - Extract unread count
    5. ConditionalTask - Validate results
    6. GmailTestReportTask - Generate comprehensive report
    """

    @staticmethod
    def _count_unread_emails(data: Any) -> int:
        """Count unread emails from message IDs list."""
        if isinstance(data, list):
            return len(cast(List[str], data))
        return 0

    @staticmethod
    def _check_retrieved_count(retrieved_count: int) -> bool:
        """Check if email retrieval was successful."""
        return retrieved_count >= 0

    @override
    def setup_tasks(self) -> List[Any]:
        """Set up the Gmail test pipeline tasks declaratively.

        Returns:
            List of configured tasks with dependencies
        """
        # Create and return all tasks
        return [
            self._create_gmail_auth_task(),
            self._create_test_list_emails_task(),
            self._create_count_unread_emails_task(),
            self._create_extract_unread_count_task(),
            self._create_check_email_results_task(),
            self._create_generate_report_task(),
        ]

    def _create_gmail_auth_task(self) -> GoogleOAuthTask:
        """Create the Gmail authentication task.

        Returns:
            Configured GoogleOAuthTask for Gmail authentication
        """
        return GoogleOAuthTask(
            "gmail_auth",
            {
                "service_types": ["gmail"],
                "credentials_file": self.config.get("google_credentials_file"),
                "token_file": self.config.get("google_token_file"),
            },
        )

    def _create_test_list_emails_task(self) -> GmailListEmailsTask:
        """Create the test list emails task.

        Returns:
            Configured GmailListEmailsTask for testing email listing
        """
        return GmailListEmailsTask(
            "test_list_emails",
            {
                "query": self.config.get("test_query", "newer_than:7d"),  # Last 7 days
                "max_results": self.config.get(
                    "max_emails", 5
                ),  # Reduced for faster testing
                "parse_messages": False,  # Skip parsing for faster performance
                "timeout": self.config.get("api_timeout", 30),  # Configurable timeout
            },
            [
                TaskDependency(
                    "credentials",
                    "gmail_auth.credentials",
                    DependencyType.REQUIRED,
                )
            ],
        )

    def _create_count_unread_emails_task(self) -> GmailListEmailsTask:
        """Create the count unread emails task.

        Returns:
            Configured GmailListEmailsTask for counting unread emails
        """
        return GmailListEmailsTask(
            "count_unread_emails",
            {
                "query": "is:unread",
                "max_results": self.config.get(
                    "max_unread_check", 25
                ),  # Reduced from 50 for speed
                "parse_messages": False,  # Just get IDs for counting
                "timeout": self.config.get("api_timeout", 30),  # Configurable timeout
            },
            [
                TaskDependency(
                    "credentials",
                    "gmail_auth.credentials",
                    DependencyType.REQUIRED,
                )
            ],
        )

    def _create_extract_unread_count_task(self) -> TransformTask:
        """Create the extract unread count task.

        Returns:
            Configured TransformTask for extracting unread count
        """
        return TransformTask(
            "extract_unread_count",
            {
                "transform_function": self._count_unread_emails,
                "output_name": "unread_count",
            },
            [
                TaskDependency(
                    "data",
                    "count_unread_emails.message_ids",
                    DependencyType.REQUIRED,
                )
            ],
        )

    def _create_check_email_results_task(self) -> ConditionalTask:
        """Create the check email results task.

        Returns:
            Configured ConditionalTask for validating email listing results
        """
        return ConditionalTask(
            "check_email_results",
            {
                "condition_function": self._check_retrieved_count,
                "condition_inputs": ["retrieved_count"],
                "true_action": lambda: "Email listing successful - API connection working",
                "false_action": lambda: "Email listing failed - check API credentials and connection",
                "action_field": "status_message",
            },
            [
                TaskDependency(
                    "retrieved_count",
                    "test_list_emails.retrieved_count",
                    DependencyType.REQUIRED,
                )
            ],
        )

    def _create_generate_report_task(self) -> GmailTestReportTask:
        """Create the generate report task.

        Returns:
            Configured GmailTestReportTask for generating test report
        """
        return GmailTestReportTask(
            "generate_report",
            {
                # Data will be populated by dependencies
            },
            [
                TaskDependency(
                    "email_data",
                    "test_list_emails.retrieved_count",
                    DependencyType.REQUIRED,
                ),
                TaskDependency(
                    "query_used",
                    "test_list_emails.query_used",
                    DependencyType.REQUIRED,
                ),
                TaskDependency(
                    "messages", "test_list_emails.messages", DependencyType.REQUIRED
                ),
                TaskDependency(
                    "message_ids",
                    "test_list_emails.message_ids",
                    DependencyType.REQUIRED,
                ),
                TaskDependency(
                    "parse_messages",
                    "test_list_emails.parse_messages",
                    DependencyType.REQUIRED,
                ),
                TaskDependency(
                    "unread_count",
                    "extract_unread_count.unread_count",
                    DependencyType.REQUIRED,
                ),
                TaskDependency(
                    "check_result",
                    "check_email_results.status_message",
                    DependencyType.OPTIONAL,
                ),
            ],
        )


def main() -> None:
    """Run the Gmail test agent."""
    print_header("Gmail API Test Agent")

    # Configuration with backward compatibility
    config = {
        # Google API settings (unified credentials for Gmail, Sheets, etc.)
        "google_credentials_file": os.getenv(
            "GOOGLE_CREDENTIALS_FILE", "google_credentials.json"
        ),
        "google_token_file": os.getenv("GMAIL_TOKEN_FILE", "gmail_token.json"),
        # Test settings
        "test_query": os.getenv("GMAIL_TEST_QUERY", "newer_than:7d"),  # Last 7 days
        "max_emails": int(os.getenv("GMAIL_TEST_MAX_EMAILS", "5")),  # Reduced from 10
        "max_unread_check": int(
            os.getenv("GMAIL_TEST_MAX_UNREAD", "25")
        ),  # Max unread to check
        "api_timeout": int(
            os.getenv("GMAIL_TEST_TIMEOUT", "30")
        ),  # API timeout in seconds
    }

    print_message_box(
        "CONFIGURATION",
        [
            f"Credentials file: {config['google_credentials_file']}",
            f"Token file: {config['google_token_file']}",
            f"Test query: {config['test_query']}",
            f"Max emails: {config['max_emails']}",
            f"Max unread check: {config['max_unread_check']}",
            f"API timeout: {config['api_timeout']}s",
        ],
    )

    try:
        # Create and run the agent
        agent = GmailTestAgent("gmail_test_agent", config)
        agent.setup_tasks()
        results = agent.run()

        # Display results using framework method
        agent.display_results()

        # Display final summary - access task result through agent context
        if results and results.get("status") == "completed":
            # Get the final report task result from the agent's context
            final_report_result = agent.context.get_result("generate_report")

            # The context.get_result() returns the task data directly, not wrapped in a "data" field
            if final_report_result and final_report_result.get("test_passed"):
                print_message_box(
                    "SUCCESS",
                    [
                        "Gmail API test completed successfully!",
                        "Your Gmail integration is working correctly.",
                    ],
                )
            else:
                print_message_box(
                    "WARNING",
                    [
                        "Gmail API test completed with issues.",
                        "Check the error details above for troubleshooting.",
                    ],
                )
        else:
            print_message_box("ERROR", ["No results returned from Gmail test agent"])
            sys.exit(1)

    except Exception as e:
        print_message_box(
            "EXCEPTION",
            [
                f"Gmail test agent failed with exception: {e}",
                "",
                "Exception Details:",
                f"- Exception type: {type(e).__name__}",
                f"- Exception message: {str(e)}",
            ],
        )

        print_message_box(
            "TROUBLESHOOTING",
            [
                "Common issues and solutions:",
                "",
                "• Google credentials file not found or invalid",
                "  → Check GOOGLE_CREDENTIALS_FILE path and file existence",
                "",
                "• Gmail API not enabled in Google Cloud Console",
                "  → Enable Gmail API in your Google Cloud project",
                "",
                "• OAuth2 consent screen not configured",
                "  → Complete OAuth consent screen setup for external users",
                "",
                "• Unified authentication issues",
                "  → GoogleOAuthTask handles authentication for Gmail and future Google services",
                "",
                "• Network connectivity or timeout issues",
                f"  → Check internet connection, try increasing timeout (current: {config['api_timeout']}s)",
                "",
                "• Invalid OAuth2 token",
                f"  → Delete {config['google_token_file']} to force re-authentication",
            ],
        )

        import traceback

        print_header("FULL ERROR TRACEBACK")
        traceback.print_exc()

        sys.exit(1)

    print_header("Gmail test agent finished!")


if __name__ == "__main__":
    main()
