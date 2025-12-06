"""Gmail Test Agent using @task decorators.

This agent demonstrates the declarative syntax for the aipype framework,
using @task decorators for automatic dependency inference and cleaner code.

Features:
1. Unified Google OAuth2 authentication (supports Gmail + Sheets + other services)
2. Testing Gmail API connectivity and basic operations
3. Listing recent emails with configurable queries
4. Counting unread emails
5. Comprehensive test result reporting

To run this example:
1. Set up Google API credentials (supports Gmail and Sheets APIs)
2. Set environment variables for Google authentication
3. Run: python examples/gmail_test_agent_declarative.py

Environment Variables:
- GOOGLE_CREDENTIALS_FILE: Path to Google credentials JSON file
- GMAIL_TOKEN_FILE: Path to Gmail OAuth2 token file (auto-generated)
- GMAIL_TEST_QUERY: Gmail search query (default: newer_than:7d)
- GMAIL_TEST_MAX_EMAILS: Max emails to test (default: 5)
- GMAIL_TEST_TIMEOUT: API timeout in seconds (default: 30)

This is a read-only test agent - it won't modify any emails.
"""

import os
import sys
from typing import Any, Dict, List

from aipype import (
    PipelineAgent,
    task,
    print_header,
    print_message_box,
)
from aipype_g import GoogleOAuthTask, GmailListEmailsTask


class GmailTestAgent(PipelineAgent):
    """Gmail test agent using @task decorators.

    This agent showcases the simplified declarative syntax where:
    - Tasks are defined as methods decorated with @task
    - Dependencies are inferred from parameter names
    - No manual TaskDependency declarations needed

    Pipeline Flow:
    1. gmail_auth - OAuth2 authentication
    2. test_list_emails - List recent emails using credentials
    3. count_unread_emails - Count unread emails using credentials
    4. generate_report - Generate comprehensive report from results
    """

    @task
    def gmail_auth(self) -> GoogleOAuthTask:
        """Authenticate with Google Gmail API."""
        return GoogleOAuthTask(
            "gmail_auth",
            {
                "service_types": ["gmail"],
                "credentials_file": self.config.get("google_credentials_file"),
                "token_file": self.config.get("google_token_file"),
            },
        )

    @task
    def test_list_emails(self, gmail_auth: Dict[str, Any]) -> GmailListEmailsTask:
        """List recent emails using authenticated credentials."""
        return GmailListEmailsTask(
            "test_list_emails",
            {
                "query": self.config.get("test_query", "newer_than:7d"),
                "max_results": self.config.get("max_emails", 5),
                "parse_messages": False,  # Skip parsing for faster performance
                "timeout": self.config.get("api_timeout", 30),
                "credentials": gmail_auth["credentials"],
            },
        )

    @task
    def count_unread_emails(self, gmail_auth: Dict[str, Any]) -> GmailListEmailsTask:
        """Count unread emails using authenticated credentials."""
        return GmailListEmailsTask(
            "count_unread_emails",
            {
                "query": "is:unread",
                "max_results": self.config.get("max_unread_check", 25),
                "parse_messages": False,  # Just get IDs for counting
                "timeout": self.config.get("api_timeout", 30),
                "credentials": gmail_auth["credentials"],
            },
        )

    @task
    def generate_report(
        self,
        test_list_emails: Dict[str, Any],
        count_unread_emails: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate and display test report from email listing results."""
        # Extract data from task results
        total_found = test_list_emails.get("retrieved_count", 0)
        query_used = test_list_emails.get("query_used", "")
        messages = test_list_emails.get("messages", [])
        message_ids = test_list_emails.get("message_ids", [])
        unread_count = len(count_unread_emails.get("message_ids", []))

        print()
        print_header("GMAIL API TEST RESULTS")

        # Display test results
        print_message_box(
            "GMAIL API TEST RESULTS",
            [
                "GMAIL AUTHENTICATION: SUCCESS",
                f"Query used: '{query_used}'",
                f"Total emails found: {total_found}",
                f"Unread emails: {unread_count}",
            ],
        )

        # Debug info display
        debug_content = [
            f"Messages parsed: {len(messages)}",
            f"Message IDs found: {len(message_ids)}",
            f"Parse messages was: {test_list_emails.get('parse_messages', 'unknown')}",
        ]
        print_message_box("DEBUG INFO", debug_content)

        # Sample emails display
        sample_content: List[str] = []
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

        if sample_content:
            print_message_box("EMAIL DETAILS", sample_content)

        # Test summary
        summary_content = [
            "Authentication: SUCCESS",
            f"Email listing: SUCCESS ({total_found} found)",
            f"Unread counting: SUCCESS ({unread_count} unread)",
            "API Status: Email listing successful - API connection working",
        ]
        print_message_box("TEST SUMMARY", summary_content)

        # Recommendations
        recommendations: List[str] = []
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

        print_message_box("RECOMMENDATIONS", recommendations)

        return {
            "test_passed": True,
            "total_emails": total_found,
            "unread_count": unread_count,
            "authentication_success": True,
            "sample_emails": len(messages),
        }


def main() -> None:
    """Run the Gmail test agent."""
    print_header("Gmail API Test Agent (Declarative Syntax)")

    # Configuration
    config = {
        # Google API settings
        "google_credentials_file": os.getenv(
            "GOOGLE_CREDENTIALS_FILE", "google_credentials.json"
        ),
        "google_token_file": os.getenv("GMAIL_TOKEN_FILE", "gmail_token.json"),
        # Test settings
        "test_query": os.getenv("GMAIL_TEST_QUERY", "newer_than:7d"),
        "max_emails": int(os.getenv("GMAIL_TEST_MAX_EMAILS", "5")),
        "max_unread_check": int(os.getenv("GMAIL_TEST_MAX_UNREAD", "25")),
        "api_timeout": int(os.getenv("GMAIL_TEST_TIMEOUT", "30")),
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
        results = agent.run()

        # Display results
        agent.display_results()

        # Display final summary
        if results and results.is_success():
            final_report = agent.context.get_result("generate_report")

            if final_report and final_report.get("test_passed"):
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
                "- Google credentials file not found or invalid",
                "  Check GOOGLE_CREDENTIALS_FILE path",
                "",
                "- Gmail API not enabled in Google Cloud Console",
                "  Enable Gmail API in your Google Cloud project",
                "",
                "- Invalid OAuth2 token",
                f"  Delete {config['google_token_file']} to force re-auth",
            ],
        )

        import traceback

        print_header("FULL ERROR TRACEBACK")
        traceback.print_exc()

        sys.exit(1)

    print_header("Gmail test agent finished!")


if __name__ == "__main__":
    main()
