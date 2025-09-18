"""Gmail List Emails Task - List Gmail messages with filters and queries."""

from typing import List, Dict, Any, Optional

from typing import override

from aipype.framework.base_task import BaseTask
from aipype.framework.task_dependencies import TaskDependency
from aipype.framework.task_result import TaskResult
from .gmail_service import GmailService, GmailServiceError
from .gmail_models import GmailMessage, GmailSearchResult, GmailSearchOperators


class GmailListEmailsTask(BaseTask):
    """Task that lists Gmail messages with optional filters and search queries."""

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        dependencies: Optional[List[TaskDependency]] = None,
    ):
        """Initialize Gmail list emails task.

        Args:
            name: Task name
            config: Configuration dictionary containing:
                - query: Gmail search query (optional, can be resolved from dependencies)
                - max_results: Maximum number of messages to return (default: 10)
                - include_spam_trash: Include spam and trash messages (default: False)
                - label_ids: List of label IDs to filter by (optional)
                - credentials_file: Path to OAuth2 credentials file (optional)
                - token_file: Path to OAuth2 token file (optional)
                - parse_messages: Whether to parse full message content (default: True)
            dependencies: List of task dependencies
        """
        super().__init__(name, config)
        self.dependencies = dependencies or []
        self.validation_rules = {
            "defaults": {
                "query": "",
                "max_results": 10,
                "include_spam_trash": False,
                "parse_messages": True,
                "timeout": 30,
            },
            "types": {
                "query": str,
                "max_results": int,
                "include_spam_trash": bool,
                "label_ids": list,
                "credentials_file": str,
                "token_file": str,
                "parse_messages": bool,
                "timeout": int,
                # Note: credentials is not validated by type as it's a Google credentials object
            },
            "ranges": {
                "max_results": (1, 100),
                "timeout": (5, 300),
            },
        }

    @override
    def get_dependencies(self) -> List[TaskDependency]:
        """Get the list of task dependencies.

        Returns:
            List of TaskDependency objects
        """
        return self.dependencies

    @override
    def run(self) -> TaskResult:
        """List Gmail messages based on query and filters.

        Returns:
            TaskResult containing:
                - messages: List of GmailMessage objects (if parse_messages=True)
                - message_ids: List of message IDs (if parse_messages=False)
                - search_result: GmailSearchResult object with metadata
                - query_used: The Gmail query that was executed
                - total_found: Total number of messages found
                - retrieved_count: Number of messages actually retrieved
        """
        from datetime import datetime

        start_time = datetime.now()

        # Validate configuration using instance validation rules
        validation_failure = self._validate_or_fail(start_time)
        if validation_failure:
            return validation_failure

        # Get configuration values (may have been updated after initialization via dependencies)
        query = self.config.get("query", "")
        max_results = self.config.get("max_results", 10)
        include_spam_trash = self.config.get("include_spam_trash", False)
        label_ids = self.config.get("label_ids")
        credentials = self.config.get(
            "credentials"
        )  # Pre-authenticated credentials from GoogleOAuthTask
        credentials_file = self.config.get("credentials_file")
        token_file = self.config.get("token_file")
        parse_messages = self.config.get("parse_messages", True)
        timeout = self.config.get("timeout", 30)

        # Progress callback for detailed operation logging
        def progress_callback(message: str) -> None:
            self.logger.debug(f"[{self.name}] {message}")

        self.logger.info(
            f"Starting Gmail list emails task with query: '{query}', max_results: {max_results}"
        )

        try:
            # Initialize Gmail service with timeout
            progress_callback("Initializing Gmail service...")
            gmail_service = GmailService(
                credentials=credentials,  # Pre-authenticated credentials take priority
                credentials_file=credentials_file,
                token_file=token_file,
                timeout=timeout,
            )

            # List messages with progress tracking
            progress_callback(
                f"Listing messages (query: '{query}', max: {max_results})..."
            )
            # Gmail service list_messages has partially unknown return types but is well-defined
            message_list = gmail_service.list_messages(
                query=query,
                max_results=max_results,
                label_ids=label_ids,
                include_spam_trash=include_spam_trash,
                progress_callback=progress_callback,
            )

            messages: List[GmailMessage] = []
            message_ids: List[str] = []

            # Parse messages if requested
            if parse_messages and message_list:
                progress_callback(f"Starting to parse {len(message_list)} messages...")
                for i, msg_data in enumerate(message_list, 1):
                    try:
                        message_id = msg_data["id"]
                        progress_callback(
                            f"Parsing message {i}/{len(message_list)}: {message_id[:8]}..."
                        )

                        # Get full message data with progress tracking
                        # Gmail service get_message has partially unknown return types but is well-defined
                        full_message_data = gmail_service.get_message(
                            message_id, progress_callback=progress_callback
                        )
                        parsed_message = gmail_service.parse_message(full_message_data)
                        messages.append(parsed_message)
                        message_ids.append(message_id)

                    except GmailServiceError as e:
                        error_msg = f"Gmail API error parsing message {msg_data.get('id', 'unknown')}: {str(e)}"
                        self.logger.warning(error_msg)
                        progress_callback(f"WARNING: {error_msg}")
                        # Continue processing other messages
                        continue
                    except Exception as e:
                        error_msg = f"Unexpected error parsing message {msg_data.get('id', 'unknown')}: {str(e)}"
                        self.logger.warning(error_msg)
                        progress_callback(f"WARNING: {error_msg}")
                        # Continue processing other messages
                        continue

                progress_callback(
                    f"Completed parsing {len(messages)} messages successfully"
                )
            else:
                # Just collect message IDs
                message_ids = [msg["id"] for msg in message_list]
                progress_callback(
                    f"Retrieved {len(message_ids)} message IDs without parsing (faster mode)"
                )

            # Create search result object
            search_result = GmailSearchResult(
                query=query,
                total_count=len(message_ids),
                messages=messages,
                next_page_token=None,  # Note: Gmail API pagination would require additional implementation
                estimated_result_size=len(message_ids),
            )

            result_data = {
                "messages": messages if parse_messages else [],
                "message_ids": message_ids,
                "search_result": search_result.to_dict(),
                "query_used": query,
                "total_found": len(message_ids),
                "retrieved_count": len(messages)
                if parse_messages
                else len(message_ids),
                "parse_messages": parse_messages,
                "search_metadata": {
                    "include_spam_trash": include_spam_trash,
                    "label_ids": label_ids,
                    "max_results_requested": max_results,
                },
            }

            execution_time = (datetime.now() - start_time).total_seconds()

            if message_ids:
                self.logger.info(
                    f"Gmail list emails task completed: {len(message_ids)} messages found, "
                    f"{len(messages) if parse_messages else len(message_ids)} retrieved"
                )

                return TaskResult.success(
                    data=result_data,
                    execution_time=execution_time,
                    metadata={
                        "task_type": "gmail_list_emails",
                        "query": query,
                        "total_found": len(message_ids),
                        "retrieved_count": len(messages)
                        if parse_messages
                        else len(message_ids),
                        "parse_messages": parse_messages,
                    },
                )
            else:
                # No messages found - this is still a success, just empty results
                self.logger.info(
                    f"Gmail list emails task completed: No messages found for query '{query}'"
                )

                return TaskResult.success(
                    data=result_data,
                    execution_time=execution_time,
                    metadata={
                        "task_type": "gmail_list_emails",
                        "query": query,
                        "total_found": 0,
                        "retrieved_count": 0,
                        "parse_messages": parse_messages,
                    },
                )

        except GmailServiceError as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"GmailListEmailsTask Gmail operation failed: {str(e)}"
            self.logger.error(error_msg)

            return TaskResult.failure(
                error_message=error_msg,
                execution_time=execution_time,
                metadata={
                    "task_type": "gmail_list_emails",
                    "query": query,
                    "error_type": "GmailServiceError",
                    "max_results": max_results,
                },
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"GmailListEmailsTask operation failed: {str(e)}"
            self.logger.error(error_msg)

            return TaskResult.failure(
                error_message=error_msg,
                execution_time=execution_time,
                metadata={
                    "task_type": "gmail_list_emails",
                    "query": query,
                    "error_type": type(e).__name__,
                    "max_results": max_results,
                },
            )

    @staticmethod
    def create_search_query(**kwargs: Any) -> str:
        """Helper method to create Gmail search queries using common patterns.

        Supported kwargs:
            - from_sender: str - Filter by sender email
            - to_recipient: str - Filter by recipient email
            - subject: str - Filter by subject
            - newer_than_days: int - Messages newer than X days
            - older_than_days: int - Messages older than X days
            - has_attachment: bool - Messages with attachments
            - is_unread: bool - Unread messages
            - is_important: bool - Important messages
            - is_starred: bool - Starred messages
            - label: str - Messages with specific label
            - custom_query: str - Additional custom query terms

        Returns:
            Gmail search query string
        """
        # Initialize query parts list with explicit type annotation
        query_parts: List[str] = []

        # Query parts list for building Gmail search query
        query_parts: List[str] = []

        if "from_sender" in kwargs:
            # GmailSearchOperators methods return well-defined string queries
            query_parts.append(GmailSearchOperators.from_sender(kwargs["from_sender"]))

        if "to_recipient" in kwargs:
            query_parts.append(
                GmailSearchOperators.to_recipient(kwargs["to_recipient"])
            )

        if "subject" in kwargs:
            query_parts.append(GmailSearchOperators.with_subject(kwargs["subject"]))

        if "newer_than_days" in kwargs:
            query_parts.append(
                GmailSearchOperators.newer_than(kwargs["newer_than_days"])
            )

        if "older_than_days" in kwargs:
            query_parts.append(
                GmailSearchOperators.older_than(kwargs["older_than_days"])
            )

        if kwargs.get("has_attachment"):
            query_parts.append(GmailSearchOperators.has_attachment())

        if kwargs.get("is_unread"):
            query_parts.append(GmailSearchOperators.is_unread())

        if kwargs.get("is_important"):
            query_parts.append(GmailSearchOperators.is_important())

        if kwargs.get("is_starred"):
            query_parts.append(GmailSearchOperators.is_starred())

        if "label" in kwargs:
            query_parts.append(GmailSearchOperators.with_label(kwargs["label"]))

        if "custom_query" in kwargs:
            query_parts.append(kwargs["custom_query"])

        # Combine all query parts with AND operator
        return GmailSearchOperators.combine_and(*query_parts) if query_parts else ""
