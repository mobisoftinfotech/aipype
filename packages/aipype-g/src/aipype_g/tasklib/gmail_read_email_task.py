"""Gmail Read Email Task - Read specific Gmail message content."""

from typing import List, Dict, Any, Optional

from typing import override

from aipype.framework.base_task import BaseTask
from aipype.framework.task_dependencies import TaskDependency
from aipype.framework.task_result import TaskResult
from .gmail_service import GmailService, GmailServiceError
from .gmail_models import GmailMessage


class GmailReadEmailTask(BaseTask):
    """Task that reads specific Gmail message content by ID."""

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        dependencies: Optional[List[TaskDependency]] = None,
    ):
        """Initialize Gmail read email task.

        Args:
            name: Task name
            config: Configuration dictionary containing:
                - message_id: Gmail message ID to read (can be resolved from dependencies)
                - message_ids: List of message IDs to read (alternative to message_id)
                - format: Message format ("full", "minimal", "raw", "metadata") (default: "full")
                - include_attachments: Whether to include attachment info (default: True)
                - credentials_file: Path to OAuth2 credentials file (optional)
                - token_file: Path to OAuth2 token file (optional)
            dependencies: List of task dependencies
        """
        super().__init__(name, config)
        self.dependencies = dependencies or []
        self.validation_rules = {
            "defaults": {
                "format": "full",
                "include_attachments": True,
            },
            "types": {
                "message_id": str,
                "message_ids": list,
                "format": str,
                "include_attachments": bool,
                "credentials_file": str,
                "token_file": str,
                # Note: credentials is not validated by type as it's a Google credentials object
            },
            "choices": {
                "format": ["full", "minimal", "raw", "metadata"],
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
        """Read Gmail message(s) content.

        Returns:
            TaskResult containing:
                - message: GmailMessage object (if single message_id)
                - messages: List of GmailMessage objects (if multiple message_ids)
                - message_count: Number of messages read
                - failed_reads: List of failed message IDs with error details
                - success_count: Number of successfully read messages
                - format_used: Message format that was used
        """
        from datetime import datetime

        start_time = datetime.now()

        # Validate configuration using instance validation rules
        validation_failure = self._validate_or_fail(start_time)
        if validation_failure:
            return validation_failure

        # Get configuration values (may have been updated after initialization via dependencies)
        message_id = self.config.get("message_id")
        message_ids = self.config.get("message_ids", [])
        message_format = self.config.get("format", "full")
        include_attachments = self.config.get("include_attachments", True)
        credentials = self.config.get(
            "credentials"
        )  # Pre-authenticated credentials from GoogleOAuthTask
        credentials_file = self.config.get("credentials_file")
        token_file = self.config.get("token_file")

        # Determine which messages to read
        if message_id:
            target_message_ids = [message_id]
            single_message_mode = True
        elif message_ids:
            target_message_ids = message_ids
            single_message_mode = False
        else:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = "GmailReadEmailTask configuration failed: Either message_id or message_ids must be provided"
            self.logger.error(error_msg)

            return TaskResult.failure(
                error_message=error_msg,
                execution_time=execution_time,
                metadata={
                    "task_type": "gmail_read_email",
                    "error_type": "ConfigurationError",
                },
            )

        self.logger.info(
            f"Starting Gmail read email task for {len(target_message_ids)} message(s)"
        )

        try:
            # Initialize Gmail service
            gmail_service = GmailService(
                credentials=credentials,  # Pre-authenticated credentials take priority
                credentials_file=credentials_file,
                token_file=token_file,
            )

            messages: List[GmailMessage] = []
            failed_reads: List[Dict[str, Any]] = []
            success_count = 0

            # Read each message
            for i, msg_id in enumerate(target_message_ids, 1):
                self.logger.debug(
                    f"Reading message {i}/{len(target_message_ids)}: {msg_id}"
                )

                try:
                    # Get message data from Gmail API
                    # Gmail service get_message has partially unknown return types but is well-defined
                    message_data = gmail_service.get_message(
                        msg_id, format=message_format
                    )

                    # Parse the message
                    parsed_message = gmail_service.parse_message(message_data)

                    # Filter out attachments if not requested
                    if not include_attachments:
                        parsed_message.attachments = []

                    messages.append(parsed_message)
                    success_count += 1

                    self.logger.debug(
                        f"Successfully read message {msg_id}: '{parsed_message.subject}' from {parsed_message.sender_email}"
                    )

                except GmailServiceError as e:
                    error_info = {
                        "message_id": msg_id,
                        "error": str(e),
                        "error_type": "GmailServiceError",
                    }
                    failed_reads.append(error_info)
                    self.logger.warning(f"Failed to read message {msg_id}: {e}")

                except Exception as e:
                    error_info = {
                        "message_id": msg_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                    failed_reads.append(error_info)
                    self.logger.warning(f"Failed to read message {msg_id}: {e}")

            # Prepare result data
            result_data = {
                "message_count": len(target_message_ids),
                "success_count": success_count,
                "failed_count": len(failed_reads),
                "failed_reads": failed_reads,
                "format_used": message_format,
                "include_attachments": include_attachments,
            }

            if single_message_mode:
                result_data["message"] = messages[0].to_dict() if messages else None
            else:
                result_data["messages"] = [msg.to_dict() for msg in messages]

            execution_time = (datetime.now() - start_time).total_seconds()

            if success_count > 0:
                self.logger.info(
                    f"Gmail read email task completed: {success_count}/{len(target_message_ids)} messages read successfully"
                )

                return TaskResult.success(
                    data=result_data,
                    execution_time=execution_time,
                    metadata={
                        "task_type": "gmail_read_email",
                        "total_messages": len(target_message_ids),
                        "success_count": success_count,
                        "failed_count": len(failed_reads),
                        "format_used": message_format,
                        "single_message_mode": single_message_mode,
                    },
                )
            else:
                # All reads failed
                error_msg = f"GmailReadEmailTask read operation failed: All {len(target_message_ids)} message read(s) failed"
                self.logger.error(error_msg)

                return TaskResult.failure(
                    error_message=error_msg,
                    execution_time=execution_time,
                    metadata={
                        "task_type": "gmail_read_email",
                        "total_messages": len(target_message_ids),
                        "success_count": 0,
                        "failed_count": len(failed_reads),
                        "failed_reads": failed_reads,
                        "format_used": message_format,
                        "single_message_mode": single_message_mode,
                    },
                )

        except GmailServiceError as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"GmailReadEmailTask Gmail service operation failed: {str(e)}"
            self.logger.error(error_msg)

            return TaskResult.failure(
                error_message=error_msg,
                execution_time=execution_time,
                metadata={
                    "task_type": "gmail_read_email",
                    "total_messages": len(target_message_ids),
                    "error_type": "GmailServiceError",
                    "format_requested": message_format,
                },
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"GmailReadEmailTask operation failed: {str(e)}"
            self.logger.error(error_msg)

            return TaskResult.failure(
                error_message=error_msg,
                execution_time=execution_time,
                metadata={
                    "task_type": "gmail_read_email",
                    "total_messages": len(target_message_ids),
                    "error_type": type(e).__name__,
                    "format_requested": message_format,
                },
            )

    @staticmethod
    def extract_message_ids_from_list_result(
        list_result_data: Dict[str, Any],
    ) -> List[str]:
        """Helper method to extract message IDs from GmailListEmailsTask result.

        Args:
            list_result_data: Result data from GmailListEmailsTask

        Returns:
            List of message IDs
        """
        message_ids = list_result_data.get("message_ids", [])
        if not message_ids:
            # Try to extract from messages if available
            messages = list_result_data.get("messages", [])
            if messages:
                message_ids = [
                    msg.get("message_id", "")
                    for msg in messages
                    if msg.get("message_id")
                ]

        # Ensure we return a list of strings
        return [str(msg_id) for msg_id in message_ids if msg_id]

    @staticmethod
    def create_batch_read_config(
        message_ids: List[str],
        format: str = "full",
        include_attachments: bool = True,
        credentials_file: Optional[str] = None,
        token_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Helper method to create configuration for reading multiple messages.

        Args:
            message_ids: List of Gmail message IDs to read
            format: Message format to use
            include_attachments: Whether to include attachment information
            credentials_file: Path to OAuth2 credentials file
            token_file: Path to OAuth2 token file

        Returns:
            Configuration dictionary for GmailReadEmailTask
        """
        config = {
            "message_ids": message_ids,
            "format": format,
            "include_attachments": include_attachments,
        }

        if credentials_file:
            config["credentials_file"] = credentials_file
        if token_file:
            config["token_file"] = token_file

        return config
