"""Gmail API Service - Core Gmail operations with OAuth2 authentication."""

import os
import base64
from typing import Dict, Any, List, Optional, Callable
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from google.oauth2.credentials import Credentials

# Google API client library lacks comprehensive type stubs
from googleapiclient.discovery import build  # pyright: ignore[reportUnknownVariableType]
from googleapiclient.errors import HttpError

from aipype.framework.utils.common import setup_logger
from .gmail_models import GmailMessage, GmailLabel, GmailAttachment
from .google_auth_service import GoogleAuthService


# Type aliases for Google API responses
GmailApiResponse = Dict[str, Any]
GmailMessageData = Dict[str, Any]
# Google API service object lacks comprehensive type stubs
GmailServiceType = Any
ProgressCallback = Callable[[str], None]

# Gmail API scopes
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.labels",
]


class GmailServiceError(Exception):
    """Custom exception for Gmail service errors."""

    pass


class GmailService:
    """Gmail API service with OAuth2 authentication and common operations."""

    def __init__(
        self,
        credentials: Optional[Credentials] = None,
        credentials_file: Optional[str] = None,
        token_file: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        timeout: int = 30,
    ) -> None:
        """Initialize Gmail service.

        Args:
            credentials: Pre-authenticated Google credentials (from GoogleOAuthTask)
            credentials_file: Path to OAuth2 credentials JSON file
            token_file: Path to store/load OAuth2 tokens
            scopes: List of Gmail API scopes to request
            timeout: Request timeout in seconds (default: 30)
        """
        self.logger = setup_logger("gmail_service")
        self.scopes = scopes or GMAIL_SCOPES
        self.credentials_file = credentials_file or os.getenv(
            "GOOGLE_CREDENTIALS_FILE", "google_credentials.json"
        )
        self.token_file = token_file or os.getenv(
            "GMAIL_TOKEN_FILE", "gmail_token.json"
        )
        self.timeout = timeout

        # Google API client lacks comprehensive type stubs, using Any for service object
        self.service: Optional[GmailServiceType] = None
        self.credentials: Optional[Credentials] = credentials

        # Try to initialize service
        try:
            self._authenticate()
        except Exception as e:
            self.logger.warning(
                f"Failed to initialize Gmail service: {e}. Service will be initialized on first use."
            )

    def _authenticate(self) -> None:
        """Authenticate with Gmail API using OAuth2."""
        # If we already have credentials from constructor (e.g., from GoogleOAuthTask), use them
        if self.credentials:
            self.logger.debug("Using pre-authenticated credentials")
        else:
            # Use GoogleAuthService for unified authentication
            self.logger.debug("Using GoogleAuthService for authentication")
            auth_service = GoogleAuthService(
                credentials_file=self.credentials_file,
                token_file=self.token_file,
                scopes=self.scopes,
            )
            self.credentials = auth_service.authenticate()

        # Build service with authenticated credentials
        self._build_service()

    def _build_service(self) -> None:
        """Build Gmail service with authenticated credentials."""
        if not self.credentials:
            raise GmailServiceError("Cannot build service without credentials")

        # Build service with HTTP timeout configuration
        # Google API build function returns service object with unknown type
        self.service = build("gmail", "v1", credentials=self.credentials)

        # Configure HTTP timeout on the service's HTTP object
        # Google API service object structure is not fully typed
        if hasattr(self.service, "_http") and self.service._http:  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportOptionalMemberAccess]
            # Set socket timeout for the underlying HTTP connection
            import socket

            original_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(self.timeout)
            # Store original timeout to restore if needed
            self._original_socket_timeout = original_timeout

        self.logger.info("Gmail API service initialized successfully")

    def _ensure_authenticated(self) -> None:
        """Ensure service is authenticated and ready."""
        if not self.service:
            self._authenticate()

    def list_messages(
        self,
        query: str = "",
        max_results: int = 10,
        label_ids: Optional[List[str]] = None,
        include_spam_trash: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[GmailMessageData]:
        """List Gmail messages with optional query and filters.

        Args:
            query: Gmail search query (same syntax as Gmail search)
            max_results: Maximum number of messages to return
            label_ids: List of label IDs to filter by
            include_spam_trash: Whether to include spam and trash
            progress_callback: Optional callback to report progress

        Returns:
            List of message metadata dictionaries

        Raises:
            GmailServiceError: If API call fails
        """
        self._ensure_authenticated()

        if progress_callback:
            progress_callback(f"Starting Gmail message list with query: '{query}'")

        try:
            if progress_callback:
                progress_callback("Executing Gmail API list messages request...")

            # Google API service methods have unknown return types
            result = (
                self.service.users()  # pyright: ignore[reportOptionalMemberAccess]
                .messages()
                .list(
                    userId="me",
                    q=query,
                    maxResults=max_results,
                    labelIds=label_ids,
                    includeSpamTrash=include_spam_trash,
                )
                .execute()
            )

            messages: List[GmailMessageData] = result.get("messages", [])
            self.logger.info(f"Listed {len(messages)} messages with query: '{query}'")

            if progress_callback:
                progress_callback(f"Successfully retrieved {len(messages)} message IDs")

            return messages

        except HttpError as error:
            error_msg = f"Failed to list messages: {error}"
            if progress_callback:
                progress_callback(f"ERROR: {error_msg}")
            raise GmailServiceError(error_msg)
        except Exception as error:
            error_msg = f"Unexpected error listing messages: {error}"
            if progress_callback:
                progress_callback(f"ERROR: {error_msg}")
            raise GmailServiceError(error_msg)

    def get_message(
        self,
        message_id: str,
        format: str = "full",
        progress_callback: Optional[ProgressCallback] = None,
    ) -> GmailMessageData:
        """Get a specific Gmail message by ID.

        Args:
            message_id: The Gmail message ID
            format: Message format ('minimal', 'full', 'raw', 'metadata')
            progress_callback: Optional callback to report progress

        Returns:
            Message data dictionary

        Raises:
            GmailServiceError: If API call fails
        """
        self._ensure_authenticated()

        if progress_callback:
            progress_callback(
                f"Fetching message {message_id[:8]}... (format: {format})"
            )

        try:
            # Google API service methods have unknown return types
            message = (
                self.service.users()  # pyright: ignore[reportOptionalMemberAccess]
                .messages()
                .get(userId="me", id=message_id, format=format)
                .execute()
            )

            self.logger.debug(f"Retrieved message {message_id}")

            if progress_callback:
                progress_callback(f"Successfully retrieved message {message_id[:8]}...")

            return message

        except HttpError as error:
            error_msg = f"Failed to get message {message_id}: {error}"
            if progress_callback:
                progress_callback(f"ERROR: {error_msg}")
            raise GmailServiceError(error_msg)
        except Exception as error:
            error_msg = f"Unexpected error getting message {message_id}: {error}"
            if progress_callback:
                progress_callback(f"ERROR: {error_msg}")
            raise GmailServiceError(error_msg)

    def parse_message(self, message_data: GmailMessageData) -> GmailMessage:
        """Parse Gmail API message data into GmailMessage object.

        Args:
            message_data: Raw message data from Gmail API

        Returns:
            Parsed GmailMessage object
        """
        payload = message_data.get("payload", {})
        headers = {h["name"]: h["value"] for h in payload.get("headers", [])}

        # Extract message body
        body = ""
        if "parts" in payload:
            body = self._extract_body_from_parts(payload["parts"])
        else:
            body = self._extract_body_from_payload(payload)

        # Extract attachments
        attachments: List[GmailAttachment] = []
        if "parts" in payload:
            attachments = self._extract_attachments(payload["parts"])

        return GmailMessage(
            message_id=message_data["id"],
            thread_id=message_data["threadId"],
            label_ids=message_data.get("labelIds", []),
            snippet=message_data.get("snippet", ""),
            history_id=message_data.get("historyId"),
            internal_date=message_data.get("internalDate"),
            size_estimate=message_data.get("sizeEstimate", 0),
            subject=headers.get("Subject", ""),
            sender=headers.get("From", ""),
            recipient=headers.get("To", ""),
            cc=headers.get("Cc", ""),
            bcc=headers.get("Bcc", ""),
            date=headers.get("Date", ""),
            body=body,
            html_body=self._extract_html_body_from_parts(payload.get("parts", [])),
            attachments=attachments,
            headers=headers,
        )

    def _extract_body_from_parts(self, parts: List[GmailMessageData]) -> str:
        """Extract text body from message parts."""
        body = ""
        for part in parts:
            mime_type = part.get("mimeType", "")
            if mime_type == "text/plain":
                part_data = part.get("body", {}).get("data", "")
                if part_data:
                    body += base64.urlsafe_b64decode(part_data).decode("utf-8")
            elif "parts" in part:
                body += self._extract_body_from_parts(part["parts"])
        return body

    def _extract_html_body_from_parts(self, parts: List[GmailMessageData]) -> str:
        """Extract HTML body from message parts."""
        html_body = ""
        for part in parts:
            mime_type = part.get("mimeType", "")
            if mime_type == "text/html":
                part_data = part.get("body", {}).get("data", "")
                if part_data:
                    html_body += base64.urlsafe_b64decode(part_data).decode("utf-8")
            elif "parts" in part:
                html_body += self._extract_html_body_from_parts(part["parts"])
        return html_body

    def _extract_body_from_payload(self, payload: GmailMessageData) -> str:
        """Extract body from single payload (no parts)."""
        body_data = payload.get("body", {}).get("data", "")
        if body_data:
            return base64.urlsafe_b64decode(body_data).decode("utf-8")
        return ""

    def _extract_attachments(
        self, parts: List[GmailMessageData]
    ) -> List[GmailAttachment]:
        """Extract attachment information from message parts."""
        attachments: List[GmailAttachment] = []
        for part in parts:
            if part.get("filename"):
                attachment = GmailAttachment(
                    attachment_id=part.get("body", {}).get("attachmentId", ""),
                    filename=part.get("filename", ""),
                    mime_type=part.get("mimeType", ""),
                    size=part.get("body", {}).get("size", 0),
                )
                attachments.append(attachment)
            elif "parts" in part:
                attachments.extend(self._extract_attachments(part["parts"]))
        return attachments

    def send_message(
        self,
        to: str,
        subject: str,
        body: str,
        html_body: Optional[str] = None,
        cc: Optional[str] = None,
        bcc: Optional[str] = None,
        reply_to_message_id: Optional[str] = None,
    ) -> GmailApiResponse:
        """Send an email message.

        Args:
            to: Recipient email address
            subject: Email subject
            body: Plain text body
            html_body: HTML body (optional)
            cc: CC recipients (optional)
            bcc: BCC recipients (optional)
            reply_to_message_id: Message ID to reply to (optional)

        Returns:
            Sent message data

        Raises:
            GmailServiceError: If sending fails
        """
        self._ensure_authenticated()

        try:
            # Create message
            if html_body:
                message = MIMEMultipart("alternative")
                message.attach(MIMEText(body, "plain"))
                message.attach(MIMEText(html_body, "html"))
            else:
                message = MIMEText(body)

            message["to"] = to
            message["subject"] = subject
            if cc:
                message["cc"] = cc
            if bcc:
                message["bcc"] = bcc

            # Handle reply-to
            thread_id = None
            if reply_to_message_id:
                try:
                    # get_message returns GmailMessageData which has unknown member types
                    original_message = self.get_message(reply_to_message_id)
                    thread_id = original_message.get("threadId")
                    message["In-Reply-To"] = reply_to_message_id
                    # Add References header for proper threading
                    references: List[str] = []
                    for header in original_message.get("payload", {}).get(
                        "headers", []
                    ):
                        if header["name"] == "Message-ID":
                            # Gmail API header values have unknown types but are strings in practice
                            references.append(header["value"])
                        elif header["name"] == "References":
                            # Split References header which contains multiple message IDs
                            references.extend(header["value"].split())
                    if references:
                        # Join all reference message IDs for proper email threading
                        message["References"] = " ".join(references)
                except Exception as e:
                    self.logger.warning(f"Failed to set reply headers: {e}")

            # Encode and send
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
            send_request = {"raw": raw_message}
            if thread_id:
                send_request["threadId"] = thread_id

            # Google API service methods have unknown return types
            result = (
                self.service.users()  # pyright: ignore[reportOptionalMemberAccess]
                .messages()
                .send(userId="me", body=send_request)
                .execute()
            )

            self.logger.info(f"Sent message to {to}: {subject}")
            return result

        except HttpError as error:
            raise GmailServiceError(f"Failed to send message: {error}")

    def modify_message_labels(
        self,
        message_id: str,
        add_label_ids: Optional[List[str]] = None,
        remove_label_ids: Optional[List[str]] = None,
    ) -> GmailApiResponse:
        """Add or remove labels from a message.

        Args:
            message_id: The Gmail message ID
            add_label_ids: List of label IDs to add
            remove_label_ids: List of label IDs to remove

        Returns:
            Modified message data

        Raises:
            GmailServiceError: If modification fails
        """
        self._ensure_authenticated()

        if not add_label_ids and not remove_label_ids:
            raise GmailServiceError("Must specify labels to add or remove")

        try:
            modify_request = {}
            if add_label_ids:
                modify_request["addLabelIds"] = add_label_ids
            if remove_label_ids:
                modify_request["removeLabelIds"] = remove_label_ids

            # Google API service methods have unknown return types
            result = (
                self.service.users()  # pyright: ignore[reportOptionalMemberAccess]
                .messages()
                .modify(userId="me", id=message_id, body=modify_request)
                .execute()
            )

            self.logger.info(
                f"Modified labels for message {message_id}: "
                f"added {add_label_ids or []}, removed {remove_label_ids or []}"
            )
            return result

        except HttpError as error:
            raise GmailServiceError(f"Failed to modify message labels: {error}")

    def list_labels(self) -> List[GmailLabel]:
        """List all Gmail labels.

        Returns:
            List of GmailLabel objects

        Raises:
            GmailServiceError: If API call fails
        """
        self._ensure_authenticated()

        try:
            # Google API service methods have unknown return types
            result = self.service.users().labels().list(userId="me").execute()  # pyright: ignore[reportOptionalMemberAccess]
            labels_data: List[Dict[str, Any]] = result.get("labels", [])

            labels: List[GmailLabel] = []
            for label_data in labels_data:
                label = GmailLabel(
                    label_id=label_data["id"],
                    name=label_data["name"],
                    message_list_visibility=label_data.get("messageListVisibility"),
                    label_list_visibility=label_data.get("labelListVisibility"),
                    label_type=label_data.get("type"),
                    messages_total=label_data.get("messagesTotal"),
                    messages_unread=label_data.get("messagesUnread"),
                    threads_total=label_data.get("threadsTotal"),
                    threads_unread=label_data.get("threadsUnread"),
                )
                labels.append(label)

            self.logger.info(f"Retrieved {len(labels)} labels")
            return labels

        except HttpError as error:
            raise GmailServiceError(f"Failed to list labels: {error}")

    def create_label(
        self,
        name: str,
        message_list_visibility: str = "show",
        label_list_visibility: str = "labelShow",
    ) -> GmailLabel:
        """Create a new Gmail label.

        Args:
            name: Label name
            message_list_visibility: 'show' or 'hide'
            label_list_visibility: 'labelShow', 'labelShowIfUnread', or 'labelHide'

        Returns:
            Created GmailLabel object

        Raises:
            GmailServiceError: If creation fails
        """
        self._ensure_authenticated()

        try:
            label_request = {
                "name": name,
                "messageListVisibility": message_list_visibility,
                "labelListVisibility": label_list_visibility,
            }

            # Google API service methods have unknown return types
            result = (
                self.service.users()  # pyright: ignore[reportOptionalMemberAccess]
                .labels()
                .create(userId="me", body=label_request)
                .execute()
            )

            label = GmailLabel(
                label_id=result["id"],
                name=result["name"],
                message_list_visibility=result.get("messageListVisibility"),
                label_list_visibility=result.get("labelListVisibility"),
                label_type=result.get("type"),
            )

            self.logger.info(f"Created label: {name} ({label.label_id})")
            return label

        except HttpError as error:
            raise GmailServiceError(f"Failed to create label '{name}': {error}")
