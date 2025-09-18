"""Gmail data models for structured email data representation."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class GmailAttachment:
    """Represents a Gmail message attachment."""

    attachment_id: str
    filename: str
    mime_type: str
    size: int


@dataclass
class GmailMessage:
    """Represents a Gmail message with parsed content."""

    message_id: str
    thread_id: str
    label_ids: List[str]
    snippet: str
    subject: str
    sender: str
    recipient: str
    date: str
    body: str
    history_id: Optional[str] = None
    internal_date: Optional[str] = None
    size_estimate: int = 0
    cc: str = ""
    bcc: str = ""
    html_body: str = ""
    # Pyright incorrectly reports these as partially unknown despite explicit type annotations
    # These are legitimate dataclass fields with proper generic type parameters
    attachments: List[GmailAttachment] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    headers: Dict[str, str] = field(default_factory=dict)  # pyright: ignore[reportUnknownVariableType]

    @property
    def has_attachments(self) -> bool:
        """Check if message has attachments."""
        return len(self.attachments) > 0

    @property
    def attachment_count(self) -> int:
        """Get number of attachments."""
        return len(self.attachments)

    @property
    def is_unread(self) -> bool:
        """Check if message is unread."""
        return "UNREAD" in self.label_ids

    @property
    def is_important(self) -> bool:
        """Check if message is marked as important."""
        return "IMPORTANT" in self.label_ids

    @property
    def is_starred(self) -> bool:
        """Check if message is starred."""
        return "STARRED" in self.label_ids

    @property
    def sender_name(self) -> str:
        """Extract sender name from sender field."""
        if "<" in self.sender and ">" in self.sender:
            return self.sender.split("<")[0].strip().strip('"')
        return self.sender

    @property
    def sender_email(self) -> str:
        """Extract sender email from sender field."""
        if "<" in self.sender and ">" in self.sender:
            return self.sender.split("<")[1].split(">")[0].strip()
        return self.sender

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "message_id": self.message_id,
            "thread_id": self.thread_id,
            "label_ids": self.label_ids,
            "snippet": self.snippet,
            "subject": self.subject,
            "sender": self.sender,
            "sender_name": self.sender_name,
            "sender_email": self.sender_email,
            "recipient": self.recipient,
            "cc": self.cc,
            "bcc": self.bcc,
            "date": self.date,
            "body": self.body,
            "html_body": self.html_body,
            "attachments": [
                {
                    "attachment_id": att.attachment_id,
                    "filename": att.filename,
                    "mime_type": att.mime_type,
                    "size": att.size,
                }
                for att in self.attachments
            ],
            "has_attachments": self.has_attachments,
            "attachment_count": self.attachment_count,
            "is_unread": self.is_unread,
            "is_important": self.is_important,
            "is_starred": self.is_starred,
            "size_estimate": self.size_estimate,
            "headers": self.headers,
        }


@dataclass
class GmailThread:
    """Represents a Gmail conversation thread."""

    thread_id: str
    history_id: str
    # Pyright incorrectly reports this as partially unknown despite explicit type annotation
    # This is a legitimate dataclass field with proper generic type parameter
    messages: List[GmailMessage] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]

    @property
    def message_count(self) -> int:
        """Get number of messages in thread."""
        return len(self.messages)

    @property
    def is_unread(self) -> bool:
        """Check if thread has unread messages."""
        return any(msg.is_unread for msg in self.messages)

    @property
    def latest_message(self) -> Optional[GmailMessage]:
        """Get the latest message in the thread."""
        if not self.messages:
            return None
        return max(self.messages, key=lambda m: m.internal_date or "0")

    @property
    def participants(self) -> List[str]:
        """Get all unique participants in the thread."""
        # Explicit type annotation for set to resolve generic type warnings
        participants: set[str] = set()
        for message in self.messages:
            # Set add operations have well-defined types
            participants.add(message.sender_email)
            if message.recipient:
                # Adding recipient email to participant set
                participants.add(message.recipient)
        # Convert set of participants to list
        return list(participants)

    def to_dict(self) -> Dict[str, Any]:
        """Convert thread to dictionary."""
        return {
            "thread_id": self.thread_id,
            "history_id": self.history_id,
            "message_count": self.message_count,
            "is_unread": self.is_unread,
            "participants": self.participants,
            "messages": [msg.to_dict() for msg in self.messages],
            "latest_message": self.latest_message.to_dict()
            if self.latest_message
            else None,
        }


@dataclass
class GmailLabel:
    """Represents a Gmail label."""

    label_id: str
    name: str
    message_list_visibility: Optional[str] = None
    label_list_visibility: Optional[str] = None
    label_type: Optional[str] = None
    messages_total: Optional[int] = None
    messages_unread: Optional[int] = None
    threads_total: Optional[int] = None
    threads_unread: Optional[int] = None

    @property
    def is_system_label(self) -> bool:
        """Check if this is a system label."""
        return self.label_type == "system"

    @property
    def is_user_label(self) -> bool:
        """Check if this is a user-created label."""
        return self.label_type == "user"

    def to_dict(self) -> Dict[str, Any]:
        """Convert label to dictionary."""
        return {
            "label_id": self.label_id,
            "name": self.name,
            "message_list_visibility": self.message_list_visibility,
            "label_list_visibility": self.label_list_visibility,
            "label_type": self.label_type,
            "messages_total": self.messages_total,
            "messages_unread": self.messages_unread,
            "threads_total": self.threads_total,
            "threads_unread": self.threads_unread,
            "is_system_label": self.is_system_label,
            "is_user_label": self.is_user_label,
        }


@dataclass
class GmailSearchResult:
    """Represents the result of a Gmail search operation."""

    query: str
    total_count: int
    # Pyright incorrectly reports this as partially unknown despite explicit type annotation
    # This is a legitimate dataclass field with proper generic type parameter
    messages: List[GmailMessage] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    next_page_token: Optional[str] = None
    estimated_result_size: Optional[int] = None

    @property
    def message_count(self) -> int:
        """Get number of messages returned."""
        return len(self.messages)

    @property
    def has_more_results(self) -> bool:
        """Check if there are more results available."""
        return self.next_page_token is not None

    @property
    def unread_count(self) -> int:
        """Count unread messages in results."""
        return sum(1 for msg in self.messages if msg.is_unread)

    @property
    def important_count(self) -> int:
        """Count important messages in results."""
        return sum(1 for msg in self.messages if msg.is_important)

    @property
    def starred_count(self) -> int:
        """Count starred messages in results."""
        return sum(1 for msg in self.messages if msg.is_starred)

    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary."""
        return {
            "query": self.query,
            "total_count": self.total_count,
            "message_count": self.message_count,
            "messages": [msg.to_dict() for msg in self.messages],
            "next_page_token": self.next_page_token,
            "estimated_result_size": self.estimated_result_size,
            "has_more_results": self.has_more_results,
            "unread_count": self.unread_count,
            "important_count": self.important_count,
            "starred_count": self.starred_count,
        }


@dataclass
class GmailCategorization:
    """Represents the result of email categorization."""

    message_id: str
    category: str
    confidence: float
    reasoning: str
    # Pyright incorrectly reports this as partially unknown despite explicit type annotation
    # This is a legitimate dataclass field with proper generic type parameter
    suggested_labels: List[str] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    priority_score: Optional[float] = None
    urgency_level: Optional[str] = None
    requires_response: bool = False
    response_deadline: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert categorization to dictionary."""
        return {
            "message_id": self.message_id,
            "category": self.category,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "suggested_labels": self.suggested_labels,
            "priority_score": self.priority_score,
            "urgency_level": self.urgency_level,
            "requires_response": self.requires_response,
            "response_deadline": self.response_deadline,
        }


# System label constants for easy reference
class GmailSystemLabels:
    """Gmail system label IDs."""

    INBOX = "INBOX"
    SENT = "SENT"
    DRAFT = "DRAFT"
    SPAM = "SPAM"
    TRASH = "TRASH"
    UNREAD = "UNREAD"
    STARRED = "STARRED"
    IMPORTANT = "IMPORTANT"
    CATEGORY_PERSONAL = "CATEGORY_PERSONAL"
    CATEGORY_SOCIAL = "CATEGORY_SOCIAL"
    CATEGORY_PROMOTIONS = "CATEGORY_PROMOTIONS"
    CATEGORY_UPDATES = "CATEGORY_UPDATES"
    CATEGORY_FORUMS = "CATEGORY_FORUMS"


# Common Gmail search operators for building queries
class GmailSearchOperators:
    """Common Gmail search operators and helpers."""

    @staticmethod
    def from_sender(email: str) -> str:
        """Create query for messages from specific sender."""
        return f"from:{email}"

    @staticmethod
    def to_recipient(email: str) -> str:
        """Create query for messages to specific recipient."""
        return f"to:{email}"

    @staticmethod
    def with_subject(subject: str) -> str:
        """Create query for messages with specific subject."""
        return f'subject:"{subject}"'

    @staticmethod
    def newer_than(days: int) -> str:
        """Create query for messages newer than specified days."""
        return f"newer_than:{days}d"

    @staticmethod
    def older_than(days: int) -> str:
        """Create query for messages older than specified days."""
        return f"older_than:{days}d"

    @staticmethod
    def has_attachment() -> str:
        """Create query for messages with attachments."""
        return "has:attachment"

    @staticmethod
    def is_unread() -> str:
        """Create query for unread messages."""
        return "is:unread"

    @staticmethod
    def is_important() -> str:
        """Create query for important messages."""
        return "is:important"

    @staticmethod
    def is_starred() -> str:
        """Create query for starred messages."""
        return "is:starred"

    @staticmethod
    def with_label(label_name: str) -> str:
        """Create query for messages with specific label."""
        return f"label:{label_name}"

    @staticmethod
    def combine_and(*queries: str) -> str:
        """Combine queries with AND operator."""
        return " ".join(queries)

    @staticmethod
    def combine_or(*queries: str) -> str:
        """Combine queries with OR operator."""
        return " OR ".join(f"({q})" for q in queries)
