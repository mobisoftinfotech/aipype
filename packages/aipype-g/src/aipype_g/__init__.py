"""aipype-g: Google API integrations for the aipype framework."""

__version__ = "0.1.0a3"

# Google tasklib exports
from .tasklib.google_oauth_task import GoogleOAuthTask
from .tasklib.gmail_list_emails_task import GmailListEmailsTask
from .tasklib.gmail_read_email_task import GmailReadEmailTask
from .tasklib.read_google_sheet_task import ReadGoogleSheetTask

# Google service exports
from .tasklib.gmail_service import GmailService
from .tasklib.google_sheets_service import GoogleSheetsService
from .tasklib.google_auth_service import GoogleAuthService

# Google models exports
from .tasklib.gmail_models import GmailMessage, GmailThread, GmailLabel, GmailAttachment
from .tasklib.google_sheets_models import SheetData, SheetRange, SpreadsheetInfo

__all__ = [
    # Tasks
    "GoogleOAuthTask",
    "GmailListEmailsTask",
    "GmailReadEmailTask",
    "ReadGoogleSheetTask",
    # Services
    "GmailService",
    "GoogleSheetsService",
    "GoogleAuthService",
    # Models
    "GmailMessage",
    "GmailThread",
    "GmailLabel",
    "GmailAttachment",
    "SheetData",
    "SheetRange",
    "SpreadsheetInfo",
]
