"""Google Services TaskLib - Gmail, Sheets, and other Google API integrations.

This package contains task implementations for Google services,
including Gmail API for email automation and Sheets API for spreadsheet operations.

Authentication:
- GoogleOAuthTask: Unified OAuth2 authentication for multiple Google services
- GoogleAuthService: Core authentication service with multi-service support

Gmail Tasks:
- GmailListEmailsTask: List emails with filters and queries
- GmailReadEmailTask: Read specific email content

Sheets Tasks:
- ReadGoogleSheetTask: Read data from Google Sheets as 2D arrays

Services:
- GmailService: Core Gmail API service with OAuth2 authentication
- GoogleSheetsService: Core Sheets API service for reading spreadsheet data

Models:
- Gmail: GmailMessage, GmailThread, GmailLabel, GmailAttachment
- Sheets: SheetData, SheetRange, SpreadsheetInfo
"""

# Authentication
from .google_auth_service import GoogleAuthService
from .google_oauth_task import GoogleOAuthTask

# Gmail
from .gmail_list_emails_task import GmailListEmailsTask
from .gmail_read_email_task import GmailReadEmailTask
from .gmail_service import GmailService
from .gmail_models import GmailMessage, GmailThread, GmailLabel, GmailAttachment

# Sheets
from .read_google_sheet_task import ReadGoogleSheetTask
from .google_sheets_service import GoogleSheetsService
from .google_sheets_models import SheetData, SheetRange, SpreadsheetInfo

__all__ = [
    # Authentication
    "GoogleAuthService",
    "GoogleOAuthTask",
    # Gmail Tasks
    "GmailListEmailsTask",
    "GmailReadEmailTask",
    # Gmail Service
    "GmailService",
    # Gmail Models
    "GmailMessage",
    "GmailThread",
    "GmailLabel",
    "GmailAttachment",
    # Sheets Tasks
    "ReadGoogleSheetTask",
    # Sheets Service
    "GoogleSheetsService",
    # Sheets Models
    "SheetData",
    "SheetRange",
    "SpreadsheetInfo",
]
