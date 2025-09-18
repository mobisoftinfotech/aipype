"""Google Authentication Service - Unified OAuth2 authentication for Google APIs."""

import os
from typing import List, Optional, Dict, Any
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from aipype.framework.utils.common import setup_logger


# Google API scopes by service
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.labels",
]

SHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
]

DRIVE_SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
]

# Predefined scope combinations for common use cases
GMAIL_ONLY_SCOPES = GMAIL_SCOPES
SHEETS_ONLY_SCOPES = SHEETS_SCOPES
GMAIL_AND_SHEETS_SCOPES = GMAIL_SCOPES + SHEETS_SCOPES
ALL_GOOGLE_SCOPES = GMAIL_SCOPES + SHEETS_SCOPES + DRIVE_SCOPES


class GoogleAuthError(Exception):
    """Custom exception for Google authentication errors."""

    pass


class GoogleAuthService:
    """Unified Google OAuth2 authentication service for multiple Google APIs."""

    def __init__(
        self,
        credentials_file: Optional[str] = None,
        token_file: Optional[str] = None,
        scopes: Optional[List[str]] = None,
    ) -> None:
        """Initialize Google authentication service.

        Args:
            credentials_file: Path to OAuth2 credentials JSON file
            token_file: Path to store/load OAuth2 tokens
            scopes: List of Google API scopes to request
        """
        self.logger = setup_logger("google_auth_service")

        # Default to Gmail and Sheets scopes if none specified
        self.scopes = scopes or GMAIL_AND_SHEETS_SCOPES

        self.credentials_file = credentials_file or os.getenv(
            "GOOGLE_CREDENTIALS_FILE",
            os.getenv("GMAIL_CREDENTIALS_FILE", "google_credentials.json"),
        )
        self.token_file = token_file or self._get_default_token_file()

        self.credentials: Optional[Credentials] = None
        self._authenticated = False

    def _get_default_token_file(self) -> str:
        """Determine the appropriate default token file based on scopes.

        Returns:
            Default token file path for the current scopes
        """
        # Check if scopes are Gmail-only
        if self.has_gmail_access() and not self.has_sheets_access():
            return os.getenv("GMAIL_TOKEN_FILE", "gmail_token.json")

        # Check if scopes are Sheets-only
        if self.has_sheets_access() and not self.has_gmail_access():
            return os.getenv("SHEETS_TOKEN_FILE", "sheets_token.json")

        # For combined scopes or other combinations, use a combined token file
        return "google_combined_token.json"

    def authenticate(self) -> Credentials:
        """Authenticate with Google APIs using OAuth2.

        Returns:
            Authenticated Google credentials object

        Raises:
            GoogleAuthError: If authentication fails
        """
        if self._authenticated and self.credentials:
            return self.credentials

        creds = None

        # Load existing token
        if os.path.exists(self.token_file):
            try:
                # Google API credential loading method has partially unknown types
                creds = Credentials.from_authorized_user_file(  # pyright: ignore[reportUnknownMemberType]
                    self.token_file, self.scopes
                )
                self.logger.debug("Loaded existing credentials from token file")
            except Exception as e:
                self.logger.warning(f"Failed to load existing token: {e}")

        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            # Google credential refresh_token attribute has unknown type
            if creds and creds.expired and creds.refresh_token:  # pyright: ignore[reportUnknownMemberType]
                try:
                    # Google credential refresh method has unknown types
                    creds.refresh(Request())  # pyright: ignore[reportUnknownMemberType]
                    self.logger.info("Refreshed expired credentials")
                except Exception as e:
                    self.logger.warning(f"Failed to refresh credentials: {e}")
                    creds = None

            if not creds:
                # Start OAuth2 flow
                if not os.path.exists(self.credentials_file):
                    raise GoogleAuthError(
                        f"Credentials file not found: {self.credentials_file}. "
                        "Please download from Google Cloud Console."
                    )

                # Google OAuth flow methods have partially unknown types
                flow = InstalledAppFlow.from_client_secrets_file(  # pyright: ignore[reportUnknownMemberType]
                    self.credentials_file, self.scopes
                )
                creds = flow.run_local_server(port=0)  # pyright: ignore[reportUnknownMemberType]
                self.logger.info("Completed OAuth2 flow with new credentials")

            # Save credentials for next run
            try:
                with open(self.token_file, "w") as token:
                    # Google credential to_json method has unknown return type
                    token.write(creds.to_json())  # pyright: ignore[reportUnknownMemberType]
                self.logger.debug("Saved credentials to token file")
            except Exception as e:
                self.logger.warning(f"Failed to save token: {e}")

        # Google credential types have complex inheritance that causes assignment issues
        self.credentials = creds  # pyright: ignore[reportAttributeAccessIssue]
        self._authenticated = True

        self.logger.info(f"Google authentication successful with scopes: {self.scopes}")
        # Type checker assertion - authenticate method guarantees non-None credentials
        assert self.credentials is not None
        return self.credentials

    def get_credentials(self) -> Credentials:
        """Get authenticated credentials, authenticating if necessary.

        Returns:
            Authenticated Google credentials object
        """
        return self.authenticate()

    def is_authenticated(self) -> bool:
        """Check if already authenticated.

        Returns:
            True if authenticated, False otherwise
        """
        return self._authenticated and self.credentials is not None

    def get_scopes(self) -> List[str]:
        """Get the configured scopes.

        Returns:
            List of Google API scopes
        """
        return self.scopes.copy()

    def has_scope(self, scope: str) -> bool:
        """Check if a specific scope is included.

        Args:
            scope: Google API scope to check

        Returns:
            True if scope is included, False otherwise
        """
        return scope in self.scopes

    def has_gmail_access(self) -> bool:
        """Check if Gmail scopes are included.

        Returns:
            True if any Gmail scope is included
        """
        return any(
            scope.startswith("https://www.googleapis.com/auth/gmail")
            for scope in self.scopes
        )

    def has_sheets_access(self) -> bool:
        """Check if Sheets scopes are included.

        Returns:
            True if any Sheets scope is included
        """
        return any(
            scope.startswith("https://www.googleapis.com/auth/spreadsheets")
            for scope in self.scopes
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert authentication info to dictionary.

        Returns:
            Dictionary with authentication status and scopes
        """
        return {
            "authenticated": self._authenticated,
            "scopes": self.scopes,
            "credentials_file": self.credentials_file,
            "token_file": self.token_file,
            "has_gmail_access": self.has_gmail_access(),
            "has_sheets_access": self.has_sheets_access(),
        }

    @staticmethod
    def create_service_with_scopes(
        service_types: List[str],
        credentials_file: Optional[str] = None,
        token_file: Optional[str] = None,
    ) -> "GoogleAuthService":
        """Create authentication service with scopes for specific service types.

        Args:
            service_types: List of service types ("gmail", "sheets", "drive")
            credentials_file: Path to credentials file
            token_file: Path to token file

        Returns:
            GoogleAuthService configured with appropriate scopes
        """
        # Explicit type annotation for accumulated scopes list
        scopes: List[str] = []
        for service_type in service_types:
            if service_type.lower() == "gmail":
                scopes.extend(GMAIL_SCOPES)
            elif service_type.lower() == "sheets":
                scopes.extend(SHEETS_SCOPES)
            elif service_type.lower() == "drive":
                scopes.extend(DRIVE_SCOPES)
            else:
                raise ValueError(f"Unknown service type: {service_type}")

        # Remove duplicates while preserving order
        unique_scopes: List[str] = []
        seen: set[str] = set()
        # Explicit type annotation for loop variable
        scope: str
        for scope in scopes:
            if scope not in seen:
                unique_scopes.append(scope)
                seen.add(scope)

        return GoogleAuthService(
            credentials_file=credentials_file,
            token_file=token_file,
            scopes=unique_scopes,
        )
