"""Google Sheets API Service - Core Sheets operations with OAuth2 authentication."""

import os
from typing import Any, List, Optional, Callable
from google.oauth2.credentials import Credentials

# Google API client library lacks comprehensive type stubs
from googleapiclient.discovery import build  # pyright: ignore[reportUnknownVariableType]
from googleapiclient.errors import HttpError

from aipype.framework.utils.common import setup_logger
from .google_sheets_models import SheetData, SheetRange, SpreadsheetInfo
from .google_auth_service import GoogleAuthService, SHEETS_SCOPES


# Type aliases for Google Sheets API
SheetsServiceType = Any
ProgressCallback = Callable[[str], None]


class GoogleSheetsError(Exception):
    """Custom exception for Google Sheets service errors."""

    pass


class GoogleSheetsService:
    """Google Sheets API service with OAuth2 authentication and read operations."""

    def __init__(
        self,
        credentials: Optional[Credentials] = None,
        credentials_file: Optional[str] = None,
        token_file: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        """Initialize Google Sheets service.

        Args:
            credentials: Pre-authenticated Google credentials (from GoogleOAuthTask)
            credentials_file: Path to OAuth2 credentials JSON file (fallback)
            token_file: Path to store/load OAuth2 tokens (fallback)
            timeout: Request timeout in seconds (default: 30)
        """
        self.logger = setup_logger("google_sheets_service")
        self.timeout = timeout
        # Google API client lacks comprehensive type stubs, using Any for service object
        self.service: Optional[SheetsServiceType] = None
        self.credentials: Optional[Credentials] = credentials

        # Store auth params for fallback authentication
        self.credentials_file = credentials_file or os.getenv(
            "GOOGLE_CREDENTIALS_FILE",
            os.getenv("GMAIL_CREDENTIALS_FILE", "google_credentials.json"),
        )
        self.token_file = token_file or os.getenv(
            "SHEETS_TOKEN_FILE", "sheets_token.json"
        )

        # Try to initialize service
        try:
            self._initialize_service()
        except Exception as e:
            self.logger.warning(
                f"Failed to initialize Sheets service: {e}. Service will be initialized on first use."
            )

    def _initialize_service(self) -> None:
        """Initialize the Google Sheets service."""
        if not self.credentials:
            # Use GoogleAuthService for authentication
            auth_service = GoogleAuthService(
                credentials_file=self.credentials_file,
                token_file=self.token_file,
                scopes=SHEETS_SCOPES,
            )
            self.credentials = auth_service.authenticate()

        # Build service
        # Google API build function returns service object with unknown type
        self.service = build("sheets", "v4", credentials=self.credentials)

        self.logger.info("Google Sheets API service initialized successfully")

    def _ensure_service(self) -> None:
        """Ensure service is initialized and ready."""
        if not self.service:
            self._initialize_service()

    def get_spreadsheet_info(
        self,
        spreadsheet_id: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> SpreadsheetInfo:
        """Get information about a spreadsheet.

        Args:
            spreadsheet_id: ID of the Google Spreadsheet
            progress_callback: Optional callback for progress updates

        Returns:
            SpreadsheetInfo object with spreadsheet metadata

        Raises:
            GoogleSheetsError: If API call fails
        """
        self._ensure_service()

        if progress_callback:
            progress_callback(f"Fetching spreadsheet info for {spreadsheet_id[:8]}...")

        try:
            # Google API service methods have unknown return types
            spreadsheet = (
                self.service.spreadsheets()  # pyright: ignore[reportOptionalMemberAccess]
                .get(spreadsheetId=spreadsheet_id)
                .execute()
            )

            title = spreadsheet.get("properties", {}).get("title", "Untitled")
            sheets = spreadsheet.get("sheets", [])
            sheet_names = [sheet["properties"]["title"] for sheet in sheets]

            info = SpreadsheetInfo(
                spreadsheet_id=spreadsheet_id,
                title=title,
                sheet_names=sheet_names,
                properties=spreadsheet.get("properties", {}),
            )

            if progress_callback:
                progress_callback(
                    f"Retrieved info for '{title}' with {len(sheet_names)} sheets"
                )

            self.logger.debug(
                f"Retrieved spreadsheet info: {title} ({len(sheet_names)} sheets)"
            )
            return info

        except HttpError as e:
            error_msg = f"Google Sheets API error getting spreadsheet info: {e}"
            self.logger.error(error_msg)
            raise GoogleSheetsError(error_msg) from e

    def read_range(
        self,
        spreadsheet_id: str,
        range_a1: str,
        include_headers: bool = True,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> SheetData:
        """Read data from a specific range in a spreadsheet.

        Args:
            spreadsheet_id: ID of the Google Spreadsheet
            range_a1: Range in A1 notation (e.g., 'Sheet1!A1:C10')
            include_headers: Whether first row should be treated as headers
            progress_callback: Optional callback for progress updates

        Returns:
            SheetData object with the read data

        Raises:
            GoogleSheetsError: If API call fails or range is invalid
        """
        self._ensure_service()

        if progress_callback:
            progress_callback(f"Reading range {range_a1} from spreadsheet...")

        try:
            # Google API service methods have unknown return types
            result = (
                self.service.spreadsheets()  # pyright: ignore[reportOptionalMemberAccess]
                .values()
                .get(spreadsheetId=spreadsheet_id, range=range_a1)
                .execute()
            )

            # Extract raw values
            raw_values: List[List[Any]] = result.get("values", [])

            # Convert all values to strings for consistency
            string_values: List[List[str]] = []
            for row in raw_values:
                string_row = [str(cell) if cell is not None else "" for cell in row]
                string_values.append(string_row)

            # Parse sheet name from range
            if "!" in range_a1:
                sheet_name = range_a1.split("!", 1)[0]
            else:
                sheet_name = "Sheet1"  # Default

            # Handle headers
            headers: List[str] = []
            data_values = string_values

            if include_headers and string_values:
                headers = string_values[0]
                data_values = string_values[1:]

            sheet_data = SheetData(
                spreadsheet_id=spreadsheet_id,
                sheet_name=sheet_name,
                range_read=range_a1,
                values=data_values,
                headers=headers,
                metadata={
                    "include_headers": include_headers,
                    "raw_row_count": len(raw_values),
                    "processed_row_count": len(data_values),
                },
            )

            if progress_callback:
                progress_callback(
                    f"Read {len(data_values)} rows from {sheet_name} "
                    f"({len(headers)} headers)"
                    if headers
                    else "(no headers)"
                )

            self.logger.info(
                f"Successfully read range {range_a1}: {len(data_values)} data rows, "
                f"{len(headers)} headers"
            )

            return sheet_data

        except HttpError as e:
            error_msg = f"Google Sheets API error reading range {range_a1}: {e}"
            self.logger.error(error_msg)
            raise GoogleSheetsError(error_msg) from e

    def read_sheet(
        self,
        spreadsheet_id: str,
        sheet_name: str,
        start_row: int = 1,
        end_row: Optional[int] = None,
        start_col: int = 1,
        end_col: Optional[int] = None,
        include_headers: bool = True,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> SheetData:
        """Read data from a sheet with row/column bounds.

        Args:
            spreadsheet_id: ID of the Google Spreadsheet
            sheet_name: Name of the sheet to read
            start_row: Starting row (1-based, default: 1)
            end_row: Ending row (1-based, optional)
            start_col: Starting column (1-based, default: 1)
            end_col: Ending column (1-based, optional)
            include_headers: Whether first row should be treated as headers
            progress_callback: Optional callback for progress updates

        Returns:
            SheetData object with the read data

        Raises:
            GoogleSheetsError: If API call fails
        """
        # Create range object and convert to A1 notation
        range_obj = SheetRange(
            sheet_name=sheet_name,
            start_row=start_row,
            start_col=start_col,
            end_row=end_row,
            end_col=end_col,
        )

        return self.read_range(
            spreadsheet_id=spreadsheet_id,
            range_a1=range_obj.a1_notation,
            include_headers=include_headers,
            progress_callback=progress_callback,
        )

    def read_all_data(
        self,
        spreadsheet_id: str,
        sheet_name: str,
        include_headers: bool = True,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> SheetData:
        """Read all data from a sheet.

        Args:
            spreadsheet_id: ID of the Google Spreadsheet
            sheet_name: Name of the sheet to read
            include_headers: Whether first row should be treated as headers
            progress_callback: Optional callback for progress updates

        Returns:
            SheetData object with all data from the sheet

        Raises:
            GoogleSheetsError: If API call fails
        """
        # Use sheet name only to get all data
        range_a1 = sheet_name

        return self.read_range(
            spreadsheet_id=spreadsheet_id,
            range_a1=range_a1,
            include_headers=include_headers,
            progress_callback=progress_callback,
        )

    def get_sheet_names(
        self,
        spreadsheet_id: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[str]:
        """Get list of sheet names in a spreadsheet.

        Args:
            spreadsheet_id: ID of the Google Spreadsheet
            progress_callback: Optional callback for progress updates

        Returns:
            List of sheet names

        Raises:
            GoogleSheetsError: If API call fails
        """
        info = self.get_spreadsheet_info(spreadsheet_id, progress_callback)
        return info.sheet_names

    def validate_range(
        self,
        spreadsheet_id: str,
        range_a1: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> bool:
        """Validate if a range exists and is accessible.

        Args:
            spreadsheet_id: ID of the Google Spreadsheet
            range_a1: Range in A1 notation
            progress_callback: Optional callback for progress updates

        Returns:
            True if range is valid and accessible, False otherwise
        """
        try:
            # Attempt to read just the first cell to validate range
            self.read_range(
                spreadsheet_id=spreadsheet_id,
                range_a1=range_a1,
                include_headers=False,
                progress_callback=progress_callback,
            )
            return True
        except GoogleSheetsError:
            return False
