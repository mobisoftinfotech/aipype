"""Integration tests for Google Sheets functionality with real Google Sheets API calls.

These tests verify the Google Sheets implementations (GoogleAuthService, GoogleSheetsService,
ReadGoogleSheetTask) work correctly with real Google Sheets API calls and return proper TaskResult objects.

Prerequisites:
- Valid Google credentials file (google_credentials.json or gmail_credentials.json)
- OAuth2 tokens will be created/refreshed automatically
- Google Cloud Console project with Sheets API enabled
- Test Google Spreadsheet with some data
- Internet connection for API calls

Setup:
1. Download OAuth2 credentials from Google Cloud Console
2. Enable Google Sheets API in Google Cloud Console
3. Set environment variables:
   export GOOGLE_CREDENTIALS_FILE=./google_credentials.json
   export SHEETS_TOKEN_FILE=./sheets_token.json
   export TEST_SPREADSHEET_ID=your_test_spreadsheet_id
4. Run tests - first run will trigger OAuth2 flow in browser

Run with: pytest integration_tests/tasklib/google/test_google_sheets_integration.py -v

IMPORTANT: These tests will read real data from your Google Sheets.
Use a dedicated test spreadsheet when possible.
"""

import os
import pytest
import time
from typing import Any, Optional, List

from aipype import (
    TaskDependency,
    DependencyType,
    TaskResult,
)
from aipype_g import (
    GoogleAuthService,
    GoogleOAuthTask,
    GoogleSheetsService,
    ReadGoogleSheetTask,
    SheetData,
    SpreadsheetInfo,
)
from aipype_g.tasklib.google_auth_service import SHEETS_SCOPES


@pytest.fixture(scope="session")
def google_credentials_available() -> bool:
    """Check if Google credentials file exists."""
    credentials_file = os.getenv(
        "GOOGLE_CREDENTIALS_FILE",
        os.getenv("GMAIL_CREDENTIALS_FILE", "google_credentials.json"),
    )
    return os.path.exists(credentials_file)


@pytest.fixture(scope="session")
def test_spreadsheet_id() -> Optional[str]:
    """Get test spreadsheet ID from environment."""
    return os.getenv("TEST_SPREADSHEET_ID")


@pytest.fixture(scope="session")
def skip_if_no_google_credentials(google_credentials_available: bool) -> None:
    """Skip test if Google credentials are not available."""
    if not google_credentials_available:
        pytest.skip(
            "Google credentials not available. "
            "Download from Google Cloud Console and set GOOGLE_CREDENTIALS_FILE environment variable."
        )


@pytest.fixture(scope="session")
def skip_if_no_test_spreadsheet(test_spreadsheet_id: Optional[str]) -> None:
    """Skip test if test spreadsheet ID is not provided."""
    if not test_spreadsheet_id:
        pytest.skip(
            "Test spreadsheet ID not provided. "
            "Set TEST_SPREADSHEET_ID environment variable with a test Google Spreadsheet ID."
        )


@pytest.mark.integration
@pytest.mark.sheets
@pytest.mark.slow
class TestGoogleAuthServiceIntegration:
    """Integration tests for GoogleAuthService with real Google OAuth2."""

    def test_sheets_authentication_flow(
        self, skip_if_no_google_credentials: Any
    ) -> None:
        """Test Google Sheets authentication with GoogleAuthService."""
        auth_service = GoogleAuthService(scopes=SHEETS_SCOPES)

        start_time = time.time()
        credentials = auth_service.authenticate()
        execution_time = time.time() - start_time

        # Verify authentication success
        assert credentials is not None
        assert auth_service.is_authenticated()
        assert auth_service.has_sheets_access()
        assert not auth_service.has_gmail_access()  # Sheets-only scopes

        print(
            f"[SUCCESS] Google Sheets authentication successful in {execution_time:.2f}s"
        )
        print(f"   Scopes: {auth_service.get_scopes()}")

    def test_combined_authentication_flow(
        self, skip_if_no_google_credentials: Any
    ) -> None:
        """Test combined Gmail + Sheets authentication."""
        auth_service = GoogleAuthService.create_service_with_scopes(
            service_types=["gmail", "sheets"]
        )

        credentials = auth_service.authenticate()

        # Verify authentication success with combined scopes
        assert credentials is not None
        assert auth_service.is_authenticated()
        assert auth_service.has_sheets_access()
        assert auth_service.has_gmail_access()

        print("[SUCCESS] Combined Gmail + Sheets authentication successful")
        print(f"   Gmail access: {auth_service.has_gmail_access()}")
        print(f"   Sheets access: {auth_service.has_sheets_access()}")


@pytest.mark.integration
@pytest.mark.sheets
@pytest.mark.slow
class TestGoogleSheetsServiceIntegration:
    """Integration tests for GoogleSheetsService with real Google Sheets API."""

    def test_service_initialization(self, skip_if_no_google_credentials: Any) -> None:
        """Test GoogleSheetsService initialization and authentication."""
        start_time = time.time()
        sheets_service = GoogleSheetsService()
        execution_time = time.time() - start_time

        # Service should initialize successfully
        assert sheets_service is not None
        assert sheets_service.service is not None
        assert sheets_service.credentials is not None

        print(f"[SUCCESS] Google Sheets service initialized in {execution_time:.2f}s")

    def test_get_spreadsheet_info(
        self,
        skip_if_no_google_credentials: Any,
        skip_if_no_test_spreadsheet: Any,
        test_spreadsheet_id: str,
    ) -> None:
        """Test getting spreadsheet information."""
        sheets_service = GoogleSheetsService()

        start_time = time.time()
        spreadsheet_info = sheets_service.get_spreadsheet_info(test_spreadsheet_id)
        execution_time = time.time() - start_time

        # Verify spreadsheet info
        assert isinstance(spreadsheet_info, SpreadsheetInfo)
        assert spreadsheet_info.spreadsheet_id == test_spreadsheet_id
        assert spreadsheet_info.title
        assert len(spreadsheet_info.sheet_names) > 0
        assert spreadsheet_info.num_sheets > 0

        print(f"[SUCCESS] Retrieved spreadsheet info in {execution_time:.2f}s")
        print(f"   Title: '{spreadsheet_info.title}'")
        print(f"   Sheets: {spreadsheet_info.sheet_names}")
        print(f"   Sheet count: {spreadsheet_info.num_sheets}")

    def test_get_sheet_names(
        self,
        skip_if_no_google_credentials: Any,
        skip_if_no_test_spreadsheet: Any,
        test_spreadsheet_id: str,
    ) -> None:
        """Test getting sheet names from spreadsheet."""
        sheets_service = GoogleSheetsService()

        sheet_names = sheets_service.get_sheet_names(test_spreadsheet_id)

        # Verify sheet names
        assert isinstance(sheet_names, list)
        assert len(sheet_names) > 0
        assert all(isinstance(name, str) for name in sheet_names)

        print(f"[SUCCESS] Retrieved {len(sheet_names)} sheet names")
        print(f"   Names: {sheet_names}")

    def test_read_range_basic(
        self,
        skip_if_no_google_credentials: Any,
        skip_if_no_test_spreadsheet: Any,
        test_spreadsheet_id: str,
    ) -> None:
        """Test basic range reading from spreadsheet."""
        sheets_service = GoogleSheetsService()

        # Get first sheet name
        sheet_names = sheets_service.get_sheet_names(test_spreadsheet_id)
        first_sheet = sheet_names[0]

        # Read a small range
        range_a1 = f"{first_sheet}!A1:C5"

        start_time = time.time()
        sheet_data = sheets_service.read_range(
            spreadsheet_id=test_spreadsheet_id, range_a1=range_a1, include_headers=True
        )
        execution_time = time.time() - start_time

        # Verify sheet data
        assert isinstance(sheet_data, SheetData)
        assert sheet_data.spreadsheet_id == test_spreadsheet_id
        assert sheet_data.sheet_name == first_sheet
        assert sheet_data.range_read == range_a1
        assert isinstance(sheet_data.values, list)

        print(f"[SUCCESS] Read range {range_a1} in {execution_time:.2f}s")
        print(f"   Data shape: {sheet_data.shape}")
        print(f"   Headers: {sheet_data.headers}")
        if sheet_data.values:
            print(f"   First row: {sheet_data.values[0][:3]}...")

    def test_read_sheet_with_bounds(
        self,
        skip_if_no_google_credentials: Any,
        skip_if_no_test_spreadsheet: Any,
        test_spreadsheet_id: str,
    ) -> None:
        """Test reading sheet with row/column bounds."""
        sheets_service = GoogleSheetsService()

        # Get first sheet name
        sheet_names = sheets_service.get_sheet_names(test_spreadsheet_id)
        first_sheet = sheet_names[0]

        start_time = time.time()
        sheet_data = sheets_service.read_sheet(
            spreadsheet_id=test_spreadsheet_id,
            sheet_name=first_sheet,
            start_row=1,
            end_row=10,
            start_col=1,
            end_col=5,
            include_headers=True,
        )
        execution_time = time.time() - start_time

        # Verify bounded read
        assert isinstance(sheet_data, SheetData)
        assert sheet_data.sheet_name == first_sheet
        assert sheet_data.num_cols <= 5  # Should be at most 5 columns

        print(f"[SUCCESS] Read bounded sheet data in {execution_time:.2f}s")
        print(f"   Shape: {sheet_data.shape}")
        print(f"   Range read: {sheet_data.range_read}")

    def test_read_all_data(
        self,
        skip_if_no_google_credentials: Any,
        skip_if_no_test_spreadsheet: Any,
        test_spreadsheet_id: str,
    ) -> None:
        """Test reading all data from a sheet."""
        sheets_service = GoogleSheetsService()

        # Get first sheet name
        sheet_names = sheets_service.get_sheet_names(test_spreadsheet_id)
        first_sheet = sheet_names[0]

        start_time = time.time()
        sheet_data = sheets_service.read_all_data(
            spreadsheet_id=test_spreadsheet_id,
            sheet_name=first_sheet,
            include_headers=True,
        )
        execution_time = time.time() - start_time

        # Verify all data read
        assert isinstance(sheet_data, SheetData)
        assert sheet_data.sheet_name == first_sheet
        assert sheet_data.range_read == first_sheet  # Should be just sheet name

        print(f"[SUCCESS] Read all sheet data in {execution_time:.2f}s")
        print(f"   Shape: {sheet_data.shape}")
        print(f"   Has headers: {sheet_data.has_headers}")

    def test_validate_range(
        self,
        skip_if_no_google_credentials: Any,
        skip_if_no_test_spreadsheet: Any,
        test_spreadsheet_id: str,
    ) -> None:
        """Test range validation functionality."""
        sheets_service = GoogleSheetsService()

        # Get first sheet name
        sheet_names = sheets_service.get_sheet_names(test_spreadsheet_id)
        first_sheet = sheet_names[0]

        # Test valid range
        valid_range = f"{first_sheet}!A1:C3"
        is_valid = sheets_service.validate_range(test_spreadsheet_id, valid_range)
        assert is_valid

        # Test invalid range
        invalid_range = f"{first_sheet}!ZZ999:ZZ1000"
        is_invalid = sheets_service.validate_range(test_spreadsheet_id, invalid_range)
        # Note: This might still be valid if the sheet is large enough

        print("[SUCCESS] Range validation working")
        print(f"   Valid range {valid_range}: {is_valid}")
        print(f"   Large range {invalid_range}: {is_invalid}")


@pytest.mark.integration
@pytest.mark.sheets
@pytest.mark.slow
class TestGoogleOAuthTaskIntegration:
    """Integration tests for GoogleOAuthTask with Sheets authentication."""

    def test_sheets_oauth_task(self, skip_if_no_google_credentials: Any) -> None:
        """Test GoogleOAuthTask for Sheets-only authentication."""
        oauth_task = GoogleOAuthTask("sheets_auth", {"service_types": ["sheets"]})

        start_time = time.time()
        result = oauth_task.run()
        execution_time = time.time() - start_time

        # Verify task result
        assert isinstance(result, TaskResult)
        assert result.is_success()

        # Verify data structure
        assert "credentials" in result.data
        assert "scopes" in result.data
        assert "auth_info" in result.data
        assert "service_access" in result.data

        # Verify service access
        service_access = result.data["service_access"]
        assert service_access["sheets"] is True
        assert service_access["gmail"] is False

        print(f"[SUCCESS] Sheets OAuth task completed in {execution_time:.2f}s")
        print(f"   Scopes: {len(result.data['scopes'])}")
        print(f"   Service access: {service_access}")

    def test_combined_oauth_task(self, skip_if_no_google_credentials: Any) -> None:
        """Test GoogleOAuthTask for combined Gmail + Sheets authentication."""
        oauth_task = GoogleOAuthTask(
            "combined_auth", {"service_types": ["gmail", "sheets"]}
        )

        result = oauth_task.run()

        # Verify combined authentication
        assert result.is_success()
        service_access = result.data["service_access"]
        assert service_access["sheets"] is True
        assert service_access["gmail"] is True

        print("[SUCCESS] Combined OAuth task completed")
        print(f"   Gmail access: {service_access['gmail']}")
        print(f"   Sheets access: {service_access['sheets']}")


@pytest.mark.integration
@pytest.mark.sheets
@pytest.mark.slow
class TestReadGoogleSheetTaskIntegration:
    """Integration tests for ReadGoogleSheetTask with real Google Sheets API."""

    def test_basic_read_sheet_task(
        self,
        skip_if_no_google_credentials: Any,
        skip_if_no_test_spreadsheet: Any,
        test_spreadsheet_id: str,
    ) -> None:
        """Test basic ReadGoogleSheetTask functionality."""
        read_task = ReadGoogleSheetTask(
            "test_read_sheet",
            {
                "spreadsheet_id": test_spreadsheet_id,
                "include_headers": True,
            },
        )

        start_time = time.time()
        result = read_task.run()
        execution_time = time.time() - start_time

        # Verify task result structure
        assert isinstance(result, TaskResult)
        assert result.is_success()

        # Verify data structure
        assert "sheet_data" in result.data
        assert "values" in result.data
        assert "headers" in result.data
        assert "num_rows" in result.data
        assert "num_cols" in result.data
        assert "shape" in result.data
        assert "spreadsheet_info" in result.data

        # Verify values are 2D array
        values = result.data["values"]
        assert isinstance(values, list)
        if values:
            assert isinstance(values[0], list)

        print(f"[SUCCESS] Read sheet task completed in {execution_time:.2f}s")
        print(f"   Shape: {result.data['shape']}")
        print(
            f"   Headers: {result.data['headers'][:3]}..."
            if result.data["headers"]
            else "   No headers"
        )

    def test_read_specific_range(
        self,
        skip_if_no_google_credentials: Any,
        skip_if_no_test_spreadsheet: Any,
        test_spreadsheet_id: str,
    ) -> None:
        """Test reading specific range with ReadGoogleSheetTask."""
        # First get sheet names to build a valid range
        sheets_service = GoogleSheetsService()
        sheet_names = sheets_service.get_sheet_names(test_spreadsheet_id)
        first_sheet = sheet_names[0]

        range_a1 = f"{first_sheet}!A1:D10"

        read_task = ReadGoogleSheetTask(
            "test_read_range",
            {
                "spreadsheet_id": test_spreadsheet_id,
                "range": range_a1,
                "include_headers": True,
            },
        )

        result = read_task.run()

        # Verify range-specific read
        assert result.is_success()
        assert result.data["range_read"] == range_a1
        assert result.data["num_cols"] <= 4  # Should be at most 4 columns

        print("[SUCCESS] Read specific range task completed")
        print(f"   Range: {result.data['range_read']}")
        print(f"   Shape: {result.data['shape']}")

    def test_read_sheet_with_bounds(
        self,
        skip_if_no_google_credentials: Any,
        skip_if_no_test_spreadsheet: Any,
        test_spreadsheet_id: str,
    ) -> None:
        """Test reading sheet with row/column bounds."""
        # Get first sheet name
        sheets_service = GoogleSheetsService()
        sheet_names = sheets_service.get_sheet_names(test_spreadsheet_id)
        first_sheet = sheet_names[0]

        read_task = ReadGoogleSheetTask(
            "test_read_bounds",
            {
                "spreadsheet_id": test_spreadsheet_id,
                "sheet_name": first_sheet,
                "start_row": 1,
                "end_row": 8,
                "start_col": 1,
                "end_col": 6,
                "include_headers": True,
            },
        )

        result = read_task.run()

        # Verify bounded read
        assert result.is_success()
        assert result.data["num_cols"] <= 6  # Should be at most 6 columns
        assert result.data["num_rows"] <= 7  # 8 rows minus 1 for headers

        print("[SUCCESS] Read bounded sheet task completed")
        print("   Requested bounds: rows 1-8, cols 1-6")
        print(f"   Actual shape: {result.data['shape']}")

    def test_read_task_with_oauth_dependency(
        self,
        skip_if_no_google_credentials: Any,
        skip_if_no_test_spreadsheet: Any,
        test_spreadsheet_id: str,
    ) -> None:
        """Test ReadGoogleSheetTask with OAuth dependency."""
        # First run OAuth task
        oauth_task = GoogleOAuthTask("sheets_auth_dep", {"service_types": ["sheets"]})

        oauth_result = oauth_task.run()
        assert oauth_result.is_success()

        # Create read task with dependency
        read_task = ReadGoogleSheetTask(
            "test_read_with_dependency",
            {
                "spreadsheet_id": test_spreadsheet_id,
                "include_headers": True,
            },
            dependencies=[
                TaskDependency(
                    "credentials",
                    "sheets_auth_dep.credentials",
                    DependencyType.REQUIRED,
                )
            ],
        )

        # Simulate dependency resolution
        read_task.config["credentials"] = oauth_result.data["credentials"]

        result = read_task.run()

        assert result.is_success()
        print("[SUCCESS] ReadGoogleSheetTask with OAuth dependency working")

    def test_read_task_error_handling(self, skip_if_no_google_credentials: Any) -> None:
        """Test error handling with invalid spreadsheet ID."""
        read_task = ReadGoogleSheetTask(
            "test_error_handling",
            {
                "spreadsheet_id": "invalid_spreadsheet_id_12345",
                "include_headers": True,
            },
        )

        result = read_task.run()

        # Should handle error gracefully
        assert isinstance(result, TaskResult)
        assert not result.is_success()

        print("[SUCCESS] Error handling working for invalid spreadsheet ID")

    @pytest.mark.slow
    def test_comprehensive_sheets_workflow(
        self,
        skip_if_no_google_credentials: Any,
        skip_if_no_test_spreadsheet: Any,
        test_spreadsheet_id: str,
    ) -> None:
        """Test comprehensive workflow with OAuth + Sheets reading."""
        workflow_start = time.time()

        # Step 1: Authenticate with combined scopes
        oauth_task = GoogleOAuthTask(
            "workflow_auth", {"service_types": ["gmail", "sheets"]}
        )

        oauth_result = oauth_task.run()
        assert oauth_result.is_success()

        # Step 2: Read sheet data
        read_task = ReadGoogleSheetTask(
            "workflow_read",
            {
                "spreadsheet_id": test_spreadsheet_id,
                "include_headers": True,
            },
        )

        # Use credentials from OAuth task
        read_task.config["credentials"] = oauth_result.data["credentials"]

        read_result = read_task.run()
        assert read_result.is_success()

        # Step 3: Analyze results
        values: List[List[str]] = read_result.data["values"]
        headers: List[str] = read_result.data["headers"]

        # Basic data validation
        assert isinstance(values, list)
        assert isinstance(headers, list)

        workflow_time = time.time() - workflow_start

        print("[SUCCESS] Comprehensive Google Sheets workflow completed:")
        print("   OAuth authentication: [SUCCESS]")
        print("   Sheet data reading: [SUCCESS]")
        print(f"   Data shape: {read_result.data['shape']}")
        print(f"   Headers count: {len(headers)}")
        print(f"   Total workflow time: {workflow_time:.2f} seconds")

        # Performance check
        assert workflow_time < 60, f"Workflow too slow: {workflow_time:.2f}s"
