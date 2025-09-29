"""Read Google Sheet Task - Read data from Google Sheets as 2D arrays."""

from typing import List, Dict, Any, Optional

from typing import override

from aipype.base_task import BaseTask
from aipype.task_dependencies import TaskDependency
from aipype.task_result import TaskResult
from .google_sheets_service import GoogleSheetsService, GoogleSheetsError
from .google_sheets_models import SheetData


class ReadGoogleSheetTask(BaseTask):
    """Task that reads data from Google Sheets and returns as 2D arrays."""

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        dependencies: Optional[List[TaskDependency]] = None,
    ):
        """Initialize Read Google Sheet task.

        Args:
            name: Task name
            config: Configuration dictionary containing:

                - spreadsheet_id: Google Spreadsheet ID (required)
                - range: Range in A1 notation (e.g., 'Sheet1!A1:C10') (optional)
                - sheet_name: Name of sheet to read (alternative to range) (optional)
                - start_row: Starting row number (1-based) (optional, default: 1)
                - end_row: Ending row number (1-based) (optional)
                - start_col: Starting column number (1-based) (optional, default: 1)
                - end_col: Ending column number (1-based) (optional)
                - include_headers: Whether first row contains headers (default: True)
                - credentials: Pre-authenticated credentials (can be resolved from dependencies)
                - credentials_file: Path to OAuth2 credentials file (optional)
                - token_file: Path to OAuth2 token file (optional)
                - timeout: Request timeout in seconds (default: 30)

            dependencies: List of task dependencies
        """
        super().__init__(name, config)
        self.dependencies = dependencies or []
        self.validation_rules = {
            "defaults": {
                "start_row": 1,
                "start_col": 1,
                "include_headers": True,
                "timeout": 30,
            },
            "required": {
                "spreadsheet_id": str,
            },
            "types": {
                "spreadsheet_id": str,
                "range": str,
                "sheet_name": str,
                "start_row": int,
                "end_row": int,
                "start_col": int,
                "end_col": int,
                "include_headers": bool,
                "credentials_file": str,
                "token_file": str,
                "timeout": int,
            },
            "ranges": {
                "start_row": (1, None),
                "end_row": (1, None),
                "start_col": (1, None),
                "end_col": (1, None),
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
        """Read data from Google Sheet.

        Returns:
            TaskResult containing:
                - sheet_data: SheetData object (serialized)
                - values: 2D array of cell values
                - headers: List of header names (if include_headers=True)
                - num_rows: Number of data rows
                - num_cols: Number of columns
                - shape: Tuple of (rows, cols)
                - spreadsheet_info: Metadata about the spreadsheet
        """
        from datetime import datetime

        start_time = datetime.now()

        # Validate configuration using instance validation rules
        validation_failure = self._validate_or_fail(start_time)
        if validation_failure:
            return validation_failure

        # Get configuration values
        spreadsheet_id = self.config["spreadsheet_id"]
        range_a1 = self.config.get("range")
        sheet_name = self.config.get("sheet_name")
        start_row = self.config.get("start_row", 1)
        end_row = self.config.get("end_row")
        start_col = self.config.get("start_col", 1)
        end_col = self.config.get("end_col")
        include_headers = self.config.get("include_headers", True)
        credentials = self.config.get("credentials")
        credentials_file = self.config.get("credentials_file")
        token_file = self.config.get("token_file")
        timeout = self.config.get("timeout", 30)

        # Progress callback for detailed operation logging
        def progress_callback(message: str) -> None:
            self.logger.debug(f"[{self.name}] {message}")

        self.logger.info(
            f"Starting Google Sheets read task for spreadsheet: {spreadsheet_id[:8]}..."
        )

        try:
            # Initialize Google Sheets service
            progress_callback("Initializing Google Sheets service...")
            sheets_service = GoogleSheetsService(
                credentials=credentials,
                credentials_file=credentials_file,
                token_file=token_file,
                timeout=timeout,
            )

            # Determine what to read
            sheet_data: SheetData

            if range_a1:
                # Read specific range
                progress_callback(f"Reading specific range: {range_a1}")
                sheet_data = sheets_service.read_range(
                    spreadsheet_id=spreadsheet_id,
                    range_a1=range_a1,
                    include_headers=include_headers,
                    progress_callback=progress_callback,
                )

            elif sheet_name:
                # Read sheet with optional bounds
                progress_callback(f"Reading sheet '{sheet_name}' with bounds...")
                sheet_data = sheets_service.read_sheet(
                    spreadsheet_id=spreadsheet_id,
                    sheet_name=sheet_name,
                    start_row=start_row,
                    end_row=end_row,
                    start_col=start_col,
                    end_col=end_col,
                    include_headers=include_headers,
                    progress_callback=progress_callback,
                )

            else:
                # Read all data from first sheet
                progress_callback("Getting sheet names...")
                sheet_names = sheets_service.get_sheet_names(
                    spreadsheet_id=spreadsheet_id,
                    progress_callback=progress_callback,
                )

                if not sheet_names:
                    raise GoogleSheetsError("No sheets found in spreadsheet")

                first_sheet = sheet_names[0]
                progress_callback(f"Reading all data from first sheet: '{first_sheet}'")
                sheet_data = sheets_service.read_all_data(
                    spreadsheet_id=spreadsheet_id,
                    sheet_name=first_sheet,
                    include_headers=include_headers,
                    progress_callback=progress_callback,
                )

            # Get spreadsheet info for metadata
            progress_callback("Getting spreadsheet metadata...")
            spreadsheet_info = sheets_service.get_spreadsheet_info(
                spreadsheet_id=spreadsheet_id,
                progress_callback=progress_callback,
            )

            # Prepare result data
            result_data = {
                "sheet_data": sheet_data.to_dict(),
                "values": sheet_data.values,  # 2D array
                "headers": sheet_data.headers,
                "num_rows": sheet_data.num_rows,
                "num_cols": sheet_data.num_cols,
                "shape": sheet_data.shape,
                "spreadsheet_info": spreadsheet_info.to_dict(),
                "range_read": sheet_data.range_read,
                "include_headers": include_headers,
            }

            execution_time = (datetime.now() - start_time).total_seconds()

            self.logger.info(
                f"Google Sheets read task completed: {sheet_data.num_rows} rows, "
                f"{sheet_data.num_cols} cols from '{sheet_data.sheet_name}'"
            )

            return TaskResult.success(
                data=result_data,
                execution_time=execution_time,
                metadata={
                    "task_type": "read_google_sheet",
                    "spreadsheet_id": spreadsheet_id,
                    "sheet_name": sheet_data.sheet_name,
                    "range_read": sheet_data.range_read,
                    "num_rows": sheet_data.num_rows,
                    "num_cols": sheet_data.num_cols,
                    "include_headers": include_headers,
                },
            )

        except GoogleSheetsError as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"ReadGoogleSheetTask Sheets API operation failed: {str(e)}"
            self.logger.error(error_msg)

            return TaskResult.failure(
                error_message=error_msg,
                execution_time=execution_time,
                metadata={
                    "task_type": "read_google_sheet",
                    "spreadsheet_id": spreadsheet_id,
                    "error_type": "GoogleSheetsError",
                    "range_attempted": range_a1
                    or f"{sheet_name}({start_row},{start_col})",
                },
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"ReadGoogleSheetTask operation failed: {str(e)}"
            self.logger.error(error_msg)

            return TaskResult.failure(
                error_message=error_msg,
                execution_time=execution_time,
                metadata={
                    "task_type": "read_google_sheet",
                    "spreadsheet_id": spreadsheet_id,
                    "error_type": type(e).__name__,
                    "range_attempted": range_a1
                    or f"{sheet_name}({start_row},{start_col})",
                },
            )

    @staticmethod
    def create_range_config(
        spreadsheet_id: str,
        range_a1: str,
        include_headers: bool = True,
        credentials_file: Optional[str] = None,
        token_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Helper to create configuration for reading a specific range.

        Args:
            spreadsheet_id: Google Spreadsheet ID
            range_a1: Range in A1 notation (e.g., 'Sheet1!A1:C10')
            include_headers: Whether first row contains headers
            credentials_file: Path to OAuth2 credentials file
            token_file: Path to OAuth2 token file

        Returns:
            Configuration dictionary for ReadGoogleSheetTask
        """
        config = {
            "spreadsheet_id": spreadsheet_id,
            "range": range_a1,
            "include_headers": include_headers,
        }

        if credentials_file:
            config["credentials_file"] = credentials_file
        if token_file:
            config["token_file"] = token_file

        return config

    @staticmethod
    def create_sheet_config(
        spreadsheet_id: str,
        sheet_name: str,
        start_row: int = 1,
        end_row: Optional[int] = None,
        start_col: int = 1,
        end_col: Optional[int] = None,
        include_headers: bool = True,
        credentials_file: Optional[str] = None,
        token_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Helper to create configuration for reading a sheet with bounds.

        Args:
            spreadsheet_id: Google Spreadsheet ID
            sheet_name: Name of sheet to read
            start_row: Starting row (1-based)
            end_row: Ending row (1-based, optional)
            start_col: Starting column (1-based)
            end_col: Ending column (1-based, optional)
            include_headers: Whether first row contains headers
            credentials_file: Path to OAuth2 credentials file
            token_file: Path to OAuth2 token file

        Returns:
            Configuration dictionary for ReadGoogleSheetTask
        """
        config = {
            "spreadsheet_id": spreadsheet_id,
            "sheet_name": sheet_name,
            "start_row": start_row,
            "start_col": start_col,
            "include_headers": include_headers,
        }

        if end_row:
            config["end_row"] = end_row
        if end_col:
            config["end_col"] = end_col
        if credentials_file:
            config["credentials_file"] = credentials_file
        if token_file:
            config["token_file"] = token_file

        return config

    @staticmethod
    def create_full_sheet_config(
        spreadsheet_id: str,
        sheet_name: str,
        include_headers: bool = True,
        credentials_file: Optional[str] = None,
        token_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Helper to create configuration for reading an entire sheet.

        Args:
            spreadsheet_id: Google Spreadsheet ID
            sheet_name: Name of sheet to read
            include_headers: Whether first row contains headers
            credentials_file: Path to OAuth2 credentials file
            token_file: Path to OAuth2 token file

        Returns:
            Configuration dictionary for ReadGoogleSheetTask
        """
        config = {
            "spreadsheet_id": spreadsheet_id,
            "sheet_name": sheet_name,
            "include_headers": include_headers,
        }

        if credentials_file:
            config["credentials_file"] = credentials_file
        if token_file:
            config["token_file"] = token_file

        return config
