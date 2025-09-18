"""Google Sheets data models for structured sheet data representation."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Callable


@dataclass
class SheetRange:
    """Represents a range within a Google Sheet."""

    sheet_name: str
    start_row: int
    start_col: int
    end_row: Optional[int] = None
    end_col: Optional[int] = None

    @property
    def a1_notation(self) -> str:
        """Convert to A1 notation (e.g., 'Sheet1!A1:C10')."""
        start_col_letter = self._col_num_to_letter(self.start_col)

        if self.end_row is None and self.end_col is None:
            # Single cell
            range_str = f"{start_col_letter}{self.start_row}"
        elif self.end_row is None:
            # Single row, multiple columns
            end_col_letter = self._col_num_to_letter(self.end_col or self.start_col)
            range_str = (
                f"{start_col_letter}{self.start_row}:{end_col_letter}{self.start_row}"
            )
        elif self.end_col is None:
            # Single column, multiple rows
            range_str = (
                f"{start_col_letter}{self.start_row}:{start_col_letter}{self.end_row}"
            )
        else:
            # Full range
            end_col_letter = self._col_num_to_letter(self.end_col)
            range_str = (
                f"{start_col_letter}{self.start_row}:{end_col_letter}{self.end_row}"
            )

        return f"{self.sheet_name}!{range_str}"

    @staticmethod
    def _col_num_to_letter(col_num: int) -> str:
        """Convert column number (1-based) to letter (A, B, C, ..., AA, AB, etc.)."""
        result = ""
        while col_num > 0:
            col_num -= 1
            result = chr(65 + (col_num % 26)) + result
            col_num //= 26
        return result

    @staticmethod
    def from_a1_notation(a1_range: str) -> "SheetRange":
        """Create SheetRange from A1 notation (e.g., 'Sheet1!A1:C10')."""
        if "!" in a1_range:
            sheet_name, range_part = a1_range.split("!", 1)
        else:
            sheet_name = "Sheet1"  # Default sheet name
            range_part = a1_range

        if ":" in range_part:
            start_cell, end_cell = range_part.split(":", 1)
        else:
            start_cell = range_part
            end_cell = None

        # Parse start cell
        start_col, start_row = SheetRange._parse_cell(start_cell)

        # Parse end cell if present
        if end_cell:
            end_col, end_row = SheetRange._parse_cell(end_cell)
        else:
            end_col, end_row = None, None

        return SheetRange(
            sheet_name=sheet_name,
            start_row=start_row,
            start_col=start_col,
            end_row=end_row,
            end_col=end_col,
        )

    @staticmethod
    def _parse_cell(cell: str) -> tuple[int, int]:
        """Parse cell reference like 'A1' into (col_num, row_num)."""
        col_letters = ""
        row_digits = ""

        for char in cell:
            if char.isalpha():
                col_letters += char.upper()
            elif char.isdigit():
                row_digits += char

        # Convert column letters to number
        col_num = 0
        for char in col_letters:
            col_num = col_num * 26 + (ord(char) - ord("A") + 1)

        return col_num, int(row_digits)


@dataclass
class SheetData:
    """Represents data read from a Google Sheet."""

    spreadsheet_id: str
    sheet_name: str
    range_read: str
    # Pyright incorrectly reports these as partially unknown despite explicit type annotations
    # These are legitimate dataclass fields with proper generic type parameters
    values: List[List[str]] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    headers: List[str] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    metadata: Dict[str, Any] = field(default_factory=dict)  # pyright: ignore[reportUnknownVariableType]

    @property
    def num_rows(self) -> int:
        """Get number of data rows (excluding headers if present)."""
        return len(self.values)

    @property
    def num_cols(self) -> int:
        """Get number of columns."""
        return len(self.values[0]) if self.values else 0

    @property
    def has_headers(self) -> bool:
        """Check if headers are defined."""
        return len(self.headers) > 0

    @property
    def shape(self) -> tuple[int, int]:
        """Get shape as (rows, cols)."""
        return self.num_rows, self.num_cols

    def get_column(self, col_index: int) -> List[str]:
        """Get values from a specific column.

        Args:
            col_index: 0-based column index

        Returns:
            List of values in the column
        """
        if col_index >= self.num_cols:
            return []
        return [row[col_index] if col_index < len(row) else "" for row in self.values]

    def get_row(self, row_index: int) -> List[str]:
        """Get values from a specific row.

        Args:
            row_index: 0-based row index

        Returns:
            List of values in the row
        """
        if row_index >= self.num_rows:
            return []
        return self.values[row_index].copy()

    def get_cell(self, row_index: int, col_index: int) -> str:
        """Get value from a specific cell.

        Args:
            row_index: 0-based row index
            col_index: 0-based column index

        Returns:
            Cell value as string, empty string if out of bounds
        """
        if row_index >= self.num_rows or col_index >= len(self.values[row_index]):
            return ""
        return self.values[row_index][col_index]

    def to_dict_list(self) -> List[Dict[str, str]]:
        """Convert to list of dictionaries using headers as keys.

        Returns:
            List of dictionaries, one per data row

        Raises:
            ValueError: If no headers are defined
        """
        if not self.has_headers:
            raise ValueError("Cannot convert to dict list without headers")

        # Explicit type annotations for type checker
        result: List[Dict[str, str]] = []
        for row in self.values:
            row_dict: Dict[str, str] = {}
            for i, header in enumerate(self.headers):
                row_dict[header] = row[i] if i < len(row) else ""
            result.append(row_dict)

        return result

    def filter_rows(self, condition_func: Callable[[List[str]], bool]) -> "SheetData":
        """Filter rows based on a condition function.

        Args:
            condition_func: Function that takes a row (List[str]) and returns bool

        Returns:
            New SheetData with filtered rows
        """
        filtered_values = [row for row in self.values if condition_func(row)]

        return SheetData(
            spreadsheet_id=self.spreadsheet_id,
            sheet_name=self.sheet_name,
            range_read=self.range_read,
            values=filtered_values,
            headers=self.headers.copy(),
            metadata=self.metadata.copy(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "spreadsheet_id": self.spreadsheet_id,
            "sheet_name": self.sheet_name,
            "range_read": self.range_read,
            "values": self.values,
            "headers": self.headers,
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "shape": self.shape,
            "has_headers": self.has_headers,
            "metadata": self.metadata,
        }


@dataclass
class SpreadsheetInfo:
    """Represents information about a Google Spreadsheet."""

    spreadsheet_id: str
    title: str
    # Pyright incorrectly reports these as partially unknown despite explicit type annotations
    # These are legitimate dataclass fields with proper generic type parameters
    sheet_names: List[str] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    properties: Dict[str, Any] = field(default_factory=dict)  # pyright: ignore[reportUnknownVariableType]

    @property
    def num_sheets(self) -> int:
        """Get number of sheets in the spreadsheet."""
        return len(self.sheet_names)

    def has_sheet(self, sheet_name: str) -> bool:
        """Check if a sheet exists.

        Args:
            sheet_name: Name of the sheet to check

        Returns:
            True if sheet exists, False otherwise
        """
        return sheet_name in self.sheet_names

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "spreadsheet_id": self.spreadsheet_id,
            "title": self.title,
            "sheet_names": self.sheet_names,
            "num_sheets": self.num_sheets,
            "properties": self.properties,
        }


# Type aliases for Google Sheets API responses
SheetsApiResponse = Dict[str, Any]
SheetValues = List[List[Union[str, int, float]]]
