# Google Sheets API Integration Tests

This directory contains integration tests for Google Sheets API functionality, including the `GoogleAuthService`, `GoogleSheetsService`, `GoogleOAuthTask`, and `ReadGoogleSheetTask` implementations.

## Overview

These tests verify real Google Sheets API operations with OAuth2 authentication:

- **Google Auth Tests** (`test_google_sheets_integration.py`): Unified authentication for multiple Google services
- **Google Sheets Service Tests**: Core Sheets API operations (read ranges, get spreadsheet info)
- **ReadGoogleSheetTask Tests**: Task-based sheet reading and data conversion to 2D arrays

## Prerequisites

### 1. Google Cloud Console Setup

1. **Create Google Cloud Project** (same as Gmail setup)
2. **Enable Google Sheets API**:
   - Navigate to "APIs & Services" > "Library"
   - Search for "Google Sheets API" and enable it
3. **Use existing OAuth2 credentials** (same as Gmail) or create new ones
4. **Test spreadsheet**: Create a Google Spreadsheet with some test data

### 2. Environment Setup

```bash
# Use existing Gmail credentials or create new Google credentials
export GOOGLE_CREDENTIALS_FILE=./google_credentials.json
export SHEETS_TOKEN_FILE=./sheets_token.json

# Required: Set test spreadsheet ID
export TEST_SPREADSHEET_ID=1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms

# Optional: Use separate credentials for Sheets-only testing
export SHEETS_CREDENTIALS_FILE=./sheets_credentials.json
export SHEETS_TOKEN_FILE=./sheets_token.json
```

### 3. Test Spreadsheet Requirements

- **Accessible Google Spreadsheet** with read permissions
- **Some data in the first sheet** for testing (at least 5 rows, 3 columns)
- **Headers in first row** recommended for header testing
- **Public example**: You can use Google's public example spreadsheet:
  `1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms`

## Running Tests

### Run All Google Sheets Integration Tests
```bash
# All Sheets tests
uv run pytest integration_tests/tasklib/google/test_google_sheets_integration.py -v

# With detailed output
uv run pytest integration_tests/tasklib/google/test_google_sheets_integration.py -v -s --log-cli-level=INFO
```

### Run Specific Test Categories
```bash
# Only authentication tests
uv run pytest integration_tests/tasklib/google/test_google_sheets_integration.py::TestGoogleAuthServiceIntegration -v

# Only Sheets service tests  
uv run pytest integration_tests/tasklib/google/test_google_sheets_integration.py::TestGoogleSheetsServiceIntegration -v

# Only ReadGoogleSheetTask tests
uv run pytest integration_tests/tasklib/google/test_google_sheets_integration.py::TestReadGoogleSheetTaskIntegration -v

# Only Sheets marker tests
uv run pytest integration_tests/ -m sheets -v

# Skip slow tests
uv run pytest integration_tests/tasklib/google/test_google_sheets_integration.py -v -m "not slow"
```

## Test Structure

### Google Auth Service Integration Tests

| Test | Description |
|------|-------------|
| `test_sheets_authentication_flow` | OAuth2 setup for Sheets-only access |
| `test_combined_authentication_flow` | OAuth2 setup for Gmail + Sheets combined access |

### Google Sheets Service Integration Tests

| Test | Description |
|------|-------------|
| `test_service_initialization` | GoogleSheetsService initialization and auth |
| `test_get_spreadsheet_info` | Spreadsheet metadata retrieval |
| `test_get_sheet_names` | Sheet name listing |
| `test_read_range_basic` | Basic range reading (A1:C5) |
| `test_read_sheet_with_bounds` | Bounded sheet reading (rows/cols) |
| `test_read_all_data` | Complete sheet data reading |
| `test_validate_range` | Range validation functionality |

### Google OAuth Task Integration Tests

| Test | Description |
|------|-------------|
| `test_sheets_oauth_task` | Sheets-only OAuth via GoogleOAuthTask |
| `test_combined_oauth_task` | Combined Gmail+Sheets OAuth task |

### ReadGoogleSheetTask Integration Tests

| Test | Description |
|------|-------------|
| `test_basic_read_sheet_task` | Basic ReadGoogleSheetTask functionality |
| `test_read_specific_range` | Range-specific reading (A1:D10) |
| `test_read_sheet_with_bounds` | Bounded reading with start/end rows/cols |
| `test_read_task_with_oauth_dependency` | Task dependency resolution |
| `test_read_task_error_handling` | Error handling for invalid spreadsheet |
| `test_comprehensive_sheets_workflow` | Complete OAuth→Read workflow |

## Example Usage Patterns

### 1. Unified Authentication Approach
```python
# Agent with unified auth
tasks = [
    GoogleOAuthTask("auth", {"service_types": ["gmail", "sheets"]}),
    ReadGoogleSheetTask("read_data", {
        "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
        "range": "Sheet1!A1:Z100"
    }, [TaskDependency("credentials", "auth.credentials")]),
    GmailListEmailsTask("list_emails", {...}, 
                       [TaskDependency("credentials", "auth.credentials")])
]
```

### 2. Sheets-Only Authentication
```python
# Sheets-only workflow
tasks = [
    GoogleOAuthTask("sheets_auth", {"service_types": ["sheets"]}),
    ReadGoogleSheetTask("inventory", {
        "spreadsheet_id": "your_inventory_sheet_id",
        "sheet_name": "Products",
        "include_headers": True
    }, [TaskDependency("credentials", "sheets_auth.credentials")])
]
```

### 3. Multiple Sheet Reads
```python
# Read multiple sheets/ranges
tasks = [
    GoogleOAuthTask("auth", {"service_types": ["sheets"]}),
    ReadGoogleSheetTask("customers", {
        "spreadsheet_id": "business_data_sheet",
        "sheet_name": "Customers",
        "start_row": 2, "end_row": 1000
    }, [TaskDependency("credentials", "auth.credentials")]),
    ReadGoogleSheetTask("products", {
        "spreadsheet_id": "business_data_sheet", 
        "range": "Products!A1:F100"
    }, [TaskDependency("credentials", "auth.credentials")])
]
```

## Test Data Patterns

Tests expect spreadsheet data in this format:
```
| Header1 | Header2 | Header3 | ... |
|---------|---------|---------|-----|
| Data1   | Data2   | Data3   | ... |
| ...     | ...     | ...     | ... |
```

## Performance Characteristics

- **Individual tests**: 1-10 seconds each
- **Full test suite**: 2-5 minutes depending on spreadsheet size and network
- **Network dependent**: Requires stable internet connection
- **Rate limited**: Google Sheets API has usage quotas

## Troubleshooting

### Common Issues

#### "Google credentials not available"
```bash
# Download credentials and set environment
export GOOGLE_CREDENTIALS_FILE=./google_credentials.json
```

#### "Test spreadsheet ID not provided"
```bash
# Set your test spreadsheet ID
export TEST_SPREADSHEET_ID=1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms
```

#### "Sheets API not enabled"
- Enable Google Sheets API in Google Cloud Console
- Verify project has proper API access

#### "Permission denied" on spreadsheet
- Ensure spreadsheet is accessible to your Google account
- Check sharing permissions on the spreadsheet
- Use a spreadsheet you own or have edit access to

### Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_CREDENTIALS_FILE` | `google_credentials.json` | OAuth2 credentials file |
| `SHEETS_TOKEN_FILE` | `sheets_token.json` | Sheets OAuth2 token storage file |
| `TEST_SPREADSHEET_ID` | (none) | **Required**: Test spreadsheet ID |
| `SHEETS_CREDENTIALS_FILE` | (none) | Optional Sheets-specific credentials |

## Integration with MI-Agents Framework

These tests validate Google Sheets integration patterns:
- **Unified authentication** across Google services
- **TaskResult patterns** for success/failure handling  
- **Dependency resolution** between authentication and data tasks
- **2D array output** for spreadsheet data processing
- **Error handling** for API failures and invalid inputs
- **Progress reporting** during long-running operations

The tests ensure Google Sheets tasks work correctly in pipeline workflows and can be combined with Gmail tasks and other MI-Agents components for comprehensive business automation scenarios.

## CI/CD Considerations

Google Sheets integration tests are **not suitable for automated CI/CD** due to:
- **OAuth2 flow** requiring interactive browser authentication
- **Real Google account** dependency
- **Test spreadsheet** requirements
- **API quotas** and rate limits

**Recommended approach**:
- Run Sheets tests **manually** during development
- Use **separate test suite** for CI/CD with mocked APIs
- Include in **nightly builds** with pre-authenticated service accounts