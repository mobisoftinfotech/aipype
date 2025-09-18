# aipype-tl-google

Google API integrations for the aipype framework.

## Installation

```bash
pip install aipype-tl-google
```

## Setup

### 1. Google Cloud Console
1. Create project at [console.cloud.google.com](https://console.cloud.google.com/)
2. Enable Gmail/Sheets APIs
3. Configure OAuth consent screen
4. Create OAuth client ID (Desktop application)
5. Download credentials JSON

### 2. Environment Variables
```bash
export GOOGLE_CREDENTIALS_FILE=path/to/google_credentials.json
export GMAIL_TOKEN_FILE=path/to/gmail_token.json
export SHEETS_TOKEN_FILE=path/to/sheets_token.json
```

## Usage

Check the examples package in the main github repo. 

## Development

### Requirements
- Python ≥3.12
- aipype (core framework)
- google-auth ≥2.0.0
