# Testing Guide

This document provides comprehensive information about testing in the aipype project.

## Table of Contents

- [Quick Start](#quick-start)
- [Testing Philosophy](#testing-philosophy)
- [Running Tests](#running-tests)
- [Test Organization](#test-organization)
- [Manual Tests](#manual-tests)
- [Integration Test Setup](#integration-test-setup)
- [Writing Tests](#writing-tests)
- [Complete Development Workflow](#complete-development-workflow)

## Quick Start

```bash
# Run unit tests only (default)
./scripts/run_tests.sh

# Run all tests (unit + integration)
./scripts/run_tests.sh --all

# Run integration tests only
./scripts/run_tests.sh --integration

# Run with verbose output
./scripts/run_tests.sh --verbose

# Show help
./scripts/run_tests.sh --help
```

## Testing Philosophy

The aipype project follows a two-tier testing approach:

1. **Unit Tests**: Test individual functions/classes with mocks for external dependencies
   - Fast, isolated, run on every commit
   - Located in `tests/` directories
   - Use mocks for external services, APIs, and file I/O

2. **Integration Tests**: Test real functionality end-to-end without mocks
   - Test actual system behavior with real components
   - Located in `integration_tests/` directories
   - May require API keys, running services, or special setup

## Running Tests

### Quick Scripts

```bash
# Quick scripts
./scripts/run_tests.sh                   # Unit tests only (default)
./scripts/run_tests.sh --all             # Unit + integration tests
./scripts/run_tests.sh --integration     # Integration tests only
./scripts/run_tests.sh --verbose         # Verbose output
./scripts/run_tests.sh --help            # Show all options
```

### Manual Testing Commands

```bash
# Unit tests
uv run pytest packages/aipype/tests/ -v              # Unit tests for core
uv run pytest packages/aipype-extras/tests/ -v       # Test specific package

# Integration tests
uv run pytest packages/aipype/integration_tests/ -v  # Core integration tests
uv run pytest packages/aipype-examples/integration_tests/ -v  # Examples integration tests

# Run specific tests
uv run pytest -k "test_specific_function" -v         # Run specific tests

# Run tests by marker (specify package to avoid conftest.py conflicts)
uv run pytest packages/aipype-examples/integration_tests/ -m "ollama" -v      # Run Ollama tests
uv run pytest packages/aipype-examples/integration_tests/ -m "openai" -v      # Run OpenAI tests
uv run pytest packages/aipype-examples/integration_tests/ -m "not manual" -v  # Exclude manual tests
```

## Test Organization

### Test Markers

Tests use pytest markers to categorize and control test execution:

- `@pytest.mark.integration` - Integration test requiring external services
- `@pytest.mark.ollama` - Requires Ollama service running
- `@pytest.mark.openai` - Requires OpenAI API key
- `@pytest.mark.anthropic` - Requires Anthropic API key
- `@pytest.mark.gmail` - Requires Gmail API access
- `@pytest.mark.mcp` - Requires MCP server
- `@pytest.mark.slow` - Slow running test (>10 seconds)
- `@pytest.mark.manual` - Manual test requiring special setup (not run by default)

### Test Structure

```
packages/
├── aipype/
│   ├── tests/                    # Unit tests
│   │   ├── framework/            # Framework tests
│   │   └── tasks/                # Task tests
│   └── integration_tests/        # Integration tests
│
├── aipype-examples/
│   ├── tests/                    # Unit tests for examples
│   └── integration_tests/        # Integration tests for examples
│
├── aipype-extras/
│   └── tests/                    # Unit tests for extras
│
└── aipype-g/
    └── tests/                    # Unit tests for Google integrations
```

## Manual Tests

Some tests require special setup and are marked with `@pytest.mark.manual`. These tests are **excluded by default** and must be explicitly run.

### MCP (Model Context Protocol) Tests

MCP tests require NGrok setup and are marked as manual because:
- **Anthropic blocks localhost URLs for security** - Cannot use `http://127.0.0.1:8765/sse`
- They need ANTHROPIC_API_KEY environment variable
- They **require NGrok tunnel** to expose MCP server over HTTPS
- Tests will **skip automatically** if FILE_READER_MCP_URL is localhost or not set
- They incur API costs
- Setup is more complex than automated tests

**Why NGrok is Required:**
Anthropic's API refuses to connect to localhost/127.0.0.1 MCP servers for security reasons. You must expose your local MCP server via a public HTTPS URL using NGrok.

**Running MCP Tests Manually:**

```bash
# Terminal 1: Start MCP server
python packages/aipype-examples/src/aipype_examples/mcp_servers/file_reader_server.py

# Terminal 2: Start NGrok tunnel
ngrok http 8765
# Copy the HTTPS forwarding URL (e.g., https://abc123.ngrok-free.app)

# Terminal 3: Set environment variables and run tests
export ANTHROPIC_API_KEY=your_api_key_here
export FILE_READER_MCP_URL=https://abc123.ngrok-free.app/sse

# Run MCP manual tests (must specify package directory to avoid conftest.py conflicts)
uv run pytest packages/aipype-examples/integration_tests/ -m "mcp and manual" -v

# Or run all manual tests in the examples package:
uv run pytest packages/aipype-examples/integration_tests/ -m manual -v
```

**What Happens Without NGrok:**
- If FILE_READER_MCP_URL is not set: Tests skip with message about needing NGrok URL
- If FILE_READER_MCP_URL contains localhost (127.0.0.1): Tests skip with message that Anthropic blocks localhost
- Only with valid NGrok HTTPS URL: Tests run and connect to MCP server

**Note**: MCP tests are excluded from `./scripts/run_tests.sh --all` by default. To run them, specify the package directory and use the manual marker as shown above.

## Integration Test Setup

### Prerequisites by Test Type

#### Ollama Tests
- Ollama service running on `http://localhost:11434`
- Required model installed (e.g., `ollama pull gemma3:1b`)

```bash
ollama serve
ollama pull gemma3:1b
uv run pytest -m ollama -v
```

#### OpenAI Tests
- `OPENAI_API_KEY` environment variable set

```bash
export OPENAI_API_KEY=sk-your-key-here
uv run pytest -m openai -v
```

#### Anthropic Tests
- `ANTHROPIC_API_KEY` environment variable set

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
uv run pytest -m anthropic -v
```

#### Gmail/Google API Tests
- Google Cloud project with Gmail/Sheets API enabled
- OAuth2 credentials file
- First-time authentication (browser will open for OAuth consent)

See [Google APIs Integration Tests](#google-apis-integration-tests) section below for detailed setup.

#### MCP Tests (Manual)
- MCP server running
- NGrok tunnel active
- `ANTHROPIC_API_KEY` set
- `FILE_READER_MCP_URL` set to NGrok URL

See [Manual Tests](#manual-tests) section for detailed setup.

## Google APIs Integration Tests

Integration tests for Google APIs (Gmail, Sheets) need OAuth2 authentication.

### Prerequisites

#### Google Cloud Console Setup
1. Create project at [console.cloud.google.com](https://console.cloud.google.com/)
2. Enable Gmail/Sheets APIs
3. Configure OAuth consent screen (External type)
4. Create OAuth client ID (Desktop application)
5. Download credentials JSON and save as `google_credentials.json` in the project root dir

#### First-Time Authentication
Tests will open browser for OAuth2 consent and save tokens automatically.

#### Running Gmail Tests

```bash
export GOOGLE_CREDENTIALS_FILE=./google_credentials.json
uv run pytest -m gmail -v packages/aipype-examples/integration_tests/
```

## Writing Tests

### Test Requirements

All new code must have comprehensive tests:

1. **Unit Tests**: Test individual functions/classes with mocks
   - Place in `tests/` directory
   - Mock external dependencies (APIs, file I/O, network calls)
   - Fast execution (<1s per test)
   - Example:
     ```python
     from unittest.mock import Mock, patch

     @patch('aipype.tasks.llm_task.litellm.completion')
     def test_llm_task_success(mock_completion):
         mock_completion.return_value = Mock(...)
         # Test with mock
     ```

2. **Integration Tests**: Test real functionality without mocks
   - Place in `integration_tests/` directory
   - Use real components, APIs, and services
   - Add appropriate markers (`@pytest.mark.openai`, etc.)
   - May be slower, may require setup
   - Example:
     ```python
     import pytest

     @pytest.mark.openai
     def test_llm_task_real_api(skip_if_openai_unavailable):
         # Test with real OpenAI API
         agent = MyAgent(...)
         result = agent.run()
         assert result.is_success()
     ```

3. **Manual Tests**: For complex setups requiring NGrok, etc.
   - Add `@pytest.mark.manual` marker
   - Document setup requirements in docstring
   - Example:
     ```python
     @pytest.mark.manual
     @pytest.mark.mcp
     def test_mcp_integration():
         """Test MCP integration.

         Note: Requires NGrok setup and ANTHROPIC_API_KEY.
         See README-TESTING.md for setup instructions.
         """
         # Test MCP functionality
     ```

### Coverage Goals

- Aim for high test coverage on new code
- Focus on critical paths and error handling
- Test both success and failure scenarios
- Test edge cases and boundary conditions

## Complete Development Workflow

Before committing changes, run all quality checks:

```bash
# Format code
./scripts/run_ruff.sh

# Type check
./scripts/run_type_checks.sh

# Run tests (unit + integration, excludes manual tests)
./scripts/run_tests.sh --all

# Or run all in one command:
./scripts/run_ruff.sh && ./scripts/run_type_checks.sh && ./scripts/run_tests.sh --all
```

### Pre-commit Checklist

- [ ] All new code has type annotations
- [ ] Unit tests written for new functionality
- [ ] Integration tests written where appropriate
- [ ] All tests pass (excluding manual tests)
- [ ] Code formatted with ruff
- [ ] Type checking passes with pyright
- [ ] Documentation updated if needed
- [ ] Examples added for new features

## Environment Variables for Testing

```bash
# LLM Providers
export OPENAI_API_KEY=sk-your-key-here
export GEMINI_API_KEY=your-key-here
export ANTHROPIC_API_KEY=sk-ant-your-key-here
export OLLAMA_API_BASE=http://localhost:11434

# MCP (Model Context Protocol)
export FILE_READER_MCP_URL=https://your-ngrok-url.ngrok-free.app/sse

# Search
export SERPER_API_KEY=your-serper-key-here

# Google APIs
export GOOGLE_CREDENTIALS_FILE=./google_credentials.json
export TEST_EMAIL_RECIPIENT=your-email@gmail.com
export TEST_SPREADSHEET_ID=your-spreadsheet-id
```

## Continuous Integration

When running tests in CI:

- Use `./scripts/run_tests.sh --all` to run unit and integration tests
- Manual tests are excluded by default
- Set required API keys as CI secrets
- Skip tests requiring local services (Ollama) unless CI supports them

## Troubleshooting

### ImportPathMismatchError: conftest.py conflicts

**Error**: `ImportPathMismatchError: ('integration_tests.conftest', ...)`

**Cause**: Running pytest from workspace root causes conftest.py import conflicts in monorepos.

**Solution**: Always specify the package directory when running tests:
```bash
# ✗ Wrong - causes conftest.py conflicts
uv run pytest -m manual

# ✓ Correct - specify package directory
uv run pytest packages/aipype-examples/integration_tests/ -m manual -v
```

### Tests are skipped

Check skip reasons: `uv run pytest -v -rs`

Common reasons:
- Missing API key: Set required environment variable
- Service not running: Start Ollama, MCP server, etc.
- Manual test: Explicitly include with `-m manual`

### Tests fail with API errors

- Verify API keys are correct and have sufficient credits
- Check rate limits
- Ensure services (Ollama, MCP server) are running and accessible

### Integration tests are slow

- Integration tests can be slower than unit tests
- Use `-m "not slow"` to exclude slow tests
- Consider running integration tests less frequently

## Additional Resources

- [Main README](README.md) - Project overview and setup
- [CLAUDE.md](CLAUDE.md) - Coding standards and architecture
- [API Documentation](https://mobisoftinfotech.com/tools/aipype/docs/)
