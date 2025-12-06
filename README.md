# aipype

<div align="center">
  <img src="docs/aipype-logo.svg" alt="aipype logo" width="200"/>

  [**Website**](https://mobisoftinfotech.com/tools/aipype/) | [**API Docs**](https://mobisoftinfotech.com/tools/aipype/docs/)
</div>

AI agent framework with declarative pipeline-based task orchestration.

`aipype` is mainly created for automation-heavy LLM-powered workflows. The aim is to create your agents as simple python scripts which you can write with the help of LLMs. 

### Like, Share and Subscribe :-)

This is a new project. So if you find the project useful, please give it a star on Github and help spread the word about it. 

## Status

**Alpha Release** - `aipype` is currently in Alpha stage. We are improving it rapidly and adding more reusable tasks. APIs may change between releases. 

## Examples

### Declarative Syntax (Recommended)

The recommended way to build agents in `aipype` is using the `@task` decorator for clean, Pythonic code:

```python
from aipype import PipelineAgent, task, llm, search, Depends
from typing import Annotated

class ResearchAgent(PipelineAgent):
    """Agent that searches for articles and generates a summary."""

    @task
    def find_sources(self) -> dict:
        """Search for relevant articles."""
        return search(self.config["topic"], max_results=5)

    @task
    def analyze(self, find_sources: dict) -> str:
        """Analyze sources - dependency auto-inferred from parameter name."""
        return llm(
            prompt=f"Analyze these sources: {find_sources}",
            model="gpt-4o",
            temperature=0.3
        )

    @task
    def write_summary(
        self,
        content: Annotated[str, Depends("analyze.content")]
    ) -> str:
        """Write summary - uses Depends() for explicit field extraction."""
        return llm(f"Write a summary based on: {content}", model="gpt-4o")


if __name__ == "__main__":
    agent = ResearchAgent("research", {"topic": "AI agent frameworks"})
    agent.run()
    agent.display_results()
```

**Key Features:**
- `@task` decorator marks methods as pipeline tasks
- Dependencies are automatically inferred from parameter names
- `Annotated[T, Depends("task.field")]` for explicit field extraction
- `self.config` provides access to agent configuration
- Tasks with no dependencies run first (topologically sorted)

### Legacy Syntax (setup_tasks)

For backwards compatibility or complex scenarios, you can use the explicit task setup:

```python
from typing import List
from aipype import BasePipelineAgent, BaseTask, LLMTask


class OutlineAgent(BasePipelineAgent):
    """Agent that creates an article outline using legacy syntax."""

    def setup_tasks(self) -> List[BaseTask]:
        topic = self.config.get("topic", "AI agent frameworks")

        return [
            LLMTask(
                name="outline_article",
                config={
                    "llm_provider": "ollama",
                    "llm_model": "gemma3:4b",
                    "prompt_template": (
                        "Create a concise outline for an article about ${topic}."
                    ),
                    "temperature": 0.2,
                    "max_tokens": 800,
                    "topic": topic,
                },
            )
        ]


if __name__ == "__main__":
    agent = OutlineAgent(name="outline-agent", config={"topic": "AI trends"})
    agent.run()
    agent.display_results()
```

For more details, check the following:

* Minimal Tutorial: `packages/aipype-examples/src/aipype_examples/tutorial`
* Examples: `packages/aipype-examples/src/aipype_examples/examples`


## Installation

```bash
# Core framework
pip install aipype

# With extras
pip install aipype aipype-extras

# With Google integrations
pip install aipype aipype-g
```

## Development Setup

```bash
# Install dependencies
uv sync --group dev

# Install pyright for type checking (requires Node.js)
npm install -g pyright
```

## Development Tools

### Code Formatting and Linting (ruff)

```bash
# Quick script (formats and lints all packages)
./scripts/run_ruff.sh

# Manual commands
ruff format packages/                    # Format code
ruff check packages/                     # Lint code
ruff check packages/ --fix              # Auto-fix issues
ruff format packages/aipype/           # Format specific package
```

### Type Checking (pyright)

```bash
# Quick script (checks all packages)
./scripts/run_type_checks.sh

# New unified type checker
python scripts/check_types.py                           # Check all packages
python scripts/check_types.py --package aipype        # Check specific package
python scripts/check_types.py --package aipype-extras # Check extras package
python scripts/check_types.py --summary                 # Show only summary
```

### Testing

For comprehensive testing documentation, see **[README-TESTING.md](README-TESTING.md)**.

Quick commands:

```bash
# Run unit tests (default)
./scripts/run_tests.sh

# Run all tests (unit + integration, excludes manual tests)
./scripts/run_tests.sh --all

# Complete development workflow (before committing)
./scripts/run_ruff.sh && ./scripts/run_type_checks.sh && ./scripts/run_tests.sh --all
```

## Environment Variables

⚠️ **Security Note**: Never commit API keys or credentials to version control. Use `.env` files (which are gitignored) or environment variables for sensitive data.

```bash
# LLM Providers
export OPENAI_API_KEY=sk-your-key-here
export OPENAI_API_BASE=https://api.openai.com/v1  # Optional
export GEMINI_API_KEY=your-key-here
export OLLAMA_API_BASE=http://localhost:11434

# Search
export SERPER_API_KEY=your-serper-key-here

# Google APIs
export GOOGLE_CREDENTIALS_FILE=path/to/google_credentials.json
export TEST_EMAIL_RECIPIENT=your-email@gmail.com
```

## Contributing

We welcome contributions! Please follow these guidelines to ensure code quality and consistency.

### Code Quality Requirements

**All contributions must pass these checks before being merged:**

1. **Code Formatting**: Code must be formatted with ruff
2. **Linting**: Code must pass ruff linting checks
3. **Type Checking**: Code must pass pyright type checking
4. **Testing**: All new code must have comprehensive tests

### Development Workflow

1. **Fork and clone** the repository
2. **Set up development environment**:
   ```bash
   uv sync --group dev
   npm install -g pyright
   ```

3. **Make your changes** following the coding standards
4. **Write tests** for any new functionality
5. **Run all quality checks**:
   ```bash
   # This must pass without errors:
   ./scripts/run_ruff.sh && ./scripts/run_type_checks.sh && ./scripts/run_tests.sh --all
   ```

6. **Commit and push** your changes
7. **Submit a pull request** with a clear description

### Code Style Guidelines

- **Type Annotations**: All functions must have complete type annotations
- **Error Handling**: Use `TaskResult.success()` and `TaskResult.failure()` pattern for operational errors
- **Documentation**: Add docstrings for all public functions and classes
- **Testing**: Write both unit tests (with mocks) and integration tests (no mocks)

### Testing Requirements

- **Unit Tests**: Test individual functions/classes with mocks for external dependencies
- **Integration Tests**: Test real functionality end-to-end without mocks
- **Coverage**: Aim for high test coverage on new code
- **Test Organization**: Place tests in appropriate `tests/` or `integration_tests/` directories

For detailed testing guidelines and setup, see **[README-TESTING.md](README-TESTING.md)**.

### Pull Request Guidelines

- **Clear Description**: Explain what your PR does and why
- **Small Changes**: Keep PRs focused and reasonably sized
- **Update Documentation**: Update README/docstrings if needed
- **Quality Checks**: Ensure all automated checks pass
- **Examples**: Add examples for new features when appropriate

## Integration Tests

For detailed information about running integration tests, including:
- Google APIs (Gmail, Sheets) OAuth2 setup
- MCP (Model Context Protocol) tests with NGrok
- OpenAI, Anthropic, and Ollama test configuration
- Manual tests requiring special setup

See **[README-TESTING.md](README-TESTING.md)** for complete documentation.

## Publishing Releases

### For Maintainers

Publish to production PyPI:

```bash
export UV_PUBLISH_TOKEN=your-token-here
./scripts/publish_to_pypi.sh --prod
```

Verify published packages:

```bash
./scripts/create_test_project.sh /tmp/verify-release
```

## Creators

`aipype` is created by [Mobisoft Infotech](https://mobisoftinfotech.com/).

## License

This project is distributed under MIT License. So you can use it in your open source as well as commercial projects as you see fit. However, if you are doing something cool with it, we would really appreciate it if you let us know about it at: info@mobisoftinfotech.com
