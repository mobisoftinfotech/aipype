# aipype

Core AI agent framework with declarative pipeline orchestration.

## Installation

```bash
pip install aipype
```

## Quick Start

Use the `@task` decorator for clean, Pythonic agent definitions:

```python
from aipype import PipelineAgent, task, llm, search

class ResearchAgent(PipelineAgent):

    @task
    def find_sources(self) -> dict:
        """Search for articles - no dependencies, runs first."""
        return search(self.config["topic"], max_results=5)

    @task
    def summarize(self, find_sources: dict) -> str:
        """Summarize sources - dependency auto-inferred from parameter."""
        return llm(
            prompt=f"Summarize: {find_sources}",
            model="gpt-4o",
            temperature=0.3
        )

agent = ResearchAgent("research", {"topic": "AI trends"})
agent.run()
agent.display_results()
```

**Key Concepts:**
- `@task` decorator marks methods as pipeline tasks
- Parameter names matching task names create automatic dependencies
- `self.config` provides access to agent configuration
- Helper functions: `llm()`, `search()`, `mcp_server()`, `transform()`

## Environment Variables

```bash
# LLM Providers
export OPENAI_API_KEY=sk-your-key-here
export GEMINI_API_KEY=your-key-here
export OLLAMA_API_BASE=http://localhost:11434

# Search
export SERPER_API_KEY=your-serper-key-here
```

## Development

### Requirements
- Python ≥3.12

