# Release Notes

## v0.2.0a1 (December 11, 2025)

### Highlights

This release introduces a **new declarative syntax** for defining AI pipelines, making it significantly easier to build agents with clean, Pythonic code. The new `@task` decorator and helper functions (`llm()`, `search()`) reduce boilerplate while maintaining full compatibility with the existing `BasePipelineAgent` approach.

### New Features

#### Declarative Task Syntax

- **`@task` decorator**: Mark methods as pipeline tasks with automatic dependency inference from parameter names
- **`PipelineAgent` class**: New primary agent class for declarative task definition (inherits from `BasePipelineAgent`)
- **`Depends()` class**: Explicit dependency path specification using `Annotated[T, Depends("task.field")]`
- **Automatic dependency resolution**: Parameter names matching task names are auto-wired as dependencies

#### Helper Functions

- **`llm()`**: Clean API for creating LLM tasks with sensible defaults
- **`search()`**: Simplified search task creation
- **`mcp_server()`**: Helper for MCP server configuration
- **`transform()`**: Helper for data transformation tasks

#### Example

```python
from aipype import PipelineAgent, task, llm, search

class ResearchAgent(PipelineAgent):
    @task
    def find_sources(self) -> dict:
        return search(self.config["topic"], max_results=5)

    @task
    def analyze(self, find_sources: dict) -> str:
        # find_sources parameter auto-injected from task result
        return llm(
            prompt=f"Analyze: {find_sources}",
            model="gpt-4o",
            temperature=0.3
        )

agent = ResearchAgent("research", {"topic": "AI trends"})
result = agent.run()
```

### Bug Fixes

- **BaseTask delegation dependency resolution**: Fixed issue where delegated BaseTask results were not accessible via consistent path format. Results are now wrapped with `.data` key for uniform access pattern (`task_name.data`).

### Breaking Changes

- **`PipelineAgent` renamed to `BasePipelineAgent`**: The original `PipelineAgent` class (using `setup_tasks()` method) is now called `BasePipelineAgent`
- **New `PipelineAgent` class**: `PipelineAgent` now refers to the new declarative agent class that uses `@task` decorators

**Migration for existing code:**

```python
# Before (v0.1.0a5)
from aipype import PipelineAgent

class MyAgent(PipelineAgent):
    def setup_tasks(self):
        return [...]

# After (v0.2.0a1)
from aipype import BasePipelineAgent

class MyAgent(BasePipelineAgent):
    def setup_tasks(self):
        return [...]
```

### Migration Notes

The `BasePipelineAgent` with `setup_tasks()` method remains fully supported for existing code and complex scenarios. New projects are encouraged to use the declarative `PipelineAgent` syntax with `@task` decorators for cleaner code.

### Package Versions

| Package | Version |
|---------|---------|
| aipype | 0.2.0a1 |
| aipype-extras | 0.2.0a1 |
| aipype-g | 0.2.0a1 |
| aipype-examples | 0.2.0a1 |

---

## v0.1.0a5 (Previous Release)

Initial alpha release with core framework features.
