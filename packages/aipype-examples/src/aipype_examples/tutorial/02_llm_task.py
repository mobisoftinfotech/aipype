"""
LLM Task tutorial: Outline an article using Ollama (Gemma3:4b)
------------------------------------------------------------

This example demonstrates how to use the framework's `LLMTask` to generate
an article outline. It highlights key `LLMTask` features from `aipype/llm_task.py`:

- Required configuration: `llm_provider`, `llm_model`
- Prompt templating with `${variable}` placeholders via `prompt_template`
- Optional `context` and `role` to steer the model
- Safe result handling via `TaskResult` (includes `data["content"]`)
- Easy orchestration via `PipelineAgent`

Prerequisites:
- Ensure Ollama is installed and running.
- Ensure a Gemma model is available (this tutorial uses `gemma3:4b`).
  Example: `ollama pull gemma3:4b`

Environment configuration (API base):
- The framework's `LLMTask` reads provider-specific API base URLs from env vars.
- For Ollama, set `OLLAMA_API_BASE` to point to the server (default local port is 11434).
- Example `.env`:
    OLLAMA_API_BASE=http://localhost:11434
- If using a remote host or Docker mapping, update accordingly, e.g.:
    OLLAMA_API_BASE=http://192.168.1.50:11434
Note: `LLMTask` automatically loads `.env` via `dotenv` if present.
"""

import os

from typing import List, Optional
from typing import override
from aipype import PipelineAgent, BaseTask, LLMTask
from aipype import print_header


DEFAULT_TOPIC = "AI agent frameworks"


"""We directly instantiate `LLMTask` instead of subclassing it.

Notes on `LLMTask` behavior (from `aipype/llm_task.py`):
- Uses `prompt_template` with `${var}` substitution from the task `config` or context.
- Accepts `context` and `role` for a richer system message.
- Returns a `TaskResult` with `data` containing keys like `content`, `model`, `provider`.
- For provider `ollama`, function/tool calling is disabled by default (not needed here).
"""


class OutlineAgent(PipelineAgent):
    """Agent that orchestrates a single `LLMTask` to create an article outline.

    - The agent reads `topic` from its `config` and passes it to the task.
    - The task uses a `prompt_template` with `${topic}` placeholder.
    - Provider is `ollama` and model is `gemma3:4b` for local inference.
    """

    @override
    def setup_tasks(self) -> List[BaseTask]:
        topic = self.config.get("topic", DEFAULT_TOPIC)

        return [self._create_outline_task(topic)]

    def _create_outline_task(self, topic: str) -> LLMTask:
        return LLMTask(
            name="outline_article",
            config={
                # Required provider + model for LLMTask
                "llm_provider": "ollama",
                "llm_model": "gemma3:4b",
                # API base is read from env var OLLAMA_API_BASE (see header notes)
                # Prompt templating: `${topic}` will be resolved from this config
                "prompt_template": (
                    "Create a concise, hierarchical outline for an article about ${topic}.\n"
                    "- Use 5-7 top-level sections with clear headings.\n"
                    "- Add 2-3 bullet points under each section.\n"
                    "- Optimize for technical readers and clarity.\n"
                    "Return the outline in markdown. Return only the outline, "
                    "no other text."
                ),
                # Optional steering signals
                "context": (
                    "You are an expert technical editor who creates clear, useful article outlines."
                ),
                "role": "Senior tech editor",
                # Sampling and length
                "temperature": 0.2,
                "max_tokens": 800,
                # Template variables
                "topic": topic,
            },
        )

    @override
    def display_results(self, sections: Optional[List[str]] = None) -> None:
        """Display formatted outline after pipeline execution using framework utilities."""
        # Use the framework's built-in display method for basic results
        super().display_results()

        # Add custom content display using framework utilities - true one-liner!
        self.context.display_result_content(
            "outline_article", "GENERATED OUTLINE (MARKDOWN)"
        )


if __name__ == "__main__":
    print_header("LLM TASK TUTORIAL - OUTLINE GENERATION")

    # Pick a topic; you can also pass this via CLI args or env in real usage
    topic = os.getenv("OUTLINE_TOPIC", DEFAULT_TOPIC)

    # Instantiate, run, and display results like the example agent does
    agent = OutlineAgent(name="outline-agent", config={"topic": topic})
    agent.run()
    agent.display_results()
