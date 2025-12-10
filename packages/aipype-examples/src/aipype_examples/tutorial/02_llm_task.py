"""
LLM Task tutorial: Outline an article using Ollama (Gemma3:4b)
------------------------------------------------------------

This example demonstrates how to use the `llm()` helper function to generate
an article outline using the declarative @task syntax.

Key concepts:
- Use `@task` decorator to define pipeline tasks
- Use `llm()` helper for clean LLM task creation
- Return LLMTask from @task methods for automatic execution (delegation)
- Access configuration via `self.config`

The `llm()` helper simplifies LLM calls with sensible defaults:
- prompt: The prompt to send to the LLM
- model: Model identifier (e.g., "gpt-4o", "gemma3:4b")
- provider: LLM provider ("openai", "ollama", "anthropic")
- system: System/context message
- temperature: Randomness control (0.0-2.0)
- max_tokens: Maximum response tokens

Prerequisites:
- Ensure Ollama is installed and running.
- Ensure a Gemma model is available (this tutorial uses `gemma3:4b`).
  Example: `ollama pull gemma3:4b`

Environment configuration (API base):
- For Ollama, set `OLLAMA_API_BASE` to point to the server (default: 11434).
- Example: OLLAMA_API_BASE=http://localhost:11434
"""

import os
from typing import Optional, List

from aipype import PipelineAgent, task, llm, LLMTask
from aipype import print_header


DEFAULT_TOPIC = "AI agent frameworks"


class OutlineAgent(PipelineAgent):
    """Agent that generates an article outline using the declarative syntax.

    This agent demonstrates:
    - Using @task decorator to define pipeline tasks
    - Using llm() helper for clean LLM task creation
    - Returning LLMTask for delegation (automatic execution)
    """

    @task
    def outline_article(self) -> LLMTask:
        """Generate an article outline using the llm() helper.

        The llm() helper returns an LLMTask which is automatically executed
        by the framework (delegation pattern). The result is stored in the
        task context.

        Returns:
            LLMTask: Configured task for outline generation
        """
        topic = self.config.get("topic", DEFAULT_TOPIC)

        return llm(
            prompt=(
                f"Create a concise, hierarchical outline for an article about {topic}.\n"
                "- Use 5-7 top-level sections with clear headings.\n"
                "- Add 2-3 bullet points under each section.\n"
                "- Optimize for technical readers and clarity.\n"
                "Return the outline in markdown. Return only the outline, "
                "no other text."
            ),
            model="gemma3:4b",
            provider="ollama",
            system="You are an expert technical editor who creates clear, useful article outlines.",
            temperature=0.2,
            max_tokens=800,
        )

    def display_results(self, sections: Optional[List[str]] = None) -> None:  # type: ignore[override]
        """Display formatted outline after pipeline execution."""
        # Use the framework's built-in display method for basic results
        super().display_results()

        # Add custom content display using framework utilities
        self.context.display_result_content(
            "outline_article", "GENERATED OUTLINE (MARKDOWN)"
        )


if __name__ == "__main__":
    print_header("LLM TASK TUTORIAL - OUTLINE GENERATION")

    # Pick a topic; you can also pass this via CLI args or env in real usage
    topic = os.getenv("OUTLINE_TOPIC", DEFAULT_TOPIC)

    # Instantiate, run, and display results
    agent = OutlineAgent(name="outline-agent", config={"topic": topic})
    agent.run()
    agent.display_results()
