"""
Dependent tasks tutorial: Outline -> Article with automatic dependency inference
--------------------------------------------------------------------------------

This tutorial shows how one task can depend on another using the declarative
@task syntax with automatic dependency inference. We extend the outline example
by adding a second LLM task that uses the outline to write a short article.

Key concepts demonstrated:
- Automatic dependency inference from parameter names
- Parameter names matching task names create dependencies
- Annotated[T, Depends("task.field")] for explicit field extraction
- Tasks execute in dependency order (topologically sorted)

How dependency inference works:
1. If a parameter name matches another @task method name, it becomes a dependency
2. The parameter receives the full task result (dict) by default
3. Use Depends("task.field") to extract specific fields from the result

Environment configuration (API base):
- Set `OLLAMA_API_BASE` to your Ollama server (default: http://localhost:11434)
"""

import os
from typing import Annotated, Optional, List

from aipype import PipelineAgent, task, llm, Depends, LLMTask
from aipype import print_header


DEFAULT_TOPIC = "AI agent frameworks"


class OutlineToArticleAgent(PipelineAgent):
    """Agent with two dependent LLM tasks: outline -> article.

    This demonstrates automatic dependency inference:
    - Task 1 (outline_article): Generates a markdown outline (no dependencies)
    - Task 2 (write_article): Depends on outline_article via parameter name

    The framework automatically:
    1. Discovers both @task methods
    2. Infers that write_article depends on outline_article
    3. Executes outline_article first, then write_article
    """

    @task
    def outline_article(self) -> LLMTask:
        """Generate an article outline.

        This task has no parameters matching other task names,
        so it has no dependencies and runs first.

        Returns:
            LLMTask: Configured outline generation task
        """
        topic = self.config.get("topic", DEFAULT_TOPIC)

        return llm(
            prompt=(
                f"Create a concise, hierarchical outline for an article about {topic}.\n"
                "- Use 5-7 top-level sections with clear headings.\n"
                "- Add 2-3 bullet points under each section.\n"
                "- Optimize for technical readers and clarity.\n"
                "Return the outline in markdown. Output only the outline, no other text."
            ),
            model="gemma3:4b",
            provider="ollama",
            system="You are an expert technical editor who creates clear, useful article outlines.",
            temperature=0.2,
            max_tokens=800,
        )

    @task
    def write_article(
        self,
        outline_article: Annotated[str, Depends("outline_article.content")],
    ) -> LLMTask:
        """Write an article based on the outline.

        This task demonstrates explicit field extraction with Depends():
        - Parameter name: outline_article (matches the task name)
        - Depends("outline_article.content") extracts just the .content field

        The framework:
        1. Detects outline_article parameter matches a task name
        2. Sees Depends("outline_article.content") annotation
        3. Extracts .content from the outline_article result
        4. Passes it as the outline_article parameter

        Args:
            outline_article: The outline content (extracted via Depends)

        Returns:
            LLMTask: Configured article writing task
        """
        return llm(
            prompt=(
                "Write a short, well-structured article based on the following outline.\n\n"
                f"Outline (markdown):\n{outline_article}\n\n"
                "Requirements:\n"
                "- Professional, clear tone\n"
                "- 500-700 words\n"
                "- Preserve section structure\n"
                "- Output only the article, no other text."
            ),
            model="gemma3:4b",
            provider="ollama",
            system="You are an expert technical writer who crafts concise, structured articles.",
            temperature=0.5,
            max_tokens=1200,
        )

    def display_results(self, sections: Optional[List[str]] = None) -> None:  # type: ignore[override]
        """Display results using the framework display methods."""
        # Use the framework's built-in display method for basic results
        super().display_results()

        # Add custom detailed content display
        self.context.display_completed_results(
            [
                ("outline_article", "GENERATED OUTLINE"),
                ("write_article", "GENERATED ARTICLE"),
            ]
        )


if __name__ == "__main__":
    print_header("DEPENDENT TASKS TUTORIAL - OUTLINE TO ARTICLE")

    # Pick a topic; you can also pass this via CLI args or env in real usage
    topic = os.getenv("OUTLINE_TOPIC", DEFAULT_TOPIC)

    agent = OutlineToArticleAgent(
        name="outline-to-article-agent", config={"topic": topic}
    )
    agent.run()
    agent.display_results()
