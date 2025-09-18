"""
Dependent tasks tutorial: Outline → Article with LLMTask (Ollama Gemma3:4b)
--------------------------------------------------------------------------

This tutorial shows how one task can depend on another using the framework's
dependency system. We extend the outline example by adding a second LLM task
that uses the outline to write a short article.

Key concepts demonstrated:
- Two `LLMTask` instances orchestrated by a `PipelineAgent`
- `TaskDependency` wires output of one task into the config of another
- Prompt templating with `${variable}` placeholders
- Displaying both intermediate (outline) and final (article) results

Environment configuration (API base):
- Set `OLLAMA_API_BASE` to your Ollama server (default: http://localhost:11434)
- The framework auto-loads `.env` via `dotenv` if present.
"""

import os

from typing import List, Optional
from typing import override
from aipype import (
    PipelineAgent,
    BaseTask,
    LLMTask,
    TaskDependency,
    DependencyType,
)
from aipype import print_header


DEFAULT_TOPIC = "AI agent frameworks"


class OutlineToArticleAgent(PipelineAgent):
    """Agent with two dependent LLM tasks: outline → article.

    - Task 1 (`outline_article`): Generates a markdown outline
    - Task 2 (`write_article`): Consumes the outline result and writes an article
    """

    @override
    def setup_tasks(self) -> List[BaseTask]:
        topic = self.config.get("topic", DEFAULT_TOPIC)

        outline_task = self._create_outline_task(topic)
        article_task = self._create_article_task()

        return [outline_task, article_task]

    def _create_outline_task(self, topic: str) -> LLMTask:
        return LLMTask(
            name="outline_article",
            config={
                "llm_provider": "ollama",
                "llm_model": "gemma3:4b",
                # Prompt templating: `${topic}` will be resolved from this config
                "prompt_template": (
                    "Create a concise, hierarchical outline for an article about ${topic}.\n"
                    "- Use 5-7 top-level sections with clear headings.\n"
                    "- Add 2-3 bullet points under each section.\n"
                    "- Optimize for technical readers and clarity.\n"
                    "Return the outline in markdown. output only the outline, no other text."
                ),
                "context": (
                    "You are an expert technical editor who creates clear, useful article outlines."
                ),
                "role": "Senior tech editor",
                "temperature": 0.2,
                "max_tokens": 800,
                # Template variables
                "topic": topic,
            },
        )

    def _create_article_task(self) -> LLMTask:
        """Create an article-writing task that depends on the outline task output.

        Dependency wiring:
        - Declare a `TaskDependency` with target key `outline_content`
        - Source path `outline_article.content` maps Task1's `content` into Task2 config
        - In the prompt template, reference `${outline_content}`
        """
        return LLMTask(
            name="write_article",
            config={
                "llm_provider": "ollama",
                "llm_model": "gemma3:4b",
                # Use outline from the first task
                "prompt_template": (
                    "Write a short, well-structured article based on the following outline.\n\n"
                    "Outline (markdown):\n${outline_content}\n\n"
                    "Requirements:\n"
                    "- Professional, clear tone\n"
                    "- 500-700 words\n"
                    "- Preserve section structure\n"
                    "- output only the article, no other text."
                ),
                "context": (
                    "You are an expert technical writer who crafts concise, structured articles."
                ),
                "role": "Senior tech writer",
                "temperature": 0.5,
                "max_tokens": 1200,
            },
            dependencies=[
                TaskDependency(
                    name="outline_content",
                    source_path="outline_article.content",
                    dependency_type=DependencyType.REQUIRED,
                )
            ],
        )

    @override
    def display_results(self, sections: Optional[List[str]] = None) -> None:
        """Display results using the simple framework display method."""
        # Use the framework's built-in display method for basic results
        super().display_results()

        # Add custom detailed content display - ultra-clean with display_completed_results!
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
