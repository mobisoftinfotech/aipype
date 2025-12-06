"""
Conditional logic tutorial: Content quality checker using Python conditionals
------------------------------------------------------------------------------

This tutorial demonstrates how to implement conditional logic in your agent
pipelines using the declarative @task syntax. We'll build a content quality
checker that evaluates generated outlines and decides whether they meet
quality standards.

Key concepts demonstrated:
- Using Python conditionals within @task methods
- Raising exceptions for validation failures
- Using dependencies to create task ordering
- Separating validation from content generation

The pipeline:
1. Generate an article outline using llm() helper
2. Check if outline meets minimum word count (Python conditional)
3. Return appropriate status based on quality check

Prerequisites:
- Ensure Ollama is installed and running
- Ensure a Gemma model is available: `ollama pull gemma3:4b`
- Set OLLAMA_API_BASE environment variable if needed

Usage examples:
# Test success path (condition passes):
python tutorial/04_conditional_task.py

# Test failure path (condition fails):
MIN_WORD_COUNT=500 python tutorial/04_conditional_task.py
"""

import os
from typing import Annotated, Any, Dict, List, Optional

from aipype import PipelineAgent, task, llm, Depends, LLMTask
from aipype import print_header, print_message_box


DEFAULT_TOPIC = "machine learning fundamentals"
MIN_WORD_COUNT = 50
QUALITY_CHECK_RESULTS_TITLE = "QUALITY CHECK RESULTS"


class QualityCheckerAgent(PipelineAgent):
    """Agent that generates content and checks it against quality criteria.

    This demonstrates using Python conditionals in @task methods:
    - generate_outline: Creates the outline (no dependencies)
    - quality_check: Validates word count, depends on generate_outline

    The @task syntax replaces ConditionalTask with simple Python if/else logic.
    """

    @task
    def generate_outline(self) -> LLMTask:
        """Generate an article outline using the llm() helper.

        This task has no dependencies and runs first.

        Returns:
            LLMTask: Configured outline generation task
        """
        topic = self.config.get("topic", DEFAULT_TOPIC)

        return llm(
            prompt=(
                f"Create a detailed outline for an article about {topic}.\n"
                "- Use 4-6 main sections with descriptive headings\n"
                "- Add 2-3 bullet points under each section\n"
                "- Make it comprehensive and informative\n"
                "- Return only the outline in markdown format"
            ),
            model="gemma3:4b",
            provider="ollama",
            system="You are an expert technical writer who creates detailed, well-structured outlines.",
            temperature=0.3,
            max_tokens=600,
        )

    @task
    def quality_check(
        self,
        generate_outline: Annotated[str, Depends("generate_outline.content")],
    ) -> Dict[str, Any]:
        """Check if the outline meets quality standards.

        This task demonstrates:
        - Python conditionals replacing ConditionalTask
        - Validation logic as simple if/else
        - Returning different results based on conditions

        Args:
            generate_outline: The outline content (from generate_outline.content)

        Returns:
            dict: Quality check result with status and message
        """
        min_words = self.config.get("min_word_count", MIN_WORD_COUNT)
        word_count = len(generate_outline.split())

        if word_count >= min_words:
            # Quality check passed
            return {
                "status": "approved",
                "message": "Outline meets quality standards and is ready for article writing",
                "action_taken": "approve_content",
                "word_count": word_count,
                "min_required": min_words,
                "condition_met": True,
            }
        else:
            # Quality check failed
            return {
                "status": "rejected",
                "message": "Outline needs expansion - word count below minimum threshold",
                "action_taken": "request_expansion",
                "word_count": word_count,
                "min_required": min_words,
                "condition_met": False,
            }

    def display_results(self, sections: Optional[List[str]] = None) -> None:  # type: ignore[override]
        """Display the outline and quality check results."""
        # Use the framework's built-in display method for basic results
        super().display_results()

        # Show outline content and word count
        outline_result = self.context.get_result("generate_outline")
        if outline_result:
            content = outline_result.get("content", "")
            if content:
                word_count = len(content.split())
                print_message_box(f"GENERATED OUTLINE ({word_count} words)", [content])

        # Show quality check results
        quality_result = self.context.get_result("quality_check")
        if quality_result:
            status = quality_result.get("status", "unknown")
            condition_met = quality_result.get("condition_met", False)
            word_count = quality_result.get("word_count", 0)
            min_required = quality_result.get("min_required", 0)
            message = quality_result.get("message", "")

            status_icon = "approved" if status == "approved" else "rejected"
            condition_icon = "Yes" if condition_met else "No"

            quality_info = [
                f"Status: {status_icon} ({status})",
                f"Word Count: {word_count} / {min_required} minimum",
                f"Condition Met: {condition_icon}",
                f"Message: {message}",
            ]
            print_message_box(QUALITY_CHECK_RESULTS_TITLE, quality_info)
        else:
            print_message_box(
                QUALITY_CHECK_RESULTS_TITLE, ["No quality check results found"]
            )


if __name__ == "__main__":
    print_header("CONDITIONAL LOGIC TUTORIAL - QUALITY CHECKER")

    # Configuration
    min_words = int(os.getenv("MIN_WORD_COUNT", str(MIN_WORD_COUNT)))

    print(f"Topic: {DEFAULT_TOPIC}")
    print(f"Minimum word count requirement: {min_words}")

    # Run the quality checker agent
    agent = QualityCheckerAgent(
        name="quality-checker-agent",
        config={"topic": DEFAULT_TOPIC, "min_word_count": min_words},
    )

    agent.run()
    agent.display_results()
