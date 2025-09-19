"""
ConditionalTask tutorial: Content quality checker with conditional logic
-----------------------------------------------------------------------

This tutorial demonstrates how to use ConditionalTask to implement conditional
logic in your agent pipelines. We'll build a content quality checker that
evaluates generated outlines and decides whether they meet quality standards.

Key concepts demonstrated:
- ConditionalTask with condition_function and condition_inputs
- action_function for success path and else_function for failure path
- TaskDependency to wire outputs between tasks
- Built-in helper functions from conditional_task module
- Conditional execution based on content analysis

The pipeline:
1. Generate an article outline using LLMTask (from tutorial 3)
2. Use ConditionalTask to check if outline meets minimum word count
3. Execute different actions based on quality check result

Prerequisites:
- Ensure Ollama is installed and running
- Ensure a Gemma model is available: `ollama pull gemma3:4b`
- Set OLLAMA_API_BASE environment variable if needed

Environment configuration:
- Set `OLLAMA_API_BASE` to your Ollama server (default: http://localhost:11434)

Usage examples:
# Test success path (condition passes):
python tutorial/04-conditional-task.py

# Test failure path (condition fails):
MIN_WORD_COUNT=500 python tutorial/04-conditional-task.py

"""

import os

from typing import Any, Callable, List, Optional
from typing import override
from aipype import (
    PipelineAgent,
    BaseTask,
    LLMTask,
    ConditionalTask,
    TaskDependency,
    DependencyType,
)
from aipype import print_header, print_message_box


DEFAULT_TOPIC = "machine learning fundamentals"
MIN_WORD_COUNT = 50
QUALITY_CHECK_RESULTS_TITLE = "QUALITY CHECK RESULTS"


def word_count_condition(min_words: int) -> Callable[[str], bool]:
    """Create a condition function that checks if text meets minimum word count.

    Args:
        min_words: Minimum required word count

    Returns:
        Condition function that takes text and returns boolean
    """

    def condition(content: str) -> bool:
        word_count = len(content.split())
        return word_count >= min_words

    return condition


def quality_approval_action() -> Callable[[], dict[str, Any]]:
    """Create an action function that logs quality approval."""

    def action() -> dict[str, Any]:
        return {
            "status": "approved",
            "message": "Outline meets quality standards and is ready for article writing",
            "action_taken": "approve_content",
        }

    return action


def quality_rejection_action() -> Callable[[], dict[str, Any]]:
    """Create an action function that logs quality rejection."""

    def action() -> dict[str, Any]:
        return {
            "status": "rejected",
            "message": "Outline needs expansion - word count below minimum threshold",
            "action_taken": "request_expansion",
        }

    return action


class QualityCheckerAgent(PipelineAgent):
    """Agent that generates content and checks it against quality criteria.

    Pipeline flow:
    1. Generate outline using LLMTask
    2. Check outline quality using ConditionalTask
    3. Take appropriate action based on quality check
    """

    @override
    def setup_tasks(self) -> List[BaseTask]:
        topic = self.config.get("topic", DEFAULT_TOPIC)
        min_words = self.config.get("min_word_count", MIN_WORD_COUNT)

        outline_task = self._create_outline_task(topic)
        quality_check_task = self._create_quality_check_task(min_words)

        return [outline_task, quality_check_task]

    def _create_outline_task(self, topic: str) -> LLMTask:
        """Create an outline generation task (adapted from tutorial 3)."""
        return LLMTask(
            name="generate_outline",
            config={
                "llm_provider": "ollama",
                "llm_model": "gemma3:4b",
                "prompt_template": (
                    "Create a detailed outline for an article about ${topic}.\n"
                    "- Use 4-6 main sections with descriptive headings\n"
                    "- Add 2-3 bullet points under each section\n"
                    "- Make it comprehensive and informative\n"
                    "- Return only the outline in markdown format"
                ),
                "context": (
                    "You are an expert technical writer who creates detailed, well-structured outlines."
                ),
                "temperature": 0.3,
                "max_tokens": 600,
                "topic": topic,
            },
        )

    def _create_quality_check_task(self, min_words: int) -> ConditionalTask:
        """Create a conditional task that checks outline quality.

        This demonstrates key ConditionalTask concepts:
        - condition_function: Evaluates outline word count
        - condition_inputs: Specifies which config fields to pass to condition
        - action_function: Executed when condition is True (quality passes)
        - else_function: Executed when condition is False (quality fails)
        """
        return ConditionalTask(
            name="quality_check",
            config={
                # Condition configuration
                "condition_function": word_count_condition(min_words),
                "condition_inputs": ["outline_content"],
                # Action for when quality check passes
                "action_function": quality_approval_action(),
                "action_inputs": [],
                # Action for when quality check fails
                "else_function": quality_rejection_action(),
                "else_inputs": [],
                # Custom skip reason (shown when condition is False but no else_function)
                "skip_reason": f"Outline has fewer than {min_words} words",
            },
            dependencies=[
                TaskDependency(
                    name="outline_content",
                    source_path="generate_outline.content",
                    dependency_type=DependencyType.REQUIRED,
                )
            ],
        )

    @override
    def display_results(self, sections: Optional[List[str]] = None) -> None:
        """Display the outline and quality check results using framework utilities."""
        # Use the framework's built-in display method for basic results
        super().display_results()

        # Show outline content and word count using ultra-safe convenience methods!
        content = self.context.get_result_content("generate_outline")
        if content:
            word_count = len(content.split())
            print_message_box(f"GENERATED OUTLINE ({word_count} words)", [content])

        # Show quality check results using the new ConditionalTask helper method!
        quality_task: Optional[ConditionalTask] = None
        for task in self.tasks:
            if task.name == "quality_check" and isinstance(task, ConditionalTask):
                quality_task = task
                break

        if not quality_task:
            print_message_box(
                QUALITY_CHECK_RESULTS_TITLE, ["❌ Quality check task not found"]
            )
            return

        summary = quality_task.get_execution_summary(self.context)
        if not summary:
            print_message_box(
                QUALITY_CHECK_RESULTS_TITLE, ["❌ No quality check results found"]
            )
            return

        # Add emojis to the pre-formatted info for better display
        quality_info: List[str] = []
        for info in summary["quality_info"]:
            info_str = str(info)
            # Use simple string replacement mapping
            replacements = {
                "Condition Met: Yes": "Condition Met: ✅ Yes",
                "Condition Met: No": "Condition Met: ❌ No",
                "Executed: Yes": "Executed: ✅ Yes",
                "Executed: No": "Executed: ❌ No",
            }

            # Apply first matching replacement or use original
            updated_info = info_str
            for old_text, new_text in replacements.items():
                if old_text in info_str:
                    updated_info = info_str.replace(old_text, new_text)
                    break

            quality_info.append(updated_info)

        print_message_box(QUALITY_CHECK_RESULTS_TITLE, quality_info)


if __name__ == "__main__":
    print_header("CONDITIONAL TASK TUTORIAL - QUALITY CHECKER")

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
