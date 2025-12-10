"""Example LLM agent demonstrating modern pipeline architecture with @task decorators."""

from typing import Annotated, Any, Dict

from aipype import (
    PipelineAgent,
    task,
    llm,
    Depends,
    LLMTask,
)
from aipype import print_header, print_message_box


class LLMAgent(PipelineAgent):
    """Example agent demonstrating LLM capabilities with declarative @task syntax.

    This agent shows:
    - Multiple LLM tasks with automatic dependency inference
    - Using llm() helper for clean task creation
    - Task chaining with Depends() for explicit field extraction
    - Synthesis task that combines results from multiple tasks
    """

    @task
    def question_answer(self) -> LLMTask:
        """Answer a question using LLM."""
        question = self.config.get("question", "What is artificial intelligence?")
        provider = self.config.get("default_provider", "openai")
        model = self.config.get("default_model", "gpt-4o-mini")

        return llm(
            prompt=f"Please answer this question: {question}",
            model=model,
            provider=provider,
            system="You are a helpful assistant that answers questions clearly and concisely.",
            temperature=0.7,
            max_tokens=200,
        )

    @task
    def summarize_content(self) -> LLMTask:
        """Summarize the provided content."""
        content = self.config.get(
            "content_to_summarize",
            "Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines.",
        )
        provider = self.config.get("default_provider", "openai")
        model = self.config.get("default_model", "gpt-4o-mini")

        return llm(
            prompt=f"Please summarize the following text in 2-3 sentences: {content}",
            model=model,
            provider=provider,
            system="You are an expert at summarizing content.",
            temperature=0.3,
            max_tokens=150,
        )

    @task
    def creative_writing(self) -> LLMTask:
        """Write a creative short story."""
        topic = self.config.get("story_topic", "a robot learning to paint")
        provider = self.config.get("default_provider", "openai")
        model = self.config.get("default_model", "gpt-4o-mini")

        return llm(
            prompt=f"Write a short story (2-3 paragraphs) about {topic}",
            model=model,
            provider=provider,
            system="You are a creative writer who crafts engaging stories.",
            temperature=0.9,
            max_tokens=300,
        )

    @task
    def synthesis(
        self,
        question_answer: Annotated[str, Depends("question_answer.content")],
        creative_writing: Annotated[str, Depends("creative_writing.content")],
    ) -> LLMTask:
        """Synthesize insights from the Q&A and creative writing tasks.

        This task demonstrates:
        - Multiple dependencies using Annotated[T, Depends()]
        - Combining results from earlier tasks
        - Creating a synthesis from multiple sources
        """
        provider = self.config.get("default_provider", "openai")
        model = self.config.get("default_model", "gpt-4o-mini")

        return llm(
            prompt=(
                f"Based on the Q&A response: '{question_answer}' "
                f"and the creative story: '{creative_writing}', "
                "write a brief reflection on how AI creativity and "
                "analytical thinking complement each other."
            ),
            model=model,
            provider=provider,
            system="You are a synthesis expert who can combine different analyses into a coherent summary.",
            temperature=0.6,
            max_tokens=200,
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all LLM task results."""
        completed_tasks = self.context.get_completed_tasks() if self.context else []
        failed_tasks = self.context.get_failed_tasks() if self.context else []

        summary: Dict[str, Any] = {
            "total_tasks": len(self.tasks),
            "completed_tasks": len(completed_tasks),
            "failed_tasks": len(failed_tasks),
            "task_results": [],
        }

        for task_name in completed_tasks:
            result = self.context.get_result(task_name)
            if result:
                task_info = {
                    "name": task_name,
                    "content_length": len(result.get("content", "")),
                    "usage": result.get("usage"),
                }
                summary["task_results"].append(task_info)

        return summary


def main() -> None:
    """Demonstrate the modern LLM agent with declarative syntax."""
    print_header("MODERN LLM AGENT DEMONSTRATION")

    print_message_box(
        "DECLARATIVE LLM ARCHITECTURE",
        [
            "This example shows the declarative @task syntax where:",
            "- LLM tasks use llm() helper for clean creation",
            "- Dependencies inferred from parameter names",
            "- Depends() extracts specific fields from task results",
            "- Framework handles all orchestration automatically",
        ],
    )

    agent = LLMAgent(
        "modern_llm_agent",
        {
            "default_provider": "openai",
            "default_model": "gpt-4o-mini",
            "question": "How will AI transform healthcare in the next decade?",
            "story_topic": "an AI doctor making a breakthrough discovery",
            "content_to_summarize": (
                "Recent advances in artificial intelligence have shown tremendous promise "
                "in healthcare applications. Machine learning algorithms can now analyze "
                "medical images with accuracy matching or exceeding human radiologists. "
                "Natural language processing helps extract insights from electronic health records."
            ),
            "enable_parallel": False,  # Sequential for clear demonstration
            "stop_on_failure": True,
        },
    )

    agent.run()
    agent.display_results()

    print()
    print_message_box(
        "LLM PIPELINE DEMONSTRATION COMPLETE",
        [
            "The pipeline automatically:",
            "- Resolved LLM task dependencies",
            "- Executed llm() tasks in order",
            "- Passed LLM results between tasks",
            "- Synthesized multiple AI responses",
        ],
    )


if __name__ == "__main__":
    main()
