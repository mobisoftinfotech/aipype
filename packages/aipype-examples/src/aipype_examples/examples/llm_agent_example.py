"""Example LLM agent demonstrating modern pipeline architecture with ContextualLLMTask."""

from typing import Any, Dict, List, override

from aipype import (
    PipelineAgent,
    LLMTask,
    BaseTask,
    TransformTask,
    TaskDependency,
    DependencyType,
)
from aipype import print_header, print_message_box


class LLMAgent(PipelineAgent):
    """Example agent that demonstrates modern LLM task capabilities with pipeline architecture."""

    @override
    def setup_tasks(self) -> List[BaseTask]:
        """Set up various LLM tasks to demonstrate modern contextual capabilities."""

        # Get configuration from agent config
        default_provider = self.config.get("default_provider", "openai")
        default_model = self.config.get("default_model", "gpt-4.1-nano")

        # Create and return all tasks
        tasks: List[BaseTask] = [
            self._create_context_setup_task(),
            self._create_qa_task(default_provider, default_model),
        ]

        # Add optional tasks based on configuration
        if self.config.get("content_to_summarize"):
            tasks.append(self._create_summarize_task(default_provider, default_model))

        tasks.append(
            self._create_creative_writing_task(default_provider, default_model)
        )

        if self.config.get("data_to_analyze") or self.config.get("analysis_provider"):
            tasks.append(
                self._create_data_analysis_task(default_provider, default_model)
            )

        if self.config.get("code_to_explain"):
            tasks.append(
                self._create_code_explanation_task(default_provider, default_model)
            )

        # Add synthesis task if multiple tasks are present
        if len(tasks) > 3:  # More than just setup, qa, and creative
            tasks.append(self._create_synthesis_task(default_provider, default_model))

        return tasks

    def _create_context_setup_task(self) -> TransformTask:
        """Create the initial context setup task.

        Returns:
            Configured TransformTask for setting up user inputs
        """
        return TransformTask(
            "setup_context",
            {
                "transform_function": lambda _: {  # pyright: ignore
                    "user_question": self.config.get(
                        "question", "What is artificial intelligence?"
                    ),
                    "content_to_summarize": self.config.get(
                        "content_to_summarize",
                        "Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines that can think and learn like humans. It encompasses various subfields including machine learning, natural language processing, computer vision, and robotics. AI has applications in many industries including healthcare, finance, transportation, and entertainment.",
                    ),
                    "story_topic": self.config.get(
                        "story_topic", "a robot learning to paint"
                    ),
                    "data_to_analyze": self.config.get(
                        "data_to_analyze",
                        "Sales data: Q1: $100k, Q2: $150k, Q3: $120k, Q4: $180k",
                    ),
                    "code_to_explain": self.config.get(
                        "code_to_explain",
                        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
                    ),
                },
                "output_name": "user_inputs",
                "validate_input": False,  # No input validation needed for static data generator
            },
        )

    def _create_qa_task(self, default_provider: str, default_model: str) -> LLMTask:
        """Create the Q&A task.

        Args:
            default_provider: Default LLM provider to use
            default_model: Default LLM model to use

        Returns:
            Configured LLMTask for question answering
        """
        return LLMTask(
            "question_answer",
            {
                "context": "You are a helpful assistant that answers questions clearly and concisely.",
                "prompt_template": "Please answer this question: ${user_question}",
                "llm_provider": default_provider,
                "llm_model": default_model,
                "temperature": 0.7,
                "max_tokens": 200,
            },
            [
                TaskDependency(
                    "user_question",
                    "setup_context.user_inputs",
                    DependencyType.REQUIRED,
                )
            ],
        )

    def _create_summarize_task(
        self, default_provider: str, default_model: str
    ) -> LLMTask:
        """Create the content summarization task.

        Args:
            default_provider: Default LLM provider to use
            default_model: Default LLM model to use

        Returns:
            Configured LLMTask for content summarization
        """
        return LLMTask(
            "summarize_content",
            {
                "context": "You are an expert at summarizing content.",
                "role": "content summarizer",
                "prompt_template": "Please summarize the following text in 2-3 sentences: ${content_to_summarize}",
                "llm_provider": default_provider,
                "llm_model": default_model,
                "temperature": 0.3,
                "max_tokens": 150,
            },
            [
                TaskDependency(
                    "content_to_summarize",
                    "setup_context.user_inputs",
                    DependencyType.REQUIRED,
                )
            ],
        )

    def _create_creative_writing_task(
        self, default_provider: str, default_model: str
    ) -> LLMTask:
        """Create the creative writing task.

        Args:
            default_provider: Default LLM provider to use
            default_model: Default LLM model to use

        Returns:
            Configured LLMTask for creative writing
        """
        return LLMTask(
            "creative_writing",
            {
                "context": "You are a creative writer who crafts engaging stories.",
                "role": "storyteller",
                "prompt_template": "Write a short story (2-3 paragraphs) about ${story_topic}",
                "llm_provider": default_provider,
                "llm_model": default_model,
                "temperature": 0.9,
                "max_tokens": 300,
            },
            [
                TaskDependency(
                    "story_topic", "setup_context.user_inputs", DependencyType.REQUIRED
                )
            ],
        )

    def _create_data_analysis_task(
        self, default_provider: str, default_model: str
    ) -> LLMTask:
        """Create the data analysis task.

        Args:
            default_provider: Default LLM provider to use
            default_model: Default LLM model to use

        Returns:
            Configured LLMTask for data analysis
        """
        return LLMTask(
            "analyze_data",
            {
                "context": "You are a data analyst who provides insights from data.",
                "role": "data scientist",
                "prompt_template": "Analyze the following data and provide 3 key insights: ${data_to_analyze}",
                "llm_provider": self.config.get("analysis_provider", default_provider),
                "llm_model": self.config.get("analysis_model", default_model),
                "temperature": 0.2,
                "max_tokens": 250,
            },
            [
                TaskDependency(
                    "data_to_analyze",
                    "setup_context.user_inputs",
                    DependencyType.REQUIRED,
                )
            ],
        )

    def _create_code_explanation_task(
        self, default_provider: str, default_model: str
    ) -> LLMTask:
        """Create the code explanation task.

        Args:
            default_provider: Default LLM provider to use
            default_model: Default LLM model to use

        Returns:
            Configured LLMTask for code explanation
        """
        return LLMTask(
            "explain_code",
            {
                "context": "You are a programming instructor who explains code clearly.",
                "role": "coding teacher",
                "prompt_template": "Explain what this code does and how it works: ${code_to_explain}",
                "llm_provider": default_provider,
                "llm_model": default_model,
                "temperature": 0.1,
                "max_tokens": 200,
            },
            [
                TaskDependency(
                    "code_to_explain",
                    "setup_context.user_inputs",
                    DependencyType.REQUIRED,
                )
            ],
        )

    def _create_synthesis_task(
        self, default_provider: str, default_model: str
    ) -> LLMTask:
        """Create the synthesis task that combines multiple results.

        Args:
            default_provider: Default LLM provider to use
            default_model: Default LLM model to use

        Returns:
            Configured LLMTask for synthesizing multiple responses
        """
        return LLMTask(
            "synthesis",
            {
                "context": "You are a synthesis expert who can combine different analyses into a coherent summary.",
                "prompt_template": "Based on the Q&A response: '${qa_response}' and the creative story: '${story_content}', write a brief reflection on how AI creativity and analytical thinking complement each other.",
                "llm_provider": default_provider,
                "llm_model": default_model,
                "temperature": 0.6,
                "max_tokens": 200,
            },
            [
                TaskDependency(
                    "qa_response",
                    "question_answer.content",
                    DependencyType.REQUIRED,
                ),
                TaskDependency(
                    "story_content",
                    "creative_writing.content",
                    DependencyType.REQUIRED,
                ),
            ],
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all LLM task results."""
        if not self.tasks:
            return {"message": "No tasks have been set up"}

        completed_tasks = self.context.get_completed_tasks() if self.context else []
        failed_tasks = self.context.get_failed_tasks() if self.context else []

        summary: Dict[str, Any] = {
            "total_tasks": len(self.tasks),
            "completed_tasks": len(completed_tasks),
            "failed_tasks": len(failed_tasks),
            "task_results": [],
            "total_cost": 0.0,
        }

        for task in self.tasks:
            if isinstance(task, LLMTask):
                task_info = {
                    "name": task.name,
                    "status": task.get_status().value,
                    "provider": task.config.get("llm_provider"),
                    "model": task.config.get("llm_model"),
                }

                if task.is_completed():
                    result = task.get_result()
                    if result is not None:
                        task_info.update(
                            {
                                "content_length": len(result.get("content", "")),
                                "token_usage": result.get("usage"),
                            }
                        )

                elif task.has_error():
                    task_info["error"] = task.get_error()

                summary["task_results"].append(task_info)

        return summary


def main() -> None:
    """Demonstrate the modern LLM agent with pipeline architecture."""
    print_header("MODERN LLM AGENT DEMONSTRATION")

    print_message_box(
        "PIPELINE LLM ARCHITECTURE",
        [
            "This example shows the new pipeline LLM architecture where:",
            "• LLM tasks use contextual templates with dependency injection",
            "• Data flows between LLM tasks via automatic resolution",
            "• Tasks can build upon results from previous LLM calls",
            "• Framework handles all orchestration automatically",
        ],
    )

    agent = LLMAgent(
        "modern_llm_agent",
        {
            "default_provider": "openai",
            "default_model": "gpt-4.1-nano",
            "question": "How will AI transform healthcare in the next decade?",
            "story_topic": "an AI doctor making a breakthrough discovery",
            "content_to_summarize": "Recent advances in artificial intelligence have shown tremendous promise in healthcare applications. Machine learning algorithms can now analyze medical images with accuracy matching or exceeding human radiologists. Natural language processing helps extract insights from electronic health records. Predictive models identify patients at risk of complications. AI-powered drug discovery platforms accelerate the development of new treatments. However, challenges remain including data privacy, algorithmic bias, regulatory approval, and physician adoption.",
            "enable_parallel": False,  # Sequential for clear demonstration
            "stop_on_failure": True,
        },
    )

    agent.setup_tasks()
    agent.run()
    agent.display_results()

    print()
    print_message_box(
        "LLM PIPELINE DEMONSTRATION COMPLETE",
        [
            "The pipeline automatically:",
            "✓ Resolved LLM task dependencies",
            "✓ Applied contextual template substitution",
            "✓ Passed LLM results between tasks",
            "✓ Synthesized multiple AI responses",
        ],
    )


if __name__ == "__main__":
    main()
