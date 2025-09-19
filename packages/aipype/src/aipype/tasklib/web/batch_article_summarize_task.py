"""Reusable batch article summarization task for the aipype framework."""

from typing import Any, Dict, List, override
from ...base_task import BaseTask
from ...task_dependencies import TaskDependency
from ...task_result import TaskResult
from ...llm_task import LLMTask


class BatchArticleSummarizeTask(BaseTask):
    """Reusable task that creates individual summaries for each article using separate LLM calls.

    This task processes a list of articles and creates summaries with configurable length.
    It includes content validation, error handling, and detection of generic LLM responses.

    Configuration Options:
        - summary_length: Target summary length in characters (default: 1000)
        - content_limit: Maximum content length to send to LLM (default: 3000)
        - min_content_length: Minimum content length to process (default: 50)
        - llm_provider: LLM provider to use (default: "openai")
        - llm_model: LLM model to use (default: "gpt-4o-mini")
        - temperature: LLM temperature (default: 0.3)
        - max_tokens: Maximum tokens for LLM response (default: 300)
    """

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        dependencies: List[TaskDependency] | None = None,
    ):
        super().__init__(name, config)
        self.dependencies = dependencies or []

    @override
    def get_dependencies(self) -> List[TaskDependency]:
        return self.dependencies

    @override
    def run(self) -> TaskResult:
        # Get articles from config (populated by dependency resolution)
        articles = self.config.get("articles", [])
        if not articles:
            error_msg = "BatchArticleSummarizeTask validation failed: No articles provided for summarization"
            self.logger.error(error_msg)
            return TaskResult.failure(
                error_message=error_msg,
                metadata={
                    "task_type": "batch_article_summarize",
                    "error_type": "ValueError",
                },
            )

        # Get configuration with defaults
        summary_length = self.config.get("summary_length", 1000)
        content_limit = self.config.get("content_limit", 3000)
        min_content_length = self.config.get("min_content_length", 50)
        llm_provider = self.config.get("llm_provider", "openai")
        llm_model = self.config.get("llm_model", "gpt-4o-mini")
        temperature = self.config.get("temperature", 0.3)
        max_tokens = self.config.get("max_tokens", 300)

        summaries: List[Dict[str, Any]] = []
        total_tokens = 0

        for i, article in enumerate(articles, 1):
            # Create individual summarization prompt
            title = article.get("title", "Untitled")
            url = article.get("url", "Unknown")
            content = article.get("content", "")

            # Debug logging to see what content we're getting
            self.logger.info(f"Processing article {i}: {title[:50]}...")
            self.logger.info(f"Content length: {len(content)} characters")

            # Skip articles with no meaningful content
            if not content or len(content.strip()) < min_content_length:
                self.logger.warning(
                    f"Skipping article {i} due to insufficient content (length: {len(content)})"
                )
                summaries.append(
                    {
                        "article_index": i,
                        "title": title,
                        "url": url,
                        "summary": f"[Skipped: Insufficient content - only {len(content)} characters]",
                    }
                )
                continue

            # Limit content length to avoid token limits
            if len(content) > content_limit:
                content = content[: content_limit - 3] + "..."

            prompt = f"""You are an expert content summarizer. Please create a concise, informative summary of the following article.

ARTICLE DETAILS:
Title: {title}
URL: {url}

SUMMARIZATION REQUIREMENTS:
- Create a summary of approximately {summary_length} characters (including spaces)
- Focus on the key insights, main points, and important details
- Preserve technical accuracy and important context
- Make the summary comprehensive enough to understand the article's core message
- Include any specific data, statistics, or examples that are crucial
- If the content appears to be incomplete or corrupted, mention this in the summary
- Do NOT respond with generic messages like "How can I assist you today?"

ARTICLE CONTENT TO SUMMARIZE:
{content}

IMPORTANT: Please provide a substantive summary of the above content. If the content is insufficient or unclear, still attempt to extract whatever meaningful information is available and note the limitations.

Summary:"""

            # Create and run individual LLM task
            llm_task = LLMTask(
                f"summarize_article_{i}",
                {
                    "prompt_template": prompt,
                    "context": "You are an expert content summarizer.",
                    "llm_provider": llm_provider,
                    "llm_model": llm_model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )

            try:
                result = llm_task.run()
                if result.is_error():
                    raise Exception(result.error or "LLM task failed")
                summary = result.data.get("content", "")

                # Check for generic/unhelpful responses
                generic_phrases = [
                    "How can I assist you today?",
                    "It seems like your message is empty",
                    "I'm here to help",
                    "Feel free to share",
                ]

                if (
                    any(phrase in summary for phrase in generic_phrases)
                    or len(summary.strip()) < 50
                ):
                    self.logger.warning(
                        f"Article {i} received generic/insufficient LLM response: {summary[:100]}..."
                    )
                    summary = f"[LLM returned generic response - possible content issue. Original content length: {len(content)} chars]"

                summaries.append(
                    {"article_index": i, "title": title, "url": url, "summary": summary}
                )

                # Accumulate token usage
                if result.data and "usage" in result.data:
                    total_tokens += result.data["usage"].get("total_tokens", 0)

            except Exception as e:
                error_msg = f"BatchArticleSummarizeTask article processing failed: Failed to summarize article {i} ('{title}'): {str(e)}"
                self.logger.error(error_msg)
                summaries.append(
                    {
                        "article_index": i,
                        "title": title,
                        "url": url,
                        "summary": f"[Summarization failed: {str(e)}]",
                    }
                )

        # Format summaries for output
        formatted_summaries = "\n\n".join(
            [
                f"ARTICLE {s['article_index']} SUMMARY ({s['title']}):\n{s['summary']}"
                for s in summaries
            ]
        )

        result_data = {
            "summaries": summaries,
            "formatted_summaries": formatted_summaries,
            "total_articles": len(articles),
            "successful_summaries": len(
                [
                    s
                    for s in summaries
                    if not str(s.get("summary", "")).startswith(
                        "[Summarization failed:"
                    )
                ]
            ),
            "total_tokens": total_tokens,
            "model": llm_model,
            "provider": llm_provider,
            "summary_length": summary_length,
        }

        return TaskResult.success(
            data=result_data,
            metadata={
                "task_type": "batch_article_summarize",
                "articles_processed": len(articles),
                "successful_summaries": result_data["successful_summaries"],
                "total_tokens": total_tokens,
                "model": llm_model,
                "provider": llm_provider,
            },
        )
