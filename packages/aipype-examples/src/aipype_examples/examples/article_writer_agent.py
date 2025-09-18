from typing import List, Dict, Any, override

from aipype import (
    PipelineAgent,
    BaseTask,
    SearchTask,
    LLMTask,
    TransformTask,
    ConditionalTask,
    TaskDependency,
    DependencyType,
    URLFetchTask,
    FileSaveTask,
    BatchArticleSummarizeTask,
    list_size_condition,
)


class ArticleWriterAgent(PipelineAgent):
    """ArticleWriterAgent

    This agent creates high-quality articles by:
    1. Searching for relevant articles
    2. Fetching content from search results (HTTP 200 only)
    3. Validating minimum article count (3 required)
    4. Creating summaries of each article using configurable LLM parameters
    5. Generating an outline from the summaries
    6. Writing a full article based on the outline
    7. Saving the article to the output directory

    Configuration Options:
        - summary_length: Target summary length in characters (default: 1000)
        - content_limit: Maximum content length to send to LLM (default: 3000)
        - min_content_length: Minimum content length to process (default: 50)
        - summarization_temperature: LLM temperature for summarization (default: 0.3)
        - summarization_max_tokens: Maximum tokens for summarization (default: 300)

    All orchestration is handled automatically by PipelineAgent based on task dependencies.
    The agent will fail with an error if fewer than 3 successful articles are downloaded.
    """

    @override
    def setup_tasks(self) -> List[BaseTask]:
        """Set up the article writing pipeline tasks declaratively.

        Returns:
            List of configured tasks with dependencies
        """
        # Get configuration
        search_keywords = self.config.get("search_keywords", "")
        guideline = self.config.get("guideline", "")
        llm_provider = self.config.get("llm_provider", "openai")
        llm_model = self.config.get("llm_model", "gpt-4o-mini")
        max_search_results = self.config.get("max_search_results", 10)
        output_dir = self.config.get("output_dir", "output")

        # Validate required configuration
        if not search_keywords:
            raise ValueError("search_keywords is required")
        if not guideline:
            raise ValueError("guideline is required")

        # Create and return all tasks
        return [
            self._create_search_task(search_keywords, max_search_results),
            self._create_url_extract_task(),
            self._create_fetch_content_task(),
            self._create_validation_task(),
            self._create_summarization_task(llm_provider, llm_model),
            self._create_outline_prompt_task(guideline),
            self._create_outline_generation_task(llm_provider, llm_model),
            self._create_article_prompt_task(guideline),
            self._create_article_writing_task(llm_provider, llm_model),
            self._create_save_task(search_keywords, output_dir),
        ]

    def _extract_urls_from_search(self, search_results: Dict[str, Any]) -> List[str]:
        """Extract URLs from search results.

        Args:
            search_results: Search results dictionary

        Returns:
            List of URLs
        """
        results: Any = search_results.get("results", [])
        if not isinstance(results, list):
            return []

        urls: List[str] = []
        for result in results:  # pyright: ignore
            if isinstance(result, dict) and "url" in result:
                url: Any = result["url"]  # pyright: ignore
                if isinstance(url, str):
                    urls.append(url)

        return urls

    def _create_outline_prompt(
        self, articles: List[Dict[str, Any]], guideline: str
    ) -> str:
        """Create prompt for outline generation.

        Args:
            articles: List of article content
            guideline: Writing guidelines

        Returns:
            Formatted prompt for outline generation
        """
        # Prepare articles content
        articles_content = self._prepare_articles_content(articles)

        prompt = f"""Based on the articles and guidelines given later, create a comprehensive and detailed outline for a new article that will be significantly better than the existing ones.

Please create a detailed, structured outline.

For each section, provide:
- Clear section title
- Detailed key points to cover (at least 5-7 points per section)
- Specific subtopics and examples to include
- Approximate word count target
- Suggested tone and approach

The outline should be comprehensive, well-researched, and ensure the new article provides unique insights and value beyond what's already available. Make it detailed enough that someone could write a 2000+ word article from this outline.

Guidelines for the new article:
{guideline}

Existing articles content:
{articles_content}

"""

        return prompt

    def _create_article_prompt(
        self, outline_content: str, guideline: str, articles_content: str
    ) -> str:
        """Create prompt for article generation.

        Args:
            outline_content: Generated outline content
            guideline: Writing guidelines
            articles_content: Existing articles content for reference

        Returns:
            Formatted prompt for article generation
        """
        prompt = f"""Write a comprehensive, high-quality article based on the outline and guidelines. This should be a substantial, well-researched piece of content.

Please write a complete, detailed article that:

1. FOLLOWS THE OUTLINE STRUCTURE - Use the provided outline as your roadmap
2. IS COMPREHENSIVE AND DETAILED - Aim for 4000-6000 words total
3. INCORPORATES RESEARCH INSIGHTS - Reference the research articles provided
4. PROVIDES UNIQUE VALUE - Offer fresh perspectives and insights
5. IS WELL-WRITTEN AND ENGAGING - Use clear, professional language
6. HAS A COMPELLING INTRODUCTION - Hook readers immediately
7. HAS A STRONG CONCLUSION - Summarize key points and provide forward-looking insights

Writing Requirements:
- Write in a professional, authoritative tone
- Use data and statistics when available, provide reference links
- Address both benefits and challenges honestly

The article should be comprehensive enough to serve as a definitive resource on the topic. Make it engaging, informative, and valuable to readers who want to understand the current state and future of AI in healthcare.

Guidelines for the article:
{guideline}

Article Outline:
{outline_content}

Existing articles content:
{articles_content}

"""

        return prompt

    def _prepare_articles_content(self, articles: List[Dict[str, Any]]) -> str:
        """Prepare articles content for LLM processing.

        Args:
            articles: List of article dictionaries

        Returns:
            Formatted articles content
        """
        content_parts: List[str] = []

        for i, article in enumerate(articles, 1):
            content_parts.append(f"Article {i}: {article.get('title', 'Untitled')}")
            content_parts.append(f"URL: {article.get('url', 'Unknown')}")

            content = article.get("content", "")
            # Limit content length to avoid token limits
            if len(content) > 2000:
                content = content[:1997] + "..."

            content_parts.append(f"Content: {content}")
            content_parts.append("")

        return "\n".join(content_parts)

    def _create_outline_prompt_with_summaries(
        self, summaries: str, guideline: str
    ) -> str:
        """Create prompt for outline generation using article summaries.

        Args:
            summaries: LLM-generated summaries of articles
            guideline: Writing guidelines

        Returns:
            Formatted prompt for outline generation
        """
        prompt = f"""Based on the article summaries and guidelines given later, create a comprehensive and detailed outline for a new article that will be significantly better than the existing ones.

Please create a detailed, structured outline that includes:

1. Introduction Section (200-300 words)
   - Hook to grab reader attention
   - Background context on AI in healthcare
   - Thesis statement and article overview
   - Key points to cover

2. Four to Six body sections (300-500 words each)
   - Current State of AI in Healthcare
   - Key Applications and Use Cases
   - Benefits and Advantages
   - Challenges and Limitations
   - Future Trends and Developments
   - Ethical Considerations

3. Conclusion (200-300 words)
   - Summary of key points
   - Future outlook
   - Call to action or final thoughts

For each section, provide:
- Clear section title
- Detailed key points to cover (at least 5-7 points per section)
- Specific subtopics and examples to include
- Approximate word count target
- Suggested tone and approach

The outline should be comprehensive, well-researched, and ensure the new article provides unique insights and value beyond what's already available. Make it detailed enough that someone could write a 2000+ word article from this outline.

Guidelines for the new article:
{guideline}

Article summaries to reference:
{summaries}

"""
        return prompt

    # Task creation methods
    def _create_search_task(
        self, search_keywords: str, max_search_results: int
    ) -> SearchTask:
        """Create the search articles task.

        Args:
            search_keywords: Keywords to search for
            max_search_results: Maximum number of search results

        Returns:
            Configured SearchTask
        """
        return SearchTask(
            "search_articles",
            {"query": search_keywords, "max_results": max_search_results},
        )

    def _create_url_extract_task(self) -> TransformTask:
        """Create the URL extraction task.

        Returns:
            Configured TransformTask for URL extraction
        """
        return TransformTask(
            "extract_urls",
            {
                "transform_function": lambda results: [  # pyright: ignore
                    r.get("url")  # pyright: ignore
                    for r in results  # pyright: ignore
                    if isinstance(r, dict) and "url" in r
                ]
                if isinstance(results, list)
                else [],
                "input_field": "search_results",
                "output_name": "urls",
            },
            [
                TaskDependency(
                    "search_results", "search_articles.results", DependencyType.REQUIRED
                )
            ],
        )

    def _create_fetch_content_task(self) -> URLFetchTask:
        """Create the article content fetching task.

        Returns:
            Configured URLFetchTask for fetching article content
        """
        return URLFetchTask(
            "fetch_article_content",
            {"max_urls": 10, "timeout": 30},
            [TaskDependency("urls", "extract_urls.urls", DependencyType.REQUIRED)],
        )

    def _create_validation_task(self) -> ConditionalTask:
        """Create the article count validation task.

        Returns:
            Configured ConditionalTask for validating minimum article count
        """

        def raise_insufficient_articles() -> Dict[str, Any]:
            """Raise error when insufficient articles are available."""
            raise RuntimeError(
                "Insufficient articles downloaded. At least 3 successful articles "
                "with HTTP 200 status are required to proceed with article generation."
            )

        return ConditionalTask(
            "validate_article_count",
            {
                "condition_function": list_size_condition(min_size=3),
                "condition_inputs": ["articles"],
                "else_function": raise_insufficient_articles,
                "skip_reason": "Less than 3 successful articles downloaded",
            },
            [
                TaskDependency(
                    "articles",
                    "fetch_article_content.articles",
                    DependencyType.REQUIRED,
                )
            ],
        )

    def _create_summarization_task(
        self, llm_provider: str, llm_model: str
    ) -> BatchArticleSummarizeTask:
        """Create the batch article summarization task with configurable parameters.

        Args:
            llm_provider: LLM provider to use
            llm_model: LLM model to use

        Returns:
            Configured BatchArticleSummarizeTask for individual article summaries
        """
        # Get configurable parameters from agent config
        summary_length = self.config.get("summary_length", 1000)
        content_limit = self.config.get("content_limit", 3000)
        min_content_length = self.config.get("min_content_length", 50)
        summarization_temperature = self.config.get("summarization_temperature", 0.3)
        summarization_max_tokens = self.config.get("summarization_max_tokens", 300)

        return BatchArticleSummarizeTask(
            "summarize_articles",
            {
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "summary_length": summary_length,
                "content_limit": content_limit,
                "min_content_length": min_content_length,
                "temperature": summarization_temperature,
                "max_tokens": summarization_max_tokens,
            },
            [
                TaskDependency(
                    "articles",
                    "fetch_article_content.articles",
                    DependencyType.REQUIRED,
                ),
                TaskDependency(
                    "validation_result",
                    "validate_article_count.condition_result",
                    DependencyType.REQUIRED,
                ),
            ],
        )

    def _create_outline_prompt_task(self, guideline: str) -> TransformTask:
        """Create the outline prompt preparation task.

        Args:
            guideline: Writing guidelines

        Returns:
            Configured TransformTask for preparing outline prompt
        """
        return TransformTask(
            "prepare_outline_prompt",
            {
                "transform_function": lambda formatted_summaries: self._create_outline_prompt_with_summaries(  # pyright: ignore
                    formatted_summaries,  # pyright: ignore
                    guideline,
                ),
                "input_fields": ["formatted_summaries"],
                "output_name": "outline_prompt",
            },
            [
                TaskDependency(
                    "formatted_summaries",
                    "summarize_articles.formatted_summaries",
                    DependencyType.REQUIRED,
                ),
            ],
        )

    def _create_outline_generation_task(
        self, llm_provider: str, llm_model: str
    ) -> LLMTask:
        """Create the outline generation task.

        Args:
            llm_provider: LLM provider to use
            llm_model: LLM model to use

        Returns:
            Configured LLMTask for generating article outline
        """
        return LLMTask(
            "generate_outline",
            {
                "prompt_template": "${outline_prompt}",
                "context": "You are an expert content strategist and writer who creates compelling article outlines.",
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "temperature": 0.7,
                "max_tokens": 2000,
            },
            [
                TaskDependency(
                    "outline_prompt",
                    "prepare_outline_prompt.outline_prompt",
                    DependencyType.REQUIRED,
                )
            ],
        )

    def _create_article_prompt_task(self, guideline: str) -> TransformTask:
        """Create the article prompt preparation task.

        Args:
            guideline: Writing guidelines

        Returns:
            Configured TransformTask for preparing article prompt
        """

        def create_prompt(outline_content: str, formatted_summaries: str) -> str:
            return self._create_article_prompt(
                outline_content,
                guideline,
                formatted_summaries,
            )

        return TransformTask(
            "prepare_article_prompt",
            {
                "transform_function": create_prompt,
                "input_fields": ["outline_content", "formatted_summaries"],
                "output_name": "article_prompt",
            },
            [
                TaskDependency(
                    "outline_content",
                    "generate_outline.content",
                    DependencyType.REQUIRED,
                ),
                TaskDependency(
                    "formatted_summaries",
                    "summarize_articles.formatted_summaries",
                    DependencyType.REQUIRED,
                ),
            ],
        )

    def _create_article_writing_task(
        self, llm_provider: str, llm_model: str
    ) -> LLMTask:
        """Create the article writing task.

        Args:
            llm_provider: LLM provider to use
            llm_model: LLM model to use

        Returns:
            Configured LLMTask for writing the full article
        """
        return LLMTask(
            "write_article",
            {
                "prompt_template": "${article_prompt}",
                "context": "You are an expert writer who creates high-quality, engaging articles.",
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "temperature": 0.8,
                "max_tokens": 10000,
            },
            [
                TaskDependency(
                    "article_prompt",
                    "prepare_article_prompt.article_prompt",
                    DependencyType.REQUIRED,
                )
            ],
        )

    def _create_save_task(self, search_keywords: str, output_dir: str) -> FileSaveTask:
        """Create the article saving task.

        Args:
            search_keywords: Keywords used for search (for filename)
            output_dir: Output directory

        Returns:
            Configured FileSaveTask for saving the article
        """
        return FileSaveTask(
            "save_article",
            {
                "title": f"Article on {search_keywords}",
                "output_dir": output_dir,
                "file_format": "md",
            },
            [
                TaskDependency(
                    "content", "write_article.content", DependencyType.REQUIRED
                )
            ],
        )

    @override
    def __str__(self) -> str:
        """String representation of the agent."""
        return (
            f"ArticleWriterAgent(name='{self.name}', pipeline_tasks={len(self.tasks)})"
        )


def main() -> None:
    agent = ArticleWriterAgent(
        name="ArticleWriterAgent",
        config={
            "search_keywords": "React Native vs Flutter in 2025",
            "guideline": "Write a comprehensive article to compare React Native and Flutter. Compare their pros and cons in 2025",
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "max_search_results": 10,
            "output_dir": "output",
        },
    )

    agent.run()
    agent.display_results()


if __name__ == "__main__":
    main()
