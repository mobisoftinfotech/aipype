"""Article Writer Agent using @task decorators.

This agent demonstrates the declarative syntax for the aipype framework,
using @task decorators for automatic dependency inference and cleaner code.

Features:
1. Searching for relevant articles on a topic
2. Fetching content from search results
3. Validating minimum article count (3 required)
4. Creating summaries using LLM
5. Generating an outline from summaries
6. Writing a full article based on the outline
7. Saving the article to the output directory

Configuration Options:
    - search_keywords: Topic to search for (required)
    - guideline: Writing guidelines (required)
    - llm_provider: LLM provider (default: openai)
    - llm_model: LLM model (default: gpt-4o-mini)
    - max_search_results: Maximum search results (default: 10)
    - output_dir: Output directory (default: output)
"""

from typing import Annotated, Any, Dict, List

from aipype import (
    PipelineAgent,
    task,
    Depends,
    llm,
    search,
    LLMTask,
    SearchTask,
    URLFetchTask,
    FileSaveTask,
    BatchArticleSummarizeTask,
)


class ArticleWriterAgent(PipelineAgent):
    """Article Writer Agent using @task decorators.

    This agent showcases the simplified declarative syntax where:
    - Tasks are defined as methods decorated with @task
    - Dependencies are inferred from parameter names
    - Helper functions like llm() and search() simplify task creation

    Pipeline Flow:
    1. search_articles - Search for relevant articles
    2. fetch_content - Fetch article content from URLs
    3. validate_articles - Ensure minimum article count
    4. summarize - Create summaries of each article
    5. generate_outline - Create article outline
    6. write_article - Write the full article
    7. save_article - Save to file
    """

    @task
    def search_articles(self) -> SearchTask:
        """Search for relevant articles on the topic."""
        search_keywords = self.config.get("search_keywords", "")
        max_results = self.config.get("max_search_results", 10)

        if not search_keywords:
            raise ValueError("search_keywords is required in config")

        return search(search_keywords, max_results=max_results)

    @task
    def fetch_content(
        self,
        search_articles: Dict[str, Any],
    ) -> URLFetchTask:
        """Fetch article content from search result URLs."""
        # Extract URLs from search results
        results: List[Any] = search_articles.get("results", [])
        urls: List[str] = []
        for r in results:
            if isinstance(r, dict) and "url" in r:
                url: Any = r["url"]  # pyright: ignore[reportUnknownVariableType]
                if isinstance(url, str):
                    urls.append(url)

        return URLFetchTask(
            "fetch_content",
            {
                "urls": urls,
                "max_urls": 10,
                "timeout": 30,
            },
        )

    @task
    def validate_articles(
        self,
        fetch_content: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate that we have enough articles to proceed."""
        articles = fetch_content.get("articles", [])

        if len(articles) < 3:
            raise RuntimeError(
                "Insufficient articles downloaded. At least 3 successful articles "
                "with HTTP 200 status are required to proceed with article generation."
            )

        return {"validated": True, "article_count": len(articles)}

    @task
    def summarize(
        self,
        fetch_content: Dict[str, Any],
        validate_articles: Dict[str, Any],  # Ensures validation runs first
    ) -> BatchArticleSummarizeTask:
        """Create summaries of each fetched article."""
        llm_provider = self.config.get("llm_provider", "openai")
        llm_model = self.config.get("llm_model", "gpt-4o-mini")
        summary_length = self.config.get("summary_length", 1000)
        content_limit = self.config.get("content_limit", 3000)
        min_content_length = self.config.get("min_content_length", 50)
        temperature = self.config.get("summarization_temperature", 0.3)
        max_tokens = self.config.get("summarization_max_tokens", 300)

        return BatchArticleSummarizeTask(
            "summarize",
            {
                "articles": fetch_content.get("articles", []),
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "summary_length": summary_length,
                "content_limit": content_limit,
                "min_content_length": min_content_length,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

    @task
    def generate_outline(
        self,
        summarize: Annotated[str, Depends("summarize.formatted_summaries")],
    ) -> LLMTask:
        """Generate article outline from summaries."""
        guideline = self.config.get("guideline", "")
        llm_provider = self.config.get("llm_provider", "openai")
        llm_model = self.config.get("llm_model", "gpt-4o-mini")

        if not guideline:
            raise ValueError("guideline is required in config")

        prompt = self._create_outline_prompt(summarize, guideline)

        return llm(
            prompt=prompt,
            model=llm_model,
            provider=llm_provider,
            system="You are an expert content strategist and writer who creates compelling article outlines.",
            temperature=0.7,
            max_tokens=2000,
        )

    @task
    def write_article(
        self,
        generate_outline: Annotated[str, Depends("generate_outline.content")],
        summarize: Annotated[str, Depends("summarize.formatted_summaries")],
    ) -> LLMTask:
        """Write the full article based on outline and summaries."""
        guideline = self.config.get("guideline", "")
        llm_provider = self.config.get("llm_provider", "openai")
        llm_model = self.config.get("llm_model", "gpt-4o-mini")

        prompt = self._create_article_prompt(generate_outline, guideline, summarize)

        return llm(
            prompt=prompt,
            model=llm_model,
            provider=llm_provider,
            system="You are an expert writer who creates high-quality, engaging articles.",
            temperature=0.8,
            max_tokens=10000,
        )

    @task
    def save_article(
        self,
        write_article: Annotated[str, Depends("write_article.content")],
    ) -> FileSaveTask:
        """Save the article to a file."""
        search_keywords = self.config.get("search_keywords", "topic")
        output_dir = self.config.get("output_dir", "output")

        return FileSaveTask(
            "save_article",
            {
                "content": write_article,
                "title": f"Article on {search_keywords}",
                "output_dir": output_dir,
                "file_format": "md",
            },
        )

    # Helper methods for prompt generation

    def _create_outline_prompt(self, summaries: str, guideline: str) -> str:
        """Create prompt for outline generation."""
        return f"""Based on the article summaries and guidelines given later, create a comprehensive and detailed outline for a new article that will be significantly better than the existing ones.

Please create a detailed, structured outline that includes:

1. Introduction Section (200-300 words)
   - Hook to grab reader attention
   - Background context
   - Thesis statement and article overview
   - Key points to cover

2. Four to Six body sections (300-500 words each)
   - Clear section titles
   - Detailed key points (5-7 per section)
   - Specific subtopics and examples
   - Approximate word count target

3. Conclusion (200-300 words)
   - Summary of key points
   - Future outlook
   - Call to action or final thoughts

The outline should be comprehensive, well-researched, and ensure the new article provides unique insights and value. Make it detailed enough that someone could write a 2000+ word article from this outline.

Guidelines for the new article:
{guideline}

Article summaries to reference:
{summaries}

"""

    def _create_article_prompt(
        self, outline: str, guideline: str, summaries: str
    ) -> str:
        """Create prompt for article generation."""
        return f"""Write a comprehensive, high-quality article based on the outline and guidelines. This should be a substantial, well-researched piece of content.

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

The article should be comprehensive enough to serve as a definitive resource on the topic.

Guidelines for the article:
{guideline}

Article Outline:
{outline}

Reference summaries:
{summaries}

"""


def main() -> None:
    """Run the article writer agent."""
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
