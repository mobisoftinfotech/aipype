"""Integration tests for example agents from aipype-examples.

These tests run the actual example agents to ensure they work correctly.
"""

import os
import pytest
from aipype import AgentRunResult


class TestExampleAgents:
    """Integration tests for example agents."""

    def test_basic_agent_example(self) -> None:
        """Test basic agent example executes successfully."""
        from aipype_examples.examples.basic_agent_example import ExampleAgent

        agent = ExampleAgent(
            name="test-basic-example", config={"message": "Integration test message"}
        )
        result = agent.run()

        assert isinstance(result, AgentRunResult)
        assert result.is_success()
        assert result.completed_tasks >= 1
        assert result.failed_tasks == 0
        assert result.agent_name == "test-basic-example"

    @pytest.mark.ollama
    def test_llm_agent_example(
        self, skip_if_ollama_unavailable, skip_if_test_model_unavailable
    ) -> None:
        """Test LLM agent example with Ollama."""
        from aipype_examples.examples.llm_agent_example import LLMAgent

        test_model = os.getenv("INTEGRATION_TEST_MODEL", "gemma3:1b")

        agent = LLMAgent(
            name="test-llm-example",
            config={
                "topic": "integration testing best practices",
                "llm_provider": "ollama",
                "llm_model": test_model,
            },
        )
        result = agent.run()

        assert isinstance(result, AgentRunResult)
        assert result.is_success()
        assert result.completed_tasks >= 1
        assert result.agent_name == "test-llm-example"

        # Verify LLM task completed
        llm_result = agent.context.get_result("generate_content")
        if llm_result:
            assert "content" in llm_result

    def test_calculator_tool_calls_example(self) -> None:
        """Test calculator tool calls example."""
        from aipype_examples.examples.calculator_tool_calls_example import (
            CalculatorAgent,
        )

        agent = CalculatorAgent(
            name="test-calculator",
            config={
                "llm_provider": "openai",
                "llm_model": "gpt-4o-mini",  # Model that supports tools
                "prompt": "What is 5 + 3? Then calculate the average of [2, 4, 6, 8, 10]. Use the available tools.",
            },
        )
        result = agent.run()

        assert isinstance(result, AgentRunResult)
        # Tool calling might not always work perfectly, so accept partial success
        assert result.is_success() or result.is_partial()
        assert result.completed_tasks >= 1
        assert result.agent_name == "test-calculator"

    def test_search_tool_example(self) -> None:
        """Test search tool example."""
        from aipype_examples.examples.search_tool_example import SearchExampleAgent

        agent = SearchExampleAgent(
            name="test-search-example",
            config={"query": "Python integration testing", "max_results": 3},
        )
        result = agent.run()

        assert isinstance(result, AgentRunResult)
        # Search might fail or return partial results
        assert result.is_success() or result.is_partial()
        assert result.completed_tasks >= 1
        assert result.agent_name == "test-search-example"

        # Check if search results were obtained
        search_result = agent.context.get_result("search")
        if search_result:
            assert "results" in search_result

    def test_simple_display_demo(self) -> None:
        """Test simple display demo."""
        from aipype_examples.examples.simple_display_demo import SimpleDisplayDemo

        agent = SimpleDisplayDemo(name="test-display-demo", config={})
        result = agent.run()

        assert isinstance(result, AgentRunResult)
        assert result.is_success()
        assert result.completed_tasks >= 1
        assert result.agent_name == "test-display-demo"

    @pytest.mark.slow
    @pytest.mark.openai
    def test_article_writer_agent(self, test_output_dir) -> None:
        """Test ArticleWriterAgent with OpenAI (full pipeline)."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")

        from aipype_examples.examples.article_writer_agent import ArticleWriterAgent

        agent = ArticleWriterAgent(
            name="test-article-writer",
            config={
                "search_keywords": "Python testing frameworks",
                "guideline": "Write a comprehensive guide about Python testing",
                "llm_provider": "openai",
                "llm_model": "gpt-4o-mini",
                "max_search_results": 3,
                "output_dir": str(test_output_dir),
                # Use faster settings for testing
                "summary_length": 500,
                "content_limit": 2000,
                "summarization_max_tokens": 200,
            },
        )
        result = agent.run()

        assert isinstance(result, AgentRunResult)
        # Article writer may have partial success if some URLs fail to fetch
        assert result.is_success() or result.is_partial()
        assert result.completed_tasks >= 5  # Should complete most of the pipeline
        assert result.agent_name == "test-article-writer"

        # Check key pipeline milestones
        search_result = agent.context.get_result("search_articles")
        if search_result:
            assert "results" in search_result

        # If successful, check that article was generated
        if result.is_success():
            article_result = agent.context.get_result("write_article")
            assert article_result is not None
            assert "content" in article_result

            # Check that file was saved
            save_result = agent.context.get_result("save_article")
            if save_result:
                assert "file_path" in save_result

    @pytest.mark.gmail
    def test_gmail_test_agent(self) -> None:
        """Test GmailTestAgent (requires Gmail credentials)."""
        if not os.getenv("GOOGLE_CREDENTIALS_FILE"):
            pytest.skip("Gmail credentials not configured")

        from aipype_examples.examples.gmail_test_agent import GmailTestAgent

        agent = GmailTestAgent(
            "test-gmail-agent",
            config={
                "google_credentials_file": os.getenv("GOOGLE_CREDENTIALS_FILE"),
                "google_token_file": os.getenv(
                    "GMAIL_TOKEN_FILE", "test_gmail_token.json"
                ),
                "test_query": "newer_than:1d",
                "max_emails": 2,
                "max_unread_check": 5,
                "api_timeout": 30,
            },
        )
        result = agent.run()

        assert isinstance(result, AgentRunResult)
        # Gmail test should complete successfully if credentials are valid
        assert result.is_success()
        assert result.completed_tasks == 6  # All 6 tasks in the pipeline
        assert result.agent_name == "test-gmail-agent"

        # Check authentication task
        auth_result = agent.context.get_result("gmail_auth")
        assert auth_result is not None
        assert "credentials" in auth_result

        # Check email listing task
        list_result = agent.context.get_result("test_list_emails")
        assert list_result is not None
        assert "retrieved_count" in list_result

        # Check final report
        report_result = agent.context.get_result("generate_report")
        assert report_result is not None
        assert report_result.get("test_passed") is True
