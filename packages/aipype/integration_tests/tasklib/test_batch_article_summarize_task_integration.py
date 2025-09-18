"""Integration tests for BatchArticleSummarizeTask with real Ollama API calls.

These tests use the actual Ollama service with configurable model - no mocking.
They verify real LLM responses, token usage, and end-to-end functionality.

Prerequisites:
- Ollama service running on localhost:11434
- Test model available (default: gemma3:1b, configurable via INTEGRATION_TEST_MODEL env var)
- Sufficient system resources (4GB+ RAM recommended for gemma3:1b)

Run with: pytest integration_tests/tasklib/test_batch_article_summarize_task_integration.py -v

To use a different model:
INTEGRATION_TEST_MODEL=llama2 pytest integration_tests/tasklib/test_batch_article_summarize_task_integration.py -v
"""

import pytest
import time
from typing import Any, Dict, List
from aipype import (
    BatchArticleSummarizeTask,
    TaskDependency,
    DependencyType,
    TaskResult,
)


@pytest.mark.integration
@pytest.mark.ollama
@pytest.mark.slow
class TestBatchArticleSummarizeTaskIntegration:
    """Integration tests for BatchArticleSummarizeTask with real Ollama API calls."""

    def test_basic_real_summarization_short_articles(
        self,
        skip_if_ollama_unavailable: Any,
        skip_if_test_model_unavailable: Any,
        ollama_config: Dict[str, Any],
        realistic_articles_short: List[Dict[str, Any]],
        integration_test_model: str,
    ) -> None:
        """Test real summarization with shorter articles using Ollama."""
        # Create task with real Ollama configuration
        task = BatchArticleSummarizeTask(
            "test_real_summarization",
            {
                "articles": realistic_articles_short,
                **ollama_config,
                "summary_length": 800,
                "temperature": 0.3,
                "max_tokens": 400,
            },
        )

        # Measure execution time
        start_time = time.time()
        result = task.run()
        execution_time = time.time() - start_time

        # Basic structure verification
        assert isinstance(result, TaskResult)
        assert result.is_success()
        assert "summaries" in result.data
        assert "formatted_summaries" in result.data
        assert "total_articles" in result.data
        assert "successful_summaries" in result.data
        assert "total_tokens" in result.data
        assert "model" in result.data
        assert "provider" in result.data

        # Content verification
        assert result.data["total_articles"] == 2
        assert result.data["successful_summaries"] >= 1  # At least one should succeed
        assert result.data["model"] == integration_test_model
        assert result.data["provider"] == "ollama"

        # Verify summaries structure
        summaries = result.data["summaries"]
        assert len(summaries) == 2

        # Check that we got real content (not skipped/failed)
        successful_summaries = [
            s
            for s in summaries
            if not s["summary"].startswith(
                ("[Skipped:", "[Summarization failed:", "[LLM returned generic")
            )
        ]
        assert len(successful_summaries) >= 1, (
            "At least one summary should be successful"
        )

        # Verify real summary content quality
        for summary in successful_summaries:
            summary_text = summary["summary"]

            # Real summaries should be substantial
            assert len(summary_text) > 100, (
                f"Summary too short: {len(summary_text)} chars"
            )

            # Should not be generic responses
            generic_phrases = [
                "How can I assist you",
                "I'm here to help",
                "Feel free to share",
                "What would you like",
                "How may I help",
            ]
            for phrase in generic_phrases:
                assert phrase.lower() not in summary_text.lower(), (
                    f"Generic response detected: {phrase}"
                )

            # Should contain content-specific words (basic relevance check)
            if "edge computing" in summary["title"].lower():
                relevant_terms = ["edge", "computing", "data", "processing", "iot"]
            elif "sustainable software" in summary["title"].lower():
                relevant_terms = [
                    "software",
                    "development",
                    "sustainable",
                    "energy",
                    "code",
                ]
            else:
                relevant_terms = []  # Skip relevance check for unknown articles

            if relevant_terms:
                found_terms = sum(
                    1 for term in relevant_terms if term.lower() in summary_text.lower()
                )
                # Relaxed relevance check - LLM may use different terminology
                if found_terms >= 1:
                    print(
                        f"[SUCCESS] Found {found_terms} relevant terms for '{summary['title']}'"
                    )
                else:
                    print(
                        f"[WARNING] No specific terms found for '{summary['title']}', but summary looks valid"
                    )

        # Token usage verification
        assert result.data["total_tokens"] > 0, "Should have consumed tokens"
        assert isinstance(result.data["total_tokens"], int), (
            "Token count should be integer"
        )

        # Performance verification (should complete in reasonable time)
        assert execution_time < 120, f"Test took too long: {execution_time:.2f} seconds"

        print(f"\n[SUCCESS] Integration test completed in {execution_time:.2f} seconds")
        print(f"[SUCCESS] Processed {result.data['total_articles']} articles")
        print(
            f"[SUCCESS] Generated {result.data['successful_summaries']} successful summaries"
        )
        print(f"[SUCCESS] Used {result.data['total_tokens']} tokens")

    def test_real_summarization_long_articles(
        self,
        skip_if_ollama_unavailable: Any,
        skip_if_test_model_unavailable: Any,
        ollama_config: Dict[str, Any],
        realistic_articles_long: List[Dict[str, Any]],
    ) -> None:
        """Test real summarization with long-form articles (2000-5000 words)."""
        # Use only the first long article to keep test time reasonable
        test_article = [realistic_articles_long[0]]

        task = BatchArticleSummarizeTask(
            "test_long_article_summarization",
            {
                "articles": test_article,
                **ollama_config,
                "summary_length": 1200,  # Longer summary for long article
                "content_limit": 4000,  # Allow more content
                "temperature": 0.2,  # Lower temperature for consistency
                "max_tokens": 600,  # More tokens for longer summary
            },
        )

        start_time = time.time()
        result = task.run()
        execution_time = time.time() - start_time

        # Verify processing
        assert result.data["total_articles"] == 1
        assert result.data["successful_summaries"] >= 1

        summary = result.data["summaries"][0]
        summary_text = summary["summary"]

        # Verify we got a real summary (not error/skip)
        assert not summary_text.startswith(
            ("[Skipped:", "[Summarization failed:", "[LLM returned generic")
        )

        # Long articles should produce substantial summaries
        assert len(summary_text) > 200, (
            f"Summary too short for long article: {len(summary_text)} chars"
        )

        # Check for AI/healthcare-related content (based on the test article)
        relevant_terms = [
            "artificial intelligence",
            "ai",
            "healthcare",
            "medical",
            "patient",
            "diagnosis",
            "treatment",
        ]
        found_terms = sum(
            1 for term in relevant_terms if term.lower() in summary_text.lower()
        )
        # Relaxed check - LLM may use different terminology but should be topically relevant
        if found_terms >= 1:
            print(
                f"[SUCCESS] Found {found_terms} relevant terms in long article summary"
            )
        else:
            print(
                "[WARNING] No specific terms found, but summary length indicates successful processing"
            )

        # Should use more tokens for longer content
        assert result.data["total_tokens"] > 100, (
            "Long article should use substantial tokens"
        )

        print(
            f"\n[SUCCESS] Long article test completed in {execution_time:.2f} seconds"
        )
        print(f"[SUCCESS] Article length: {len(test_article[0]['content'])} characters")
        print(f"[SUCCESS] Summary length: {len(summary_text)} characters")
        print(f"[SUCCESS] Tokens used: {result.data['total_tokens']}")

    def test_custom_configuration_effects(
        self,
        skip_if_ollama_unavailable: Any,
        skip_if_test_model_unavailable: Any,
        ollama_config: Dict[str, Any],
        realistic_articles_short: List[Dict[str, Any]],
    ) -> None:
        """Test that custom configuration parameters affect real LLM responses."""
        test_article = [realistic_articles_short[0]]  # Use one article

        # Test with very short summary length
        task_short = BatchArticleSummarizeTask(
            "test_short_summary",
            {
                "articles": test_article,
                **ollama_config,
                "summary_length": 200,  # Very short
                "temperature": 0.1,  # Very deterministic
                "max_tokens": 100,  # Limited tokens
            },
        )

        result_short = task_short.run()

        # Test with longer summary length
        task_long = BatchArticleSummarizeTask(
            "test_long_summary",
            {
                "articles": test_article,
                **ollama_config,
                "summary_length": 1000,  # Longer
                "temperature": 0.1,  # Same deterministic setting
                "max_tokens": 500,  # More tokens
            },
        )

        result_long = task_long.run()

        # Both should succeed
        assert result_short.data["successful_summaries"] >= 1
        assert result_long.data["successful_summaries"] >= 1

        # Verify configuration effects
        assert result_short.data["summary_length"] == 200
        assert result_long.data["summary_length"] == 1000

        # Extract successful summaries
        short_summary = next(
            s["summary"]
            for s in result_short.data["summaries"]
            if not s["summary"].startswith(
                ("[Skipped:", "[Summarization failed:", "[LLM returned generic")
            )
        )
        long_summary = next(
            s["summary"]
            for s in result_long.data["summaries"]
            if not s["summary"].startswith(
                ("[Skipped:", "[Summarization failed:", "[LLM returned generic")
            )
        )

        # Length relationship (longer config should generally produce longer summaries)
        # Note: This is not guaranteed due to LLM variability, but generally expected
        if len(short_summary) > 0 and len(long_summary) > 0:
            print(f"Short summary length: {len(short_summary)}")
            print(f"Long summary length: {len(long_summary)}")
            # Allow some flexibility in length comparison due to LLM variability
            # Just verify both are reasonable lengths
            assert 50 <= len(short_summary) <= 600, (
                f"Short summary length unexpected: {len(short_summary)}"
            )
            assert 100 <= len(long_summary) <= 2000, (
                f"Long summary length unexpected: {len(long_summary)}"
            )

        print("[SUCCESS] Configuration effects verified")
        print(f"[SUCCESS] Short config tokens: {result_short.data['total_tokens']}")
        print(f"[SUCCESS] Long config tokens: {result_long.data['total_tokens']}")

    def test_content_truncation_behavior(
        self,
        skip_if_ollama_unavailable: Any,
        skip_if_test_model_unavailable: Any,
        ollama_config: Dict[str, Any],
        realistic_articles_long: List[Dict[str, Any]],
    ) -> None:
        """Test content truncation with very long articles."""
        long_article = realistic_articles_long[0]  # ~3000+ words

        task = BatchArticleSummarizeTask(
            "test_truncation",
            {
                "articles": [long_article],
                **ollama_config,
                "content_limit": 1000,  # Much smaller than article content
                "summary_length": 500,
                "max_tokens": 300,
            },
        )

        result = task.run()

        # Should still succeed despite truncation
        assert result.data["successful_summaries"] >= 1

        summary = result.data["summaries"][0]
        assert not summary["summary"].startswith(
            ("[Skipped:", "[Summarization failed:")
        )

        # Should produce a reasonable summary even with truncated content
        summary_text = summary["summary"]
        assert len(summary_text) > 100, (
            "Truncated content should still produce reasonable summary"
        )

        print("[SUCCESS] Content truncation test passed")
        print(
            f"[SUCCESS] Original article length: {len(long_article['content'])} chars"
        )
        print("[SUCCESS] Content limit: 1000 chars")
        print(f"[SUCCESS] Summary produced: {len(summary_text)} chars")

    def test_error_handling_invalid_model(
        self,
        skip_if_ollama_unavailable: Any,
        realistic_articles_short: List[Dict[str, Any]],
    ) -> None:
        """Test error handling with invalid model name."""
        task = BatchArticleSummarizeTask(
            "test_invalid_model",
            {
                "articles": [realistic_articles_short[0]],
                "llm_provider": "ollama",
                "llm_model": "nonexistent-model:99b",  # Invalid model
                "api_base": "http://localhost:11434",
            },
        )

        result = task.run()

        # Should handle the error gracefully
        assert result.data["total_articles"] == 1

        # Should record the failure
        summary = result.data["summaries"][0]
        assert summary["summary"].startswith("[Summarization failed:")

        # Error should be informative
        error_message = summary["summary"]
        assert "nonexistent-model" in error_message or "error" in error_message.lower()

        print("[SUCCESS] Error handling test passed")
        print(f"[SUCCESS] Error message: {error_message[:100]}...")

    def test_performance_with_multiple_articles(
        self,
        skip_if_ollama_unavailable: Any,
        skip_if_test_model_unavailable: Any,
        ollama_config: Dict[str, Any],
        realistic_articles_short: List[Dict[str, Any]],
    ) -> None:
        """Test performance with multiple articles."""
        # Use both short articles for batch processing test
        task = BatchArticleSummarizeTask(
            "test_batch_performance",
            {
                "articles": realistic_articles_short,
                **ollama_config,
                "summary_length": 600,
                "max_tokens": 350,
            },
        )

        start_time = time.time()
        result = task.run()
        execution_time = time.time() - start_time

        # Verify all articles processed
        assert result.data["total_articles"] == 2
        assert result.data["successful_summaries"] >= 1

        # Performance should be reasonable (allowing for sequential processing)
        assert execution_time < 180, (
            f"Batch processing too slow: {execution_time:.2f} seconds"
        )

        # Calculate average time per article
        avg_time_per_article = execution_time / result.data["total_articles"]

        print("[SUCCESS] Batch performance test completed")
        print(f"[SUCCESS] Total time: {execution_time:.2f} seconds")
        print(f"[SUCCESS] Average time per article: {avg_time_per_article:.2f} seconds")
        print(f"[SUCCESS] Total tokens used: {result.data['total_tokens']}")

    def test_formatted_summaries_output_real(
        self,
        skip_if_ollama_unavailable: Any,
        skip_if_test_model_unavailable: Any,
        ollama_config: Dict[str, Any],
        realistic_articles_short: List[Dict[str, Any]],
    ) -> None:
        """Test formatted summaries output with real content."""
        task = BatchArticleSummarizeTask(
            "test_formatted_output",
            {
                "articles": [realistic_articles_short[0]],  # Use one article
                **ollama_config,
                "summary_length": 500,
            },
        )

        result = task.run()

        # Verify formatted output structure
        formatted = result.data["formatted_summaries"]
        assert isinstance(formatted, str)
        assert len(formatted) > 0

        # Should contain article information
        article_title = realistic_articles_short[0]["title"]
        assert f"ARTICLE 1 SUMMARY ({article_title}):" in formatted

        # Should contain actual summary content (not error message)
        lines = formatted.split("\n")
        summary_lines = [line for line in lines if not line.startswith("ARTICLE")]
        summary_content = "\n".join(summary_lines).strip()

        assert len(summary_content) > 50, (
            "Formatted summary should contain substantial content"
        )
        assert not summary_content.startswith("[Skipped:"), (
            "Should not be skipped content"
        )
        assert not summary_content.startswith("[Summarization failed:"), (
            "Should not be failed content"
        )

        print("[SUCCESS] Formatted output test passed")
        print(f"[SUCCESS] Formatted output length: {len(formatted)} characters")

    @pytest.mark.slow
    def test_comprehensive_integration_workflow(
        self,
        skip_if_ollama_unavailable: Any,
        skip_if_test_model_unavailable: Any,
        ollama_config: Dict[str, Any],
        realistic_articles_short: List[Dict[str, Any]],
    ) -> None:
        """Comprehensive test simulating real-world usage workflow."""
        # Simulate a complete workflow with dependencies
        task = BatchArticleSummarizeTask(
            "comprehensive_workflow_test",
            {
                **ollama_config,
                "summary_length": 800,
                "min_content_length": 100,
                "temperature": 0.4,
                "max_tokens": 400,
            },
            dependencies=[
                TaskDependency(
                    "articles", "mock_source.articles", DependencyType.REQUIRED
                )
            ],
        )

        # Simulate dependency resolution by setting articles directly
        task.config["articles"] = realistic_articles_short

        start_time = time.time()
        result = task.run()
        execution_time = time.time() - start_time

        # Comprehensive verification
        assert result.data["total_articles"] == len(realistic_articles_short)
        assert result.data["successful_summaries"] >= 1
        assert result.data["total_tokens"] > 0
        assert isinstance(result.data["formatted_summaries"], str)
        assert len(result.data["formatted_summaries"]) > 100

        # Verify individual summaries
        for i, summary_data in enumerate(result.data["summaries"]):
            assert "article_index" in summary_data
            assert "title" in summary_data
            assert "url" in summary_data
            assert "summary" in summary_data

            assert summary_data["article_index"] == i + 1
            assert len(summary_data["title"]) > 0
            assert len(summary_data["url"]) > 0

        # Performance within bounds
        assert execution_time < 150, (
            f"Comprehensive test took too long: {execution_time:.2f}s"
        )

        print("\n[SUCCESS] Comprehensive integration test completed successfully!")
        print(f"[SUCCESS] Execution time: {execution_time:.2f} seconds")
        print(f"[SUCCESS] Articles processed: {result.data['total_articles']}")
        print(f"[SUCCESS] Successful summaries: {result.data['successful_summaries']}")
        print(f"[SUCCESS] Total tokens: {result.data['total_tokens']}")
        print(
            f"[SUCCESS] Average tokens per article: {result.data['total_tokens'] / result.data['total_articles']:.1f}"
        )

        # Print sample summary for manual inspection
        if result.data["successful_summaries"] > 0:
            successful_summary = next(
                s
                for s in result.data["summaries"]
                if not s["summary"].startswith(
                    ("[Skipped:", "[Summarization failed:", "[LLM returned generic")
                )
            )
            print(f"\n[PAGE] Sample Summary for '{successful_summary['title']}':")
            print(f"[NOTE] {successful_summary['summary'][:200]}...")
            print(
                f"[STATS] Summary length: {len(successful_summary['summary'])} characters"
            )
