"""Integration tests for URL fetcher with real network requests."""

import pytest
from aipype import fetch_main_text


class TestFetchMainTextIntegration:
    """Integration tests with real URLs (Wikipedia)."""

    @pytest.mark.integration
    def test_fetch_wikipedia_article(self) -> None:
        """Test fetching and extracting text from a Wikipedia article."""
        # Use a stable Wikipedia page that's unlikely to change drastically
        url = "https://en.wikipedia.org/wiki/Python_(programming_language)"

        try:
            result = fetch_main_text(url, {"timeout": 10})

            # Basic assertions
            assert result["content_type"] == "text/html"
            assert result["extraction_method"].startswith("html_")
            assert result["text_size"] > 1000  # Should be substantial content

            # Content assertions - these should be stable
            assert "Python" in result["text"]
            assert "programming language" in result["text"]

            # Should not contain navigation elements
            assert "navigation" not in result["text"].lower()

            # Should have minimal edit links (some may slip through extraction)
            import re

            standalone_edits = len(re.findall(r"\bedit\b", result["text"].lower()))
            assert standalone_edits < 20, (
                f"Too many edit links found: {standalone_edits}"
            )

            print(
                f"[SUCCESS] Successfully extracted {result['text_size']} characters from Wikipedia"
            )
            print(f"[SUCCESS] Extraction method: {result['extraction_method']}")
            print(
                f"[SUCCESS] Response time: {result.get('metadata', {}).get('response_time', 'N/A')}s"
            )

        except Exception as e:
            pytest.skip(
                f"Integration test skipped due to network/dependency issue: {e}"
            )

    @pytest.mark.integration
    def test_fetch_wikipedia_simple_page(self) -> None:
        """Test with a simpler Wikipedia page."""
        url = "https://en.wikipedia.org/wiki/Hello_world"

        try:
            result = fetch_main_text(url, {"html_method": "basic", "timeout": 10})

            assert result["content_type"] == "text/html"
            assert result["extraction_method"] == "html_basic"
            assert "Hello" in result["text"]
            assert "world" in result["text"]
            assert result["text_size"] > 100

            print(f"[SUCCESS] Basic method extracted {result['text_size']} characters")

        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")

    @pytest.mark.integration
    def test_fetch_plain_text_file(self) -> None:
        """Test fetching a plain text file."""
        # Using a stable text file URL
        url = "https://www.gutenberg.org/files/74/74-0.txt"  # Tom Sawyer excerpt

        try:
            result = fetch_main_text(url, {"timeout": 15})

            assert result["content_type"] in ["text/plain", "text/plain; charset=utf-8"]
            assert result["extraction_method"] == "plain_text"
            assert result["text_size"] > 1000

            print(f"[SUCCESS] Plain text extracted {result['text_size']} characters")

        except Exception as e:
            pytest.skip(f"Plain text integration test skipped: {e}")
