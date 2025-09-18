"""Tests for URL fetcher and text extraction functionality."""

import pytest

# from typing import Any  # Unused import
from unittest.mock import patch, Mock
import requests
from aipype.utils.url_fetcher import (
    URLFetcher,
    fetch_url,
    fetch_url_headers,
    fetch_main_text,
)


class TestURLFetcher:
    """Test URLFetcher class functionality."""

    def test_init_default_config(self) -> None:
        """Test URLFetcher initialization with default configuration."""
        fetcher = URLFetcher()

        assert fetcher.timeout == 30
        assert fetcher.follow_redirects is True
        assert fetcher.max_redirects == 10
        assert "Mozilla/5.0" in fetcher.headers["User-Agent"]
        assert "Chrome" in fetcher.headers["User-Agent"]

    def test_init_custom_config(self) -> None:
        """Test URLFetcher initialization with custom configuration."""
        config = {
            "timeout": 60,
            "user_agent": "Custom Agent",
            "headers": {"Authorization": "Bearer token"},
            "follow_redirects": False,
            "max_redirects": 5,
        }
        fetcher = URLFetcher(config)

        assert fetcher.timeout == 60
        assert fetcher.follow_redirects is False
        assert fetcher.max_redirects == 5
        assert fetcher.headers["User-Agent"] == "Custom Agent"
        assert fetcher.headers["Authorization"] == "Bearer token"

    def test_chrome_like_headers(self) -> None:
        """Test that Chrome-like headers are properly set."""
        fetcher = URLFetcher()

        # Check key Chrome headers
        assert "Sec-Ch-Ua" in fetcher.headers
        assert "Sec-Fetch-Dest" in fetcher.headers
        assert "Accept-Language" in fetcher.headers
        assert fetcher.headers["Accept-Language"] == "en-US,en;q=0.9"

    def test_context_manager(self) -> None:
        """Test URLFetcher as context manager."""
        with URLFetcher() as fetcher:
            assert hasattr(fetcher, "session")
            assert fetcher.session is not None
        # Session should be closed after context


class TestFetchURL:
    """Test fetch_url convenience function."""

    @patch("aipype.utils.url_fetcher.requests.Session.get")
    def test_fetch_url_successful(self, mock_get: Mock) -> None:
        """Test successful URL fetching."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com"
        mock_response.headers = {"content-type": "text/html; charset=utf-8"}
        mock_response.text = "<html><body>Test content</body></html>"
        mock_response.content = b"<html><body>Test content</body></html>"
        mock_response.encoding = "utf-8"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = fetch_url("https://example.com")

        assert result["status_code"] == 200
        assert result["url"] == "https://example.com"
        assert result["content_type"] == "text/html"
        assert "Test content" in result["content"]
        assert result["size"] > 0
        assert "response_time" in result

    def test_fetch_url_invalid_url(self) -> None:
        """Test fetch_url with invalid URL."""
        with pytest.raises(ValueError, match="Invalid URL format"):
            fetch_url("not-a-url")

        with pytest.raises(ValueError, match="URL cannot be empty"):
            fetch_url("")

    @patch("aipype.utils.url_fetcher.requests.Session.get")
    def test_fetch_url_timeout(self, mock_get: Mock) -> None:
        """Test fetch_url timeout handling."""
        mock_get.side_effect = requests.exceptions.Timeout()

        with pytest.raises(RuntimeError, match="Request timed out"):
            fetch_url("https://example.com")

    @patch("aipype.utils.url_fetcher.requests.Session.get")
    def test_fetch_url_connection_error(self, mock_get: Mock) -> None:
        """Test fetch_url connection error handling."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")

        with pytest.raises(RuntimeError, match="Failed to connect"):
            fetch_url("https://example.com")


class TestFetchMainText:
    """Test fetch_main_text function with various content types."""

    @patch("aipype.utils.url_fetcher.fetch_url")
    def test_fetch_main_text_html_content(self, mock_fetch_url: Mock) -> None:
        """Test HTML text extraction."""
        html_content = """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <nav>Navigation menu</nav>
            <article>
                <h1>Main Article Title</h1>
                <p>This is the main content of the article.</p>
                <p>This paragraph contains important information.</p>
            </article>
            <footer>Footer content</footer>
        </body>
        </html>
        """

        mock_fetch_url.return_value = {
            "url": "https://example.com",
            "status_code": 200,
            "content": html_content,
            "content_bytes": html_content.encode("utf-8"),
            "content_type": "text/html",
            "encoding": "utf-8",
            "size": len(html_content),
            "response_time": 0.5,
        }

        result = fetch_main_text("https://example.com")

        assert result["content_type"] == "text/html"
        assert result["extraction_method"].startswith("html_")
        assert "Main Article Title" in result["text"]
        assert "main content" in result["text"]
        assert "Navigation menu" not in result["text"]  # Should be filtered out
        assert result["text_size"] > 0
        assert (
            result["text_size"] < result["original_size"]
        )  # Extracted text should be smaller

    @patch("aipype.utils.url_fetcher.fetch_url")
    def test_fetch_main_text_basic_html_method(self, mock_fetch_url: Mock) -> None:
        """Test HTML text extraction with basic method."""
        html_content = """
        <html>
        <body>
            <main>
                <h1>Main Content</h1>
                <p>This is in the main section.</p>
            </main>
            <aside>Sidebar content</aside>
        </body>
        </html>
        """

        mock_fetch_url.return_value = {
            "url": "https://example.com",
            "status_code": 200,
            "content": html_content,
            "content_bytes": html_content.encode("utf-8"),
            "content_type": "text/html",
            "encoding": "utf-8",
            "size": len(html_content),
            "response_time": 0.3,
        }

        config = {"html_method": "basic"}
        result = fetch_main_text("https://example.com", config)

        assert result["extraction_method"] == "html_basic"
        assert "Main Content" in result["text"]
        assert "main section" in result["text"]

    @patch("aipype.utils.url_fetcher.fetch_url")
    def test_fetch_main_text_pdf_content(self, mock_fetch_url: Mock) -> None:
        """Test PDF text extraction with mocked PDF content."""
        # Create a simple mock PDF bytes
        pdf_content = b"%PDF-1.4 mock pdf content"

        mock_fetch_url.return_value = {
            "url": "https://example.com/document.pdf",
            "status_code": 200,
            "content": "PDF content (decoded)",
            "content_bytes": pdf_content,
            "content_type": "application/pdf",
            "encoding": "utf-8",
            "size": len(pdf_content),
            "response_time": 1.2,
        }

        # Mock pypdf functionality
        with patch("aipype.utils.url_fetcher.PdfReader") as mock_pdf_reader:
            mock_page = Mock()
            mock_page.extract_text.return_value = (
                "This is extracted text from the PDF page."
            )

            mock_reader_instance = Mock()
            mock_reader_instance.pages = [mock_page]
            mock_pdf_reader.return_value = mock_reader_instance

            result = fetch_main_text("https://example.com/document.pdf")

            assert result["content_type"] == "application/pdf"
            assert result["extraction_method"] == "pdf"
            assert "extracted text from the PDF" in result["text"]
            assert "--- Page 1 ---" in result["text"]

    @patch("aipype.utils.url_fetcher.fetch_url")
    def test_fetch_main_text_docx_content(self, mock_fetch_url: Mock) -> None:
        """Test DOCX text extraction with mocked DOCX content."""
        docx_content = b"PK mock docx content"

        mock_fetch_url.return_value = {
            "url": "https://example.com/document.docx",
            "status_code": 200,
            "content": "DOCX content (decoded)",
            "content_bytes": docx_content,
            "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "encoding": "utf-8",
            "size": len(docx_content),
            "response_time": 0.8,
        }

        # Mock python-docx functionality - need to patch where it's imported in url_fetcher
        with patch("aipype.utils.url_fetcher.DocxDocument") as mock_docx_doc:
            # Mock paragraphs
            mock_paragraph1 = Mock()
            mock_paragraph1.text = "First paragraph of the document."
            mock_paragraph2 = Mock()
            mock_paragraph2.text = "Second paragraph with more content."

            # Mock table
            mock_cell1 = Mock()
            mock_cell1.text = "Header 1"
            mock_cell2 = Mock()
            mock_cell2.text = "Header 2"
            mock_row = Mock()
            mock_row.cells = [mock_cell1, mock_cell2]
            mock_table = Mock()
            mock_table.rows = [mock_row]

            mock_doc_instance = Mock()
            mock_doc_instance.paragraphs = [mock_paragraph1, mock_paragraph2]
            mock_doc_instance.tables = [mock_table]
            mock_docx_doc.return_value = mock_doc_instance

            result = fetch_main_text("https://example.com/document.docx")

            assert (
                result["content_type"]
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            assert result["extraction_method"] == "docx"
            assert "First paragraph" in result["text"]
            assert "Second paragraph" in result["text"]
            assert "Header 1 | Header 2" in result["text"]

    @patch("aipype.utils.url_fetcher.fetch_url")
    def test_fetch_main_text_plain_text(self, mock_fetch_url: Mock) -> None:
        """Test plain text content handling."""
        text_content = (
            "This is plain text content.\nWith multiple lines.\nAnd some information."
        )

        mock_fetch_url.return_value = {
            "url": "https://example.com/file.txt",
            "status_code": 200,
            "content": text_content,
            "content_bytes": text_content.encode("utf-8"),
            "content_type": "text/plain",
            "encoding": "utf-8",
            "size": len(text_content),
            "response_time": 0.2,
        }

        result = fetch_main_text("https://example.com/file.txt")

        assert result["content_type"] == "text/plain"
        assert result["extraction_method"] == "plain_text"
        assert result["text"] == text_content
        assert result["text_size"] == len(text_content)

    @patch("aipype.utils.url_fetcher.fetch_url")
    def test_fetch_main_text_unsupported_content_type(
        self, mock_fetch_url: Mock
    ) -> None:
        """Test handling of unsupported content types."""
        mock_fetch_url.return_value = {
            "url": "https://example.com/image.jpg",
            "status_code": 200,
            "content": "binary image data",
            "content_bytes": b"binary image data",
            "content_type": "image/jpeg",
            "encoding": "utf-8",
            "size": 17,
            "response_time": 0.3,
        }

        with pytest.raises(ValueError, match="Unsupported content type: image/jpeg"):
            fetch_main_text("https://example.com/image.jpg")

    @patch("aipype.utils.url_fetcher.fetch_url")
    def test_fetch_main_text_without_metadata(self, mock_fetch_url: Mock) -> None:
        """Test fetch_main_text with metadata disabled."""
        html_content = "<html><body><p>Simple content</p></body></html>"

        mock_fetch_url.return_value = {
            "url": "https://example.com",
            "status_code": 200,
            "content": html_content,
            "content_bytes": html_content.encode("utf-8"),
            "content_type": "text/html",
            "encoding": "utf-8",
            "size": len(html_content),
            "response_time": 0.1,
        }

        config = {"include_metadata": False}
        result = fetch_main_text("https://example.com", config)

        assert "metadata" not in result
        assert result["text"] is not None
        assert result["extraction_method"] is not None

    def test_dependencies_available(self) -> None:
        """Test that all required dependencies are available."""
        from aipype.utils.url_fetcher import (
            extract_html_text,
            extract_pdf_text,
            extract_docx_text,
        )

        # Since these are now required dependencies, we just verify they can be imported
        # and basic functionality works
        assert extract_html_text is not None
        assert extract_pdf_text is not None
        assert extract_docx_text is not None


class TestFetchURLHeaders:
    """Test fetch_url_headers function."""

    @patch("aipype.utils.url_fetcher.requests.Session.head")
    def test_fetch_url_headers_successful(self, mock_head: Mock) -> None:
        """Test successful header fetching."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com"
        mock_response.headers = {
            "content-type": "text/html; charset=utf-8",
            "content-length": "1234",
        }
        mock_response.raise_for_status = Mock()
        mock_head.return_value = mock_response

        result = fetch_url_headers("https://example.com")

        assert result["status_code"] == 200
        assert result["content_type"] == "text/html"
        assert result["content_length"] == 1234
        assert "response_time" in result

    @patch("aipype.utils.url_fetcher.requests.Session.head")
    def test_fetch_url_headers_no_content_length(self, mock_head: Mock) -> None:
        """Test header fetching without content-length."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status = Mock()
        mock_head.return_value = mock_response

        result = fetch_url_headers("https://example.com")

        assert result["content_length"] is None
