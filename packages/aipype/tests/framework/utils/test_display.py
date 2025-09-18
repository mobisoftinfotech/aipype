"""Tests for display utilities."""

from aipype.framework.utils.display import (
    trim_text,
    wrap_single_line,
    wrap_text_lines,
    format_header,
    format_separator,
    format_message_box,
)


class TestTrimText:
    """Test text trimming functionality."""

    def test_text_fits_within_limit(self) -> None:
        """Test text that fits within the limit."""
        result = trim_text("Hello World", 20)
        assert result == "Hello World"

    def test_text_exact_limit(self) -> None:
        """Test text that exactly matches the limit."""
        result = trim_text("Hello", 5)
        assert result == "Hello"

    def test_text_exceeds_limit(self) -> None:
        """Test text that exceeds the limit gets trimmed with ellipsis."""
        result = trim_text("This is a very long text", 10)
        assert result == "This is..."
        assert len(result) == 10

    def test_empty_string(self) -> None:
        """Test empty string handling."""
        result = trim_text("", 10)
        assert result == ""

    def test_very_short_limit(self) -> None:
        """Test with very short limit."""
        result = trim_text("Hello", 3)
        assert result == "..."
        assert len(result) == 3

    def test_limit_shorter_than_ellipsis(self) -> None:
        """Test limit shorter than ellipsis length."""
        result = trim_text("Hello", 2)
        # When max_width < 3, text[:max_width-3] becomes negative slice
        # text[:-1] = "Hell", so result is "Hell..." (7 chars)
        assert result == "Hell..."
        assert len(result) == 7

    def test_unicode_characters(self) -> None:
        """Test trimming with unicode characters."""
        result = trim_text("Hello 🌍 World", 10)
        assert result == "Hello 🌍..."
        assert len(result) == 10


class TestWrapSingleLine:
    """Test single line wrapping functionality."""

    def test_text_fits_in_width(self) -> None:
        """Test text that fits within width."""
        result = wrap_single_line("Hello World", 20)
        assert result == ["Hello World"]

    def test_text_needs_wrapping_at_word_boundary(self) -> None:
        """Test text that needs wrapping at word boundaries."""
        result = wrap_single_line("This is a very long line that needs wrapping", 20)
        expected = ["This is a very long", "line that needs", "wrapping"]
        assert result == expected

    def test_single_word_too_long(self) -> None:
        """Test single word longer than width gets truncated."""
        result = wrap_single_line("supercalifragilisticexpialidocious", 10)
        # word[:width-3] = word[:7] = "superca", plus "..." = "superca..."
        assert result == ["superca..."]

    def test_multiple_long_words(self) -> None:
        """Test multiple words where some are too long."""
        result = wrap_single_line("short supercalifragilisticexpialidocious word", 10)
        # Same logic: word[:7] = "superca", plus "..." = "superca..."
        expected = ["short", "superca...", "word"]
        assert result == expected

    def test_empty_string(self) -> None:
        """Test empty string handling."""
        result = wrap_single_line("", 10)
        assert result == [""]

    def test_single_space(self) -> None:
        """Test single space handling."""
        result = wrap_single_line(" ", 10)
        assert result == [" "]

    def test_multiple_spaces(self) -> None:
        """Test multiple spaces handling."""
        result = wrap_single_line("word   with   spaces", 10)
        # split(' ') creates ['word', '', '', 'with', '', '', 'spaces']
        # Empty strings collapse in the joining logic, becoming single spaces
        assert result == ["word with", "spaces"]

    def test_formatting_prefixes(self) -> None:
        """Test lines with formatting prefixes like [OK]."""
        result = wrap_single_line(
            "[OK] This is a successful operation that completed", 25
        )
        expected = ["[OK] This is a successful", "operation that completed"]
        assert result == expected


class TestWrapTextLines:
    """Test multiple line wrapping functionality."""

    def test_all_lines_fit(self) -> None:
        """Test when all lines fit within width."""
        lines = ["Short line", "Another short", "Third line"]
        result = wrap_text_lines(lines, 20)
        assert result == lines

    def test_some_lines_need_wrapping(self) -> None:
        """Test when some lines need wrapping."""
        lines = ["Short", "This is a very long line that needs to be wrapped"]
        result = wrap_text_lines(lines, 15)
        expected = [
            "Short",
            "This is a very",
            "long line that",
            "needs to be",
            "wrapped",
        ]
        assert result == expected

    def test_empty_list(self) -> None:
        """Test empty list handling."""
        result = wrap_text_lines([], 10)
        assert result == []

    def test_list_with_empty_strings(self) -> None:
        """Test list containing empty strings."""
        lines = ["", "Some text", ""]
        result = wrap_text_lines(lines, 10)
        assert result == ["", "Some text", ""]

    def test_mixed_content_with_formatting(self) -> None:
        """Test mixed content with different formatting."""
        lines = [
            "The pipeline automatically:",
            "[OK] Resolved task dependencies and executed in optimal order",
            "[OK] Passed data between tasks",
        ]
        result = wrap_text_lines(lines, 30)

        # Test that wrapping occurs and verify structure
        assert len(result) > len(lines)  # Should expand due to wrapping
        assert "The pipeline automatically:" in result
        assert "[OK] Resolved task" in result[1]  # First part of wrapped line
        assert "[OK] Passed data between tasks" in result  # This line fits

        # Verify all content is preserved
        full_content = " ".join(result)
        assert (
            "Resolved task dependencies and executed in optimal order" in full_content
        )


class TestFormatHeader:
    """Test header formatting functionality."""

    def test_normal_header(self) -> None:
        """Test normal header formatting."""
        result = format_header("Test Header")
        expected = "=" * 80 + "\nTest Header\n" + "=" * 80
        assert result == expected

    def test_header_with_custom_width(self) -> None:
        """Test header with custom width."""
        result = format_header("Test", 10)
        expected = "=" * 10 + "\nTest\n" + "=" * 10
        assert result == expected

    def test_header_with_custom_character(self) -> None:
        """Test header with custom character."""
        result = format_header("Test", 10, "-")
        expected = "-" * 10 + "\nTest\n" + "-" * 10
        assert result == expected

    def test_header_title_needs_trimming(self) -> None:
        """Test header where title needs trimming."""
        result = format_header("This is a very long header title", 20)
        expected = "=" * 20 + "\nThis is a very lo...\n" + "=" * 20
        assert result == expected

    def test_empty_title(self) -> None:
        """Test header with empty title."""
        result = format_header("", 10)
        expected = "=" * 10 + "\n\n" + "=" * 10
        assert result == expected

    def test_very_small_width(self) -> None:
        """Test header with very small width."""
        result = format_header("Hello", 3)
        expected = "=" * 3 + "\n...\n" + "=" * 3
        assert result == expected


class TestFormatSeparator:
    """Test separator formatting functionality."""

    def test_default_separator(self) -> None:
        """Test default separator."""
        result = format_separator()
        assert result == "=" * 80

    def test_custom_width_and_character(self) -> None:
        """Test separator with custom width and character."""
        result = format_separator(10, "-")
        assert result == "-" * 10

    def test_newline_before_only(self) -> None:
        """Test separator with newline before only."""
        result = format_separator(5, "=", newline_before=True)
        assert result == "\n====="

    def test_newline_after_only(self) -> None:
        """Test separator with newline after only."""
        result = format_separator(5, "=", newline_after=True)
        assert result == "=====\n"

    def test_newline_before_and_after(self) -> None:
        """Test separator with newlines before and after."""
        result = format_separator(5, "=", newline_before=True, newline_after=True)
        assert result == "\n=====\n"

    def test_no_newlines(self) -> None:
        """Test separator with no newlines."""
        result = format_separator(5, "=", newline_before=False, newline_after=False)
        assert result == "====="

    def test_zero_width(self) -> None:
        """Test separator with zero width."""
        result = format_separator(0)
        assert result == ""


class TestFormatMessageBox:
    """Test message box formatting functionality."""

    def test_simple_message_box(self) -> None:
        """Test simple message box with title and content."""
        content = ["Line 1", "Line 2", "Line 3"]
        result = format_message_box("Test Title", content, width=20)

        expected = (
            "=" * 20
            + "\n"
            + "Test Title\n"
            + "=" * 20
            + "\n"
            + "Line 1\n"
            + "Line 2\n"
            + "Line 3\n"
            + "=" * 20
        )
        assert result == expected

    def test_message_box_with_newline_before(self) -> None:
        """Test message box with leading newline."""
        content = ["Content"]
        result = format_message_box("Title", content, width=10, newline_before=True)

        expected = (
            "\n"
            + "=" * 10
            + "\n"
            + "Title\n"
            + "=" * 10
            + "\n"
            + "Content\n"
            + "=" * 10
        )
        assert result == expected

    def test_message_box_title_trimming(self) -> None:
        """Test message box where title needs trimming."""
        content = ["Content"]
        result = format_message_box(
            "Very Long Title That Exceeds Width", content, width=15
        )

        expected = (
            "=" * 15
            + "\n"
            + "Very Long Ti...\n"
            + "=" * 15
            + "\n"
            + "Content\n"
            + "=" * 15
        )
        assert result == expected

    def test_message_box_content_wrapping(self) -> None:
        """Test message box with content that needs wrapping."""
        content = ["This is a very long line that will need to be wrapped"]
        result = format_message_box("Title", content, width=20)

        expected = (
            "=" * 20
            + "\n"
            + "Title\n"
            + "=" * 20
            + "\n"
            + "This is a very long\n"
            + "line that will need\n"
            + "to be wrapped\n"
            + "=" * 20
        )
        assert result == expected

    def test_message_box_empty_content(self) -> None:
        """Test message box with empty content list."""
        result = format_message_box("Title", [], width=10)

        expected = "=" * 10 + "\n" + "Title\n" + "=" * 10 + "\n" + "=" * 10
        assert result == expected

    def test_message_box_custom_character(self) -> None:
        """Test message box with custom separator character."""
        content = ["Content"]
        result = format_message_box("Title", content, width=10, char="-")

        expected = (
            "-" * 10 + "\n" + "Title\n" + "-" * 10 + "\n" + "Content\n" + "-" * 10
        )
        assert result == expected

    def test_real_world_example(self) -> None:
        """Test real-world example matching the original use case."""
        content = [
            "The pipeline automatically:",
            "[OK] Resolved task dependencies",
            "[OK] Executed tasks in optimal order",
            "[OK] Passed data between tasks",
            "[OK] Applied transformations and conditions",
        ]
        result = format_message_box(
            "[COMPLETE] DEMONSTRATION COMPLETE", content, newline_before=True
        )

        # Verify structure without exact string matching due to potential wrapping
        lines = result.split("\n")

        # Should start with newline + separator
        assert lines[0] == ""
        assert lines[1] == "=" * 80

        # Should have title (possibly trimmed)
        assert "[COMPLETE]" in lines[2]

        # Should have separators and content
        assert lines[3] == "=" * 80
        assert "The pipeline automatically:" in lines
        assert "[OK] Resolved task dependencies" in lines
        assert lines[-1] == "=" * 80

    def test_message_box_mixed_content_lengths(self) -> None:
        """Test message box with mixed short and long content."""
        content = [
            "Short",
            "This is a much longer line that will definitely need wrapping in the message box",
            "Medium length line here",
            "End",
        ]
        result = format_message_box("Mixed Content", content, width=25)

        lines = result.split("\n")

        # Should contain all content, possibly wrapped
        result_text = " ".join(lines)
        assert "Short" in result_text
        assert "This is a much longer line" in result_text
        assert "Medium length line here" in result_text
        assert "End" in result_text

        # All lines should respect width limit (except separators)
        for line in lines:
            if line and not line.startswith("="):
                assert len(line) <= 25
