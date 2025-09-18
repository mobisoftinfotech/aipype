"""Display utilities for formatted console output."""


def trim_text(text: str, max_width: int) -> str:
    """Trim text to fit within max_width, adding ellipsis if needed.

    Args:
        text: The text to trim
        max_width: Maximum allowed width

    Returns:
        Trimmed text with '...' if it was too long, original text if it fits
    """
    if len(text) <= max_width:
        return text
    return text[: max_width - 3] + "..."


def wrap_single_line(text: str, width: int) -> list[str]:
    """Wrap a single line to fit within width, breaking at word boundaries when possible.

    Args:
        text: The text line to wrap
        width: Maximum width per line

    Returns:
        List of wrapped lines
    """
    if len(text) <= width:
        return [text]

    # Try to break at word boundaries
    words = text.split(" ")
    lines: list[str] = []
    current_line = ""

    for word in words:
        # Check if adding this word would exceed width
        test_line = f"{current_line} {word}".strip()

        if len(test_line) <= width:
            current_line = test_line
        else:
            # Current line is full, start new line
            if current_line:
                lines.append(current_line)

            # Check if single word is too long
            if len(word) > width:
                # Break long word with ellipsis
                lines.append(word[: width - 3] + "...")
                current_line = ""
            else:
                current_line = word

    # Add any remaining text
    if current_line:
        lines.append(current_line)

    return lines


def wrap_text_lines(content_lines: list[str], width: int) -> list[str]:
    """Wrap multiple content lines, expanding the list as needed.

    Args:
        content_lines: List of content lines to wrap
        width: Maximum width per line

    Returns:
        Expanded list with all lines wrapped to fit width
    """
    wrapped_lines: list[str] = []
    for line in content_lines:
        wrapped_lines.extend(wrap_single_line(line, width))
    return wrapped_lines


def format_header(title: str, width: int = 80, char: str = "=") -> str:
    """Format a header with title centered between separator characters.

    Args:
        title: The header title to display
        width: Total width of the header line (default: 80)
        char: Character to use for the separator (default: "=")

    Returns:
        Formatted header string with newlines
    """
    trimmed_title = trim_text(title, width)
    return f"{char * width}\n{trimmed_title}\n{char * width}"


def print_header(title: str, width: int = 80, char: str = "=") -> None:
    """Print a formatted header with title centered between separator characters.

    Args:
        title: The header title to display
        width: Total width of the header line (default: 80)
        char: Character to use for the separator (default: "=")
    """
    print(format_header(title, width, char))


def format_separator(
    width: int = 80,
    char: str = "=",
    newline_before: bool = False,
    newline_after: bool = False,
) -> str:
    """Format a separator line with optional newlines before and/or after.

    Args:
        width: Width of the separator line (default: 80)
        char: Character to use for the separator (default: "=")
        newline_before: Add newline before the separator (default: False)
        newline_after: Add newline after the separator (default: False)

    Returns:
        Formatted separator string with optional newlines
    """
    separator = char * width
    if newline_before:
        separator = "\n" + separator
    if newline_after:
        separator = separator + "\n"
    return separator


def print_separator(
    width: int = 80,
    char: str = "=",
    newline_before: bool = False,
    newline_after: bool = False,
) -> None:
    """Print a separator line with optional newlines before and/or after.

    Args:
        width: Width of the separator line (default: 80)
        char: Character to use for the separator (default: "=")
        newline_before: Add newline before the separator (default: False)
        newline_after: Add newline after the separator (default: False)
    """
    print(format_separator(width, char, newline_before, newline_after))


def format_message_box(
    title: str,
    content_lines: list[str],
    width: int = 80,
    char: str = "=",
    newline_before: bool = False,
) -> str:
    """Format a message box with title and content lines, with intelligent text processing.

    Args:
        title: The message box title (will be trimmed if too long)
        content_lines: List of content lines (will be wrapped if too long)
        width: Total width for the message box (default: 80)
        char: Character to use for separators (default: "=")
        newline_before: Add newline before the message box (default: False)

    Returns:
        Formatted message box string with title, separators, and wrapped content
    """
    # Trim title if too long
    trimmed_title = trim_text(title, width)

    # Wrap all content lines
    wrapped_content = wrap_text_lines(content_lines, width)

    # Build section components
    separator = char * width

    # Start building the section
    parts: list[str] = []

    # Optional leading newline + opening separator
    if newline_before:
        parts.append(f"\n{separator}")
    else:
        parts.append(separator)

    # Title line
    parts.append(trimmed_title)

    # Middle separator
    parts.append(separator)

    # Content lines
    parts.extend(wrapped_content)

    # Closing separator
    parts.append(separator)

    return "\n".join(parts)


def print_message_box(
    title: str,
    content_lines: list[str],
    width: int = 80,
    char: str = "=",
    newline_before: bool = False,
) -> None:
    """Print a message box with title and content lines, with intelligent text processing.

    Args:
        title: The message box title (will be trimmed if too long)
        content_lines: List of content lines (will be wrapped if too long)
        width: Total width for the message box (default: 80)
        char: Character to use for separators (default: "=")
        newline_before: Add newline before the message box (default: False)
    """
    print(format_message_box(title, content_lines, width, char, newline_before))
