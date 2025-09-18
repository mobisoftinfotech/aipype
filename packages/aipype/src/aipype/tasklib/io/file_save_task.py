"""File Save Task - Saves content to files with timestamp suffixes."""

import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from typing import override

from ...base_task import BaseTask
from ...task_dependencies import TaskDependency
from ...task_result import TaskResult


class FileSaveTask(BaseTask):
    """Task that saves content to a file with timestamp suffix.

    This is a generic file saving task that can handle various content types
    and formats. Originally designed for article saving but generalized for
    broader reuse.
    """

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        dependencies: Optional[List[TaskDependency]] = None,
    ):
        """Initialize file save task.

        Args:
            name: Task name
            config: Configuration dictionary containing:
                - content: The content to save (optional, can be resolved from dependencies)
                - title: Title for the file (optional, default: "Generated Content")
                - output_dir: Output directory path (default: "output")
                - file_format: Output format - "txt", "md", "json", etc. (default: "txt")
                - filename_prefix: Custom prefix for filename (optional)
            dependencies: List of task dependencies
        """
        super().__init__(name, config)
        self.dependencies = dependencies or []
        self.validation_rules = {
            "required": ["content"],
            "defaults": {
                "title": "Generated Content",
                "output_dir": "output",
                "file_format": "txt",
                "filename_prefix": "",
            },
            "types": {
                "content": str,
                "title": str,
                "output_dir": str,
                "file_format": str,
                "filename_prefix": str,
            },
            # Validation lambdas intentionally use dynamic typing for flexibility across field types
            "custom": {
                "content": lambda x: x.strip() != "",  # pyright: ignore[reportUnknownLambdaType,reportUnknownMemberType]
                "file_format": lambda x: x.lower() in ["txt", "md", "json"],  # pyright: ignore[reportUnknownLambdaType,reportUnknownMemberType]
            },
        }
        self.content = self.config.get("content", "")
        self.title = self.config.get("title", "Generated Content")
        self.output_dir = self.config.get("output_dir", "output")
        self.file_format = self.config.get("file_format", "txt")
        self.filename_prefix = self.config.get("filename_prefix", "")

    @override
    def get_dependencies(self) -> List[TaskDependency]:
        """Get the list of task dependencies.

        Returns:
            List of TaskDependency objects
        """
        return self.dependencies

    @override
    def run(self) -> TaskResult:
        """Save the content to a file with timestamp suffix.

        Returns:
            Dictionary containing:
                - file_path: Path to the saved file
                - file_size: Size of the saved file in bytes
                - timestamp: Timestamp used in filename
                - output_dir: Output directory used
                - filename: Name of the created file
        """
        start_time = datetime.now()

        # Validate configuration using instance validation rules
        validation_failure = self._validate_or_fail(start_time)
        if validation_failure:
            return validation_failure

        # Get content from config (it may have been updated after initialization)
        content = self.config.get("content", "")

        self.logger.info(f"Starting file save task to {self.output_dir}")

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Get title from config (it may have been updated after initialization)
        title = self.config.get("title", self.title)

        # Create filename
        safe_title = self._sanitize_filename(title)

        # Build filename with optional prefix
        filename_parts: List[str] = []
        if self.filename_prefix:
            filename_parts.append(self.filename_prefix)
        filename_parts.extend([safe_title, timestamp])

        filename = f"{'_'.join(filename_parts)}.{self.file_format}"
        file_path = os.path.join(self.output_dir, filename)

        try:
            # Prepare content based on format
            formatted_content = self._format_content(content, title, self.file_format)

            # Write content to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(formatted_content)

            # Get file size
            file_size = os.path.getsize(file_path)

            result_data = {
                "file_path": file_path,
                "file_size": file_size,
                "timestamp": timestamp,
                "output_dir": self.output_dir,
                "filename": filename,
            }

            execution_time = (datetime.now() - start_time).total_seconds()

            self.logger.info(
                f"Content saved successfully: {file_path} ({file_size} bytes)"
            )

            return TaskResult.success(
                data=result_data,
                execution_time=execution_time,
                metadata={
                    "task_type": "file_save",
                    "file_format": self.file_format,
                    "file_size_bytes": file_size,
                    "output_directory": self.output_dir,
                },
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"FileSaveTask save operation failed: Failed to save content to '{file_path}': {str(e)}"
            self.logger.error(error_msg)
            return TaskResult.failure(
                error_message=error_msg,
                execution_time=execution_time,
                metadata={
                    "task_type": "file_save",
                    "attempted_file_path": file_path,
                    "file_format": self.file_format,
                    "error_type": type(e).__name__,
                },
            )

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename by removing invalid characters."""
        # Remove or replace invalid filename characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, "_")

        # Remove extra spaces and limit length
        filename = " ".join(filename.split())
        if len(filename) > 100:
            filename = filename[:97] + "..."

        return filename

    def _format_content(self, content: str, title: str, file_format: str) -> str:
        """Format the content based on the specified file format."""
        if file_format.lower() == "md":
            return self._format_markdown_content(content, title)
        elif file_format.lower() == "json":
            import json

            return json.dumps(
                {
                    "title": title,
                    "content": content,
                    "timestamp": datetime.now().isoformat(),
                    "generated_by": "MI Agents Framework",
                },
                indent=2,
            )
        else:
            # Default to plain text with minimal formatting
            lines: List[str] = []
            lines.append(f"Title: {title}")
            lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append("=" * 50)
            lines.append("")
            lines.append(content)
            return "\n".join(lines)

    def _format_markdown_content(self, content: str, title: str) -> str:
        """Format the content as markdown."""
        lines: List[str] = []

        # Add title
        lines.append(f"# {title}")
        lines.append("")

        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"*Generated on: {timestamp}*")
        lines.append("")

        # Add content
        lines.append(content)
        lines.append("")

        return "\n".join(lines)
