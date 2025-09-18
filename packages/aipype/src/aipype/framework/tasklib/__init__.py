"""TaskLib - Reusable Task Library for aipype Framework.

This package contains a collection of reusable task implementations
organized by category for common use cases in AI agent pipelines.

Subpackages:
- web: Web-related tasks (URL fetching, web scraping, etc.)
- io: Input/Output tasks (file operations, data persistence, etc.)
- media: Media processing tasks (video, audio, transcription, etc.)
"""

# Import tasks from subpackages for convenient access
from .web import URLFetchTask, BatchArticleSummarizeTask
from .io import FileSaveTask
from .media import ExtractAudioFromVideoTask, AudioTranscriptTask

__all__ = [
    # Web tasks
    "URLFetchTask",
    # IO tasks
    "FileSaveTask",
    # Media tasks
    "ExtractAudioFromVideoTask",
    "AudioTranscriptTask",
    # Content processing tasks
    "BatchArticleSummarizeTask",
]
