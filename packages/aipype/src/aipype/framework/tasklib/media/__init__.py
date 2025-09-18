"""Media processing tasks for the MI Agents framework.

This subpackage contains tasks for media file processing including:
- Video to audio extraction
- Audio transcription using AI APIs
- Media format conversions
"""

from .extract_audio_from_video_task import ExtractAudioFromVideoTask
from .audio_transcript_task import AudioTranscriptTask

__all__ = [
    "ExtractAudioFromVideoTask",
    "AudioTranscriptTask",
]
