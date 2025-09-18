"""Transcribe audio file using OpenAI Whisper API."""

import os
from typing import List, Dict, Any, override, Optional

from ...base_task import BaseTask
from ...task_dependencies import TaskDependency
from ...task_result import TaskResult
from datetime import datetime


class AudioTranscriptTask(BaseTask):
    """Transcribe audio file using OpenAI Whisper API."""

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        dependencies: Optional[List[TaskDependency]] = None,
    ):
        """Initialize audio transcription task.

        Args:
            name: Task name
            config: Task configuration
            dependencies: Task dependencies
        """
        super().__init__(name, config)
        self.dependencies = dependencies or []
        self.validation_rules = {
            "defaults": {
                "model": "whisper-1",
                "language": "en",
                "response_format": "text",
                "temperature": 0.0,
            },
            "types": {
                "model": str,
                "language": str,
                "response_format": str,
                "temperature": (int, float),
            },
            "ranges": {
                "temperature": (0, 1),
            },
            "custom": {
                # Validation lambdas intentionally use dynamic typing for flexible runtime validation
                # They accept unknown input types and safely convert them using str()
                "model": lambda x: str(x).strip() != "",  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
                "response_format": lambda x: str(x)  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
                in ["text", "json", "srt", "verbose_json", "vtt"],
            },
        }

    @override
    def get_dependencies(self) -> List[TaskDependency]:
        """Get task dependencies."""
        return self.dependencies

    @override
    def run(self) -> TaskResult:
        """Transcribe audio file using OpenAI Whisper."""
        start_time = datetime.now()

        # Validate configuration
        validation_error = self._validate_or_fail(start_time)
        if validation_error:
            return validation_error

        try:
            import openai
        except ImportError:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = "AudioTranscriptTask operation failed: openai library not installed. Please install with: pip install openai"
            self.logger.error(error_msg)
            return TaskResult.failure(
                error_message=error_msg,
                execution_time=execution_time,
                metadata={
                    "task_name": self.name,
                    "error_type": "ImportError",
                },
            )

        # Get audio file path from dependencies
        audio_file_path = self.config.get("audio_file_path")
        if not audio_file_path:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = "AudioTranscriptTask operation failed: audio_file_path not provided in config"
            self.logger.error(error_msg)
            return TaskResult.failure(
                error_message=error_msg,
                execution_time=execution_time,
                metadata={
                    "task_name": self.name,
                    "error_type": "ConfigurationError",
                },
            )

        try:
            # Validate audio file exists
            if not os.path.isfile(audio_file_path):
                execution_time = (datetime.now() - start_time).total_seconds()
                error_msg = f"AudioTranscriptTask operation failed: Audio file does not exist: {audio_file_path}"
                self.logger.error(error_msg)
                return TaskResult.failure(
                    error_message=error_msg,
                    execution_time=execution_time,
                    metadata={
                        "task_name": self.name,
                        "error_type": "FileNotFoundError",
                        "audio_path": audio_file_path,
                    },
                )

            # Check file size (OpenAI has 25MB limit for Whisper)
            file_size = os.path.getsize(audio_file_path)
            max_size = 25 * 1024 * 1024  # 25MB in bytes

            if file_size > max_size:
                execution_time = (datetime.now() - start_time).total_seconds()
                error_msg = f"AudioTranscriptTask operation failed: Audio file too large ({file_size} bytes). OpenAI Whisper limit is 25MB"
                self.logger.error(error_msg)
                return TaskResult.failure(
                    error_message=error_msg,
                    execution_time=execution_time,
                    metadata={
                        "task_name": self.name,
                        "error_type": "FileSizeError",
                        "file_size": file_size,
                        "max_size": max_size,
                    },
                )

            self.logger.info(
                f"Transcribing audio file: {audio_file_path} ({file_size} bytes)"
            )

            # Create OpenAI client
            client = openai.OpenAI()

            # Open audio file and transcribe
            with open(audio_file_path, "rb") as audio_file:
                # OpenAI Whisper API returns different response types based on format
                transcript: Any = client.audio.transcriptions.create(  # pyright: ignore[reportUnknownVariableType]
                    model=self.config["model"],
                    file=audio_file,
                    language=self.config["language"],
                    response_format=self.config["response_format"],
                    temperature=self.config["temperature"],
                )

            # Extract transcript text
            transcript_text: str
            if self.config["response_format"] == "text":
                # OpenAI Whisper API returns different response types, str() handles all safely
                transcript_text = str(transcript)  # pyright: ignore[reportUnknownArgumentType]
            else:
                # For JSON formats, extract text field
                # OpenAI response objects have dynamic attributes based on format, getattr handles unknown types
                transcript_text = getattr(transcript, "text", str(transcript))  # pyright: ignore[reportUnknownArgumentType]

            # Save transcript to organized directory structure if transcript_dir is available
            transcript_file_path: Optional[str] = None
            if "transcript_dir" in self.config:
                transcript_dir: str = self.config["transcript_dir"]

                # Create transcript filename based on audio file
                audio_filename = os.path.basename(audio_file_path)
                transcript_filename = audio_filename.replace(
                    ".mp3", "_raw_transcript.txt"
                )
                transcript_file_path = os.path.join(transcript_dir, transcript_filename)

                # Save raw transcript
                try:
                    with open(transcript_file_path, "w", encoding="utf-8") as f:
                        f.write(transcript_text)
                    self.logger.info(f"Raw transcript saved to: {transcript_file_path}")
                except Exception as save_error:
                    self.logger.warning(f"Failed to save transcript file: {save_error}")

            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"Transcription completed. Length: {len(transcript_text)} characters"
            )

            return TaskResult.success(
                data={
                    "transcript": transcript_text,
                    "model_used": self.config["model"],
                    "language": self.config["language"],
                    "audio_file_path": audio_file_path,
                    "file_size": file_size,
                    "transcript_length": len(transcript_text),
                    "transcript_file_path": transcript_file_path,
                },
                execution_time=execution_time,
                metadata={
                    "task_name": self.name,
                    "audio_path": audio_file_path,
                    "transcript_file_path": transcript_file_path,
                },
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"AudioTranscriptTask operation failed: {str(e)}"
            self.logger.error(error_msg)
            return TaskResult.failure(
                error_message=error_msg,
                execution_time=execution_time,
                metadata={
                    "task_name": self.name,
                    "error_type": type(e).__name__,
                    "audio_path": audio_file_path
                    if "audio_file_path" in locals()
                    else "unknown",
                },
            )
