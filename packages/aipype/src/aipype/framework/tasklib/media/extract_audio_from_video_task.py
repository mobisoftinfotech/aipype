"""Extract audio from video file using moviepy and compress to MP3 format."""

import os
from typing import Dict, Any, override
from datetime import datetime
from pathlib import Path

from ...base_task import BaseTask
from ...task_result import TaskResult


class ExtractAudioFromVideoTask(BaseTask):
    """Extract audio from video file using moviepy and compress to MP3 format."""

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize video audio extraction task.

        Args:
            name: Task name
            config: Task configuration
        """
        super().__init__(name, config)
        self.validation_rules = {
            "required": ["input_video_path", "audio_dir"],
            "defaults": {
                "output_format": "mp3",
                "bitrate": "16k",  # Ultra-low bitrate for maximum compression
                "sample_rate": 22050,  # Lower sample rate for smaller size
            },
            "types": {
                "input_video_path": str,
                "output_format": str,
                "bitrate": str,
                "sample_rate": int,
                "audio_dir": str,
            },
            # Validation lambdas intentionally use dynamic typing for flexibility across field types
            "custom": {
                "input_video_path": lambda x: str(x).strip() != "",  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
                "bitrate": lambda x: str(x)  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
                in ["8k", "16k", "32k", "64k", "128k", "192k", "256k", "320k"],
                "sample_rate": lambda x: int(x) in [22050, 44100, 48000],  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
            },
        }

    @override
    def run(self) -> TaskResult:
        """Extract audio from video file and compress to MP3."""
        start_time = datetime.now()

        # Validate configuration
        validation_error = self._validate_or_fail(start_time)
        if validation_error:
            return validation_error

        try:
            # Prefer the public import path
            from moviepy.editor import VideoFileClip  # type: ignore[import-untyped]
        except Exception as editor_import_error:
            # Fallback for environments where the editor shim is unavailable (some moviepy 2.x builds)
            try:
                from moviepy.video.io.VideoFileClip import (
                    VideoFileClip,
                )
            except Exception as lowlevel_import_error:
                execution_time = (datetime.now() - start_time).total_seconds()
                error_msg = (
                    "ExtractAudioFromVideoTask operation failed: could not import MoviePy VideoFileClip. "
                    "Tried 'moviepy.editor' and 'moviepy.video.io.VideoFileClip'. "
                    f"Errors -> editor: {editor_import_error}; video.io: {lowlevel_import_error}. "
                    "Ensure 'moviepy' is installed and synced (e.g., 'uv add moviepy' && 'uv sync')."
                )
                self.logger.error(error_msg)
                return TaskResult.failure(
                    error_message=error_msg,
                    execution_time=execution_time,
                    metadata={
                        "task_name": self.name,
                        "error_type": "ImportError",
                    },
                )

        input_video_path = self.config["input_video_path"]
        bitrate = self.config["bitrate"]
        sample_rate = self.config["sample_rate"]
        audio_dir = self.config["audio_dir"]  # Provided by agent

        # Get video name for output file
        video_name = Path(input_video_path).stem

        try:
            # Validate input file exists
            if not os.path.isfile(input_video_path):
                execution_time = (datetime.now() - start_time).total_seconds()
                error_msg = f"ExtractAudioFromVideoTask operation failed: Input video file does not exist: {input_video_path}"
                self.logger.error(error_msg)
                return TaskResult.failure(
                    error_message=error_msg,
                    execution_time=execution_time,
                    metadata={
                        "task_name": self.name,
                        "error_type": "FileNotFoundError",
                        "input_path": input_video_path,
                    },
                )

            # Create output filename in organized audio directory
            output_audio_path = os.path.join(audio_dir, f"{video_name}_audio.mp3")

            self.logger.info(
                f"Extracting audio from {input_video_path} to {output_audio_path}"
            )

            # Load video file - MoviePy VideoFileClip class has dynamic attributes
            video: Any = VideoFileClip(input_video_path)  # pyright: ignore[reportUnknownVariableType]

            # MoviePy VideoFileClip has dynamic attributes not known at type checking time
            if video.audio is None:  # pyright: ignore[reportUnknownMemberType]
                execution_time = (datetime.now() - start_time).total_seconds()
                error_msg = f"ExtractAudioFromVideoTask operation failed: No audio track found in video file: {input_video_path}"
                self.logger.error(error_msg)
                video.close()  # pyright: ignore[reportUnknownMemberType]
                return TaskResult.failure(
                    error_message=error_msg,
                    execution_time=execution_time,
                    metadata={
                        "task_name": self.name,
                        "error_type": "NoAudioTrackError",
                        "input_path": input_video_path,
                    },
                )

            # Extract audio with compression for OpenAI Whisper (25MB limit)
            self.logger.info(f"Extracting audio to MP3 format (bitrate: {bitrate})")

            # Apply compression settings for OpenAI Whisper compatibility
            self.logger.info("Applying compression settings")

            # Extract audio with low bitrate compression (preserves stereo quality)
            # MoviePy AudioFileClip has dynamic attributes not known at type checking time
            audio_clip: Any = video.audio  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            audio_clip.write_audiofile(output_audio_path, bitrate=bitrate)  # pyright: ignore[reportUnknownMemberType]
            audio_clip.close()  # pyright: ignore[reportUnknownMemberType]

            # Clean up video object
            video.close()  # pyright: ignore[reportUnknownMemberType]

            # Check final file size
            file_size = os.path.getsize(output_audio_path)
            max_size = 25 * 1024 * 1024  # 25MB OpenAI Whisper limit

            if file_size > max_size:
                self.logger.warning(
                    f"Audio file ({file_size:,} bytes) still exceeds OpenAI limit ({max_size:,} bytes)"
                )
                # Note: With 16k bitrate and 22050 sample rate, this should be much smaller

            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"Audio extraction completed. Output: {output_audio_path} ({file_size} bytes)"
            )

            return TaskResult.success(
                data={
                    "audio_file_path": output_audio_path,
                    "file_size": file_size,
                    "bitrate": bitrate,
                    "sample_rate": sample_rate,
                    "original_video": input_video_path,
                    "audio_dir": audio_dir,
                },
                execution_time=execution_time,
                metadata={
                    "task_name": self.name,
                    "output_path": output_audio_path,
                },
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"ExtractAudioFromVideoTask operation failed: {str(e)}"
            self.logger.error(error_msg)
            return TaskResult.failure(
                error_message=error_msg,
                execution_time=execution_time,
                metadata={
                    "task_name": self.name,
                    "error_type": type(e).__name__,
                    "input_path": input_video_path,
                },
            )
