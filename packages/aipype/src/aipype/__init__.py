"""aipype: A modular AI agent framework with declarative pipeline-based task orchestration."""

__version__ = "0.1.0a3"

# Core framework exports
from .pipeline_agent import PipelineAgent, TaskExecutionPlan
from .base_task import BaseTask
from .task_result import (
    TaskResult,
    TaskStatus,
    wrap_legacy_result,
    unwrap_to_legacy,
)
from .agent_run_result import (
    AgentRunResult,
    AgentRunStatus,
)
from .task_context import TaskContext
from .task_dependencies import (
    TaskDependency,
    DependencyType,
    DependencyResolver,
    create_required_dependency,
    create_optional_dependency,
    extract_urls_from_results,
    combine_article_content,
    format_search_query,
)
from .llm_task import LLMTask
from .search_task import SearchTask
from .conditional_task import (
    ConditionalTask,
    threshold_condition,
    contains_condition,
    list_size_condition,
    success_rate_condition,
    quality_gate_condition,
    log_action,
    increment_counter_action,
    set_flag_action,
)
from .transform_task import (
    TransformTask,
    extract_field_from_list,
    combine_text_fields,
    filter_by_condition,
    aggregate_numeric_field,
    format_as_markdown_list,
)
from .tools import (
    tool,
    ToolMetadata,
    ToolSchemaGenerator,
    search_with_content,
)
from .tool_registry import ToolRegistry
from .tool_executor import ToolExecutor

# Tasklib exports
from .tasklib.web.batch_article_summarize_task import (
    BatchArticleSummarizeTask,
)
from .tasklib.io.file_save_task import FileSaveTask
from .tasklib.web.url_fetch_task import URLFetchTask
from .tasklib.media.extract_audio_from_video_task import (
    ExtractAudioFromVideoTask,
)
from .tasklib.media.audio_transcript_task import AudioTranscriptTask

# Utils exports
from .utils.common import (
    setup_logger,
    timestamp,
    safe_dict_get,
    flatten_list,
    validate_required_fields,
)
from .utils.base_searcher import SearchResult, SearchResponse, BaseSearcher
from .utils.serper_searcher import SerperSearcher
from .utils.url_fetcher import fetch_main_text, fetch_url, URLFetcher
from .utils.display import print_header, print_message_box

__all__ = [
    # Core framework
    "PipelineAgent",
    "TaskExecutionPlan",
    "BaseTask",
    "TaskResult",
    "TaskStatus",
    "wrap_legacy_result",
    "unwrap_to_legacy",
    "AgentRunResult",
    "AgentRunStatus",
    "TaskContext",
    "TaskDependency",
    "DependencyType",
    "DependencyResolver",
    "create_required_dependency",
    "create_optional_dependency",
    "extract_urls_from_results",
    "combine_article_content",
    "format_search_query",
    "LLMTask",
    "SearchTask",
    "ConditionalTask",
    "threshold_condition",
    "contains_condition",
    "list_size_condition",
    "success_rate_condition",
    "quality_gate_condition",
    "log_action",
    "increment_counter_action",
    "set_flag_action",
    "TransformTask",
    "extract_field_from_list",
    "combine_text_fields",
    "filter_by_condition",
    "aggregate_numeric_field",
    "format_as_markdown_list",
    # Tool system
    "tool",
    "ToolMetadata",
    "ToolSchemaGenerator",
    "search_with_content",
    "ToolRegistry",
    "ToolExecutor",
    # Tasklib
    "BatchArticleSummarizeTask",
    "FileSaveTask",
    "URLFetchTask",
    "ExtractAudioFromVideoTask",
    "AudioTranscriptTask",
    # Utils
    "setup_logger",
    "timestamp",
    "safe_dict_get",
    "flatten_list",
    "validate_required_fields",
    "SearchResult",
    "SearchResponse",
    "BaseSearcher",
    "SerperSearcher",
    "fetch_main_text",
    "fetch_url",
    "URLFetcher",
    "print_header",
    "print_message_box",
]
