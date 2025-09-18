"""Core framework components for aipype."""

from .pipeline_agent import PipelineAgent
from .base_task import BaseTask
from .task_result import TaskResult, TaskStatus
from .task_context import TaskContext
from .task_dependencies import TaskDependency, DependencyType
from .llm_task import LLMTask
from .search_task import SearchTask
from .conditional_task import ConditionalTask
from .transform_task import TransformTask

__all__ = [
    "PipelineAgent",
    "BaseTask",
    "TaskResult",
    "TaskStatus",
    "TaskContext",
    "TaskDependency",
    "DependencyType",
    "LLMTask",
    "SearchTask",
    "ConditionalTask",
    "TransformTask",
]
