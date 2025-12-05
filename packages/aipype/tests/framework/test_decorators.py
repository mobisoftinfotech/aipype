"""Tests for task decorator, Depends, and dependency inference."""

from typing import Annotated, Optional

import pytest

from aipype.decorators import Depends, task
from aipype.dependency_inference import (
    _extract_depends_path,
    _is_optional_type,
    get_dependency_task_name,
    infer_dependencies_from_signature,
)
from aipype.task_dependencies import DependencyType


class TestTaskDecorator:
    """Tests for @task decorator."""

    def test_task_marks_function(self) -> None:
        """Test that @task adds metadata to function."""

        @task
        def my_task(self) -> str:  # pyright: ignore[reportUnusedFunction]
            return "result"

        assert hasattr(my_task, "_is_pipeline_task")
        assert my_task._is_pipeline_task is True  # pyright: ignore[reportFunctionMemberAccess]
        assert my_task._task_name == "my_task"  # pyright: ignore[reportFunctionMemberAccess]

    def test_task_preserves_function_behavior(self) -> None:
        """Test that @task preserves original function behavior."""

        @task
        def add(self: object, a: int, b: int) -> int:
            return a + b

        class MockAgent:
            pass

        result = add(MockAgent(), 2, 3)
        assert result == 5

    def test_task_preserves_docstring(self) -> None:
        """Test that @task preserves function docstring."""

        @task
        def documented_task(self) -> str:  # pyright: ignore[reportUnusedFunction]
            """This is a documented task."""
            return "result"

        assert documented_task.__doc__ == "This is a documented task."

    def test_task_preserves_function_name(self) -> None:
        """Test that @task preserves function __name__."""

        @task
        def my_specific_task(self) -> str:  # pyright: ignore[reportUnusedFunction]
            return "result"

        assert my_specific_task.__name__ == "my_specific_task"


class TestDepends:
    """Tests for Depends type."""

    def test_depends_stores_path(self) -> None:
        """Test that Depends stores the path correctly."""
        dep = Depends("task.field")
        assert dep.path == "task.field"

    def test_depends_requires_dot_notation(self) -> None:
        """Test that Depends requires dot notation in path."""
        with pytest.raises(ValueError, match="task.field"):
            Depends("invalid")

    def test_depends_rejects_empty_path(self) -> None:
        """Test that Depends rejects empty path."""
        with pytest.raises(ValueError):
            Depends("")

    def test_depends_class_getitem(self) -> None:
        """Test Depends["path"] syntax."""
        dep = Depends["task.nested.field"]
        assert dep.path == "task.nested.field"

    def test_depends_repr(self) -> None:
        """Test Depends string representation."""
        dep = Depends("task.field")
        assert repr(dep) == "Depends('task.field')"

    def test_depends_with_nested_path(self) -> None:
        """Test Depends with deeply nested path."""
        dep = Depends("task.level1.level2.level3")
        assert dep.path == "task.level1.level2.level3"


class TestDependencyInference:
    """Tests for automatic dependency inference."""

    def test_infer_from_param_name(self) -> None:
        """Test dependency inference from parameter name matching task name."""

        def method(self: object, fetch_data: dict) -> str:  # pyright: ignore[reportUnusedFunction]
            return ""

        deps = infer_dependencies_from_signature(
            method, known_task_names={"fetch_data"}
        )

        assert len(deps) == 1
        assert deps[0].name == "fetch_data"
        assert deps[0].source_path == "fetch_data.data"
        assert deps[0].is_required()

    def test_infer_explicit_depends(self) -> None:
        """Test dependency inference from explicit Annotated[T, Depends()]."""

        def method(
            self: object,
            content: Annotated[str, Depends("llm_task.content")],
        ) -> str:  # pyright: ignore[reportUnusedFunction]
            return ""

        deps = infer_dependencies_from_signature(method, known_task_names=set())

        assert len(deps) == 1
        assert deps[0].name == "content"
        assert deps[0].source_path == "llm_task.content"

    def test_infer_optional_with_default(self) -> None:
        """Test that parameters with defaults are optional."""

        def method(
            self: object,
            config: dict = None,  # pyright: ignore[reportArgumentType]
        ) -> str:  # pyright: ignore[reportUnusedFunction]
            return ""

        deps = infer_dependencies_from_signature(method, known_task_names={"config"})

        assert len(deps) == 1
        assert deps[0].is_optional()
        assert deps[0].default_value is None

    def test_infer_optional_type_hint(self) -> None:
        """Test that Optional[T] type hints are detected as optional."""

        def method(self: object, data: Optional[dict]) -> str:  # pyright: ignore[reportUnusedFunction]
            return ""

        deps = infer_dependencies_from_signature(method, known_task_names={"data"})

        assert len(deps) == 1
        assert deps[0].is_optional()
        assert deps[0].dependency_type == DependencyType.OPTIONAL

    def test_skip_self_parameter(self) -> None:
        """Test that 'self' parameter is skipped."""

        def method(self: object, data: dict) -> str:  # pyright: ignore[reportUnusedFunction]
            return ""

        deps = infer_dependencies_from_signature(
            method, known_task_names={"self", "data"}
        )

        # Should only have 'data', not 'self'
        assert len(deps) == 1
        assert deps[0].name == "data"

    def test_no_deps_for_unknown_params(self) -> None:
        """Test that unknown params don't create dependencies."""

        def method(self: object, unknown_param: str) -> str:  # pyright: ignore[reportUnusedFunction]
            return ""

        deps = infer_dependencies_from_signature(
            method, known_task_names={"other_task"}
        )

        # unknown_param doesn't match any task name
        assert len(deps) == 0

    def test_multiple_dependencies(self) -> None:
        """Test inference of multiple dependencies."""

        def method(
            self: object,
            task_a: dict,
            task_b: dict,
            task_c: Annotated[str, Depends("task_c.content")],
        ) -> str:  # pyright: ignore[reportUnusedFunction]
            return ""

        deps = infer_dependencies_from_signature(
            method, known_task_names={"task_a", "task_b"}
        )

        assert len(deps) == 3
        dep_names = {d.name for d in deps}
        assert dep_names == {"task_a", "task_b", "task_c"}

    def test_mixed_optional_required(self) -> None:
        """Test mix of optional and required dependencies."""

        def method(
            self: object,
            required_task: dict,
            optional_task: Optional[dict] = None,
        ) -> str:  # pyright: ignore[reportUnusedFunction]
            return ""

        deps = infer_dependencies_from_signature(
            method, known_task_names={"required_task", "optional_task"}
        )

        assert len(deps) == 2

        required = next(d for d in deps if d.name == "required_task")
        optional = next(d for d in deps if d.name == "optional_task")

        assert required.is_required()
        assert optional.is_optional()


class TestIsOptionalType:
    """Tests for _is_optional_type helper."""

    def test_optional_type(self) -> None:
        """Test detection of Optional[T]."""
        assert _is_optional_type(Optional[str]) is True

    def test_non_optional_type(self) -> None:
        """Test non-optional types return False."""
        assert _is_optional_type(str) is False
        assert _is_optional_type(dict) is False
        assert _is_optional_type(list) is False

    def test_none_hint(self) -> None:
        """Test None hint returns False."""
        assert _is_optional_type(None) is False


class TestExtractDependsPath:
    """Tests for _extract_depends_path helper."""

    def test_extract_from_annotated(self) -> None:
        """Test extracting Depends from Annotated type."""
        hint = Annotated[str, Depends("task.field")]
        path = _extract_depends_path(hint)
        assert path == "task.field"

    def test_no_depends_in_annotated(self) -> None:
        """Test Annotated without Depends returns None."""
        hint = Annotated[str, "some other metadata"]
        path = _extract_depends_path(hint)
        assert path is None

    def test_plain_type_returns_none(self) -> None:
        """Test plain type returns None."""
        path = _extract_depends_path(str)
        assert path is None

    def test_none_returns_none(self) -> None:
        """Test None input returns None."""
        path = _extract_depends_path(None)
        assert path is None


class TestGetDependencyTaskName:
    """Tests for get_dependency_task_name helper."""

    def test_simple_path(self) -> None:
        """Test extracting task name from simple path."""
        assert get_dependency_task_name("task.field") == "task"

    def test_nested_path(self) -> None:
        """Test extracting task name from nested path."""
        assert get_dependency_task_name("task.level1.level2") == "task"

    def test_single_level(self) -> None:
        """Test path with single dot."""
        assert get_dependency_task_name("fetch.data") == "fetch"
