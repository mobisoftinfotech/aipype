"""Integration tests for PipelineAgent with @task decorators.

These tests verify the declarative syntax works correctly in real execution
scenarios. Integration tests use real components (NO MOCKS).
"""

from typing import Annotated, Any, Dict, List


from aipype import (
    PipelineAgent,
    task,
    Depends,
    transform,
    TaskResult,
)


class SimpleLinearAgent(PipelineAgent):
    """Simple agent with linear task dependencies for testing."""

    @task
    def step_one(self) -> Dict[str, Any]:
        """First task - no dependencies."""
        return {"value": self.config.get("initial_value", 1), "step": 1}

    @task
    def step_two(self, step_one: Dict[str, Any]) -> Dict[str, Any]:
        """Second task - depends on step_one."""
        return {"value": step_one["value"] * 2, "step": 2}

    @task
    def step_three(self, step_two: Dict[str, Any]) -> Dict[str, Any]:
        """Third task - depends on step_two."""
        return {"value": step_two["value"] + 10, "step": 3}


class ParallelBranchAgent(PipelineAgent):
    """Agent with parallel branches for testing execution phases."""

    @task
    def source(self) -> Dict[str, Any]:
        """Source task - runs first."""
        return {"data": self.config.get("source_data", "initial")}

    @task
    def branch_a(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Branch A - depends on source."""
        return {"branch": "a", "from_source": source["data"]}

    @task
    def branch_b(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Branch B - depends on source, can run parallel with branch_a."""
        return {"branch": "b", "from_source": source["data"]}

    @task
    def merge(
        self, branch_a: Dict[str, Any], branch_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge task - depends on both branches."""
        return {
            "merged": True,
            "branches": [branch_a["branch"], branch_b["branch"]],
        }


class ConfigAccessAgent(PipelineAgent):
    """Agent demonstrating config access pattern."""

    @task
    def use_config(self) -> Dict[str, Any]:
        """Task that uses various config values."""
        return {
            "topic": self.config.get("topic", "default_topic"),
            "max_results": self.config.get("max_results", 5),
            "temperature": self.config.get("temperature", 0.7),
        }

    @task
    def process_config(self, use_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process the config values."""
        return {
            "processed_topic": use_config["topic"].upper(),
            "doubled_max": use_config["max_results"] * 2,
        }


class ExplicitDependsAgent(PipelineAgent):
    """Agent demonstrating explicit Depends() for field extraction."""

    @task
    def produce_data(self) -> Dict[str, Any]:
        """Produce data with multiple fields."""
        return {
            "content": "This is the content",
            "metadata": {"author": "test", "version": 1},
            "items": ["a", "b", "c"],
        }

    @task
    def use_content(
        self,
        content: Annotated[str, Depends("produce_data.content")],
    ) -> Dict[str, Any]:
        """Use only the content field."""
        return {"received_content": content, "length": len(content)}

    @task
    def use_items(
        self,
        items: Annotated[List[str], Depends("produce_data.items")],
    ) -> Dict[str, Any]:
        """Use only the items field."""
        return {"received_items": items, "count": len(items)}


class TransformHelperAgent(PipelineAgent):
    """Agent demonstrating transform() helper usage."""

    @task
    def source_data(self) -> Dict[str, Any]:
        """Produce source data."""
        return {
            "items": [
                {"name": "item1", "value": 10},
                {"name": "item2", "value": 20},
                {"name": "item3", "value": 30},
            ]
        }

    @task
    def transform_data(self, source_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform the data using helper."""
        items = source_data.get("items", [])
        return transform(
            items,
            fn=lambda x: [item["name"] for item in x],
            output_name="names",
        )


class ValidationTaskAgent(PipelineAgent):
    """Agent with validation task demonstrating error handling."""

    @task
    def fetch_items(self) -> Dict[str, Any]:
        """Fetch items."""
        count = self.config.get("item_count", 3)
        return {"items": list(range(count)), "count": count}

    @task
    def validate_items(self, fetch_items: Dict[str, Any]) -> Dict[str, Any]:
        """Validate minimum item count."""
        min_required = self.config.get("min_required", 5)
        if fetch_items["count"] < min_required:
            raise RuntimeError(
                f"Insufficient items: {fetch_items['count']} < {min_required}"
            )
        return {"validated": True}

    @task
    def process_items(
        self,
        fetch_items: Dict[str, Any],
        validate_items: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process items after validation."""
        return {
            "processed": True,
            "item_count": fetch_items["count"],
        }


class TestSimpleLinearPipeline:
    """Test simple linear task execution."""

    def test_linear_execution_order(self) -> None:
        """Test tasks execute in correct order."""
        agent = SimpleLinearAgent("test", {"initial_value": 5})
        result = agent.run()

        assert result.is_success()
        assert result.completed_tasks == 3

        # Verify final value: 5 * 2 + 10 = 20
        final_result = agent.context.get_result("step_three")
        assert final_result["value"] == 20
        assert final_result["step"] == 3

    def test_dependency_data_flow(self) -> None:
        """Test data flows correctly through dependencies."""
        agent = SimpleLinearAgent("test", {"initial_value": 10})
        agent.run()

        step_one = agent.context.get_result("step_one")
        step_two = agent.context.get_result("step_two")
        step_three = agent.context.get_result("step_three")

        assert step_one["value"] == 10
        assert step_two["value"] == 20  # 10 * 2
        assert step_three["value"] == 30  # 20 + 10


class TestParallelBranches:
    """Test parallel branch execution."""

    def test_parallel_branches_complete(self) -> None:
        """Test both branches complete and merge receives both."""
        agent = ParallelBranchAgent("test", {"source_data": "test_input"})
        result = agent.run()

        assert result.is_success()
        assert result.completed_tasks == 4

        merge_result = agent.context.get_result("merge")
        assert merge_result["merged"] is True
        assert set(merge_result["branches"]) == {"a", "b"}

    def test_execution_phases(self) -> None:
        """Test correct execution phases with parallel branches."""
        agent = ParallelBranchAgent(
            "test", {"source_data": "input", "enable_parallel": True}
        )
        agent.run()

        # Verify execution plan has correct phases
        plan = agent.execution_plan
        assert plan.total_phases() == 3  # source -> [a,b] -> merge


class TestConfigAccess:
    """Test self.config access patterns."""

    def test_config_values_accessible(self) -> None:
        """Test config values are accessible in tasks."""
        agent = ConfigAccessAgent(
            "test",
            {"topic": "AI trends", "max_results": 10, "temperature": 0.3},
        )
        result = agent.run()

        assert result.is_success()

        config_result = agent.context.get_result("use_config")
        assert config_result["topic"] == "AI trends"
        assert config_result["max_results"] == 10
        assert config_result["temperature"] == 0.3

    def test_config_defaults(self) -> None:
        """Test default values when config keys missing."""
        agent = ConfigAccessAgent("test", {})
        result = agent.run()

        assert result.is_success()

        config_result = agent.context.get_result("use_config")
        assert config_result["topic"] == "default_topic"
        assert config_result["max_results"] == 5


class TestExplicitDepends:
    """Test Depends() for explicit field extraction."""

    def test_explicit_field_extraction(self) -> None:
        """Test Depends() extracts specific fields."""
        agent = ExplicitDependsAgent("test", {})
        result = agent.run()

        assert result.is_success()

        content_result = agent.context.get_result("use_content")
        assert content_result["received_content"] == "This is the content"
        assert content_result["length"] == 19

        items_result = agent.context.get_result("use_items")
        assert items_result["received_items"] == ["a", "b", "c"]
        assert items_result["count"] == 3


class TestTransformHelper:
    """Test transform() helper function."""

    def test_transform_extracts_values(self) -> None:
        """Test transform() applies function and returns result."""
        agent = TransformHelperAgent("test", {})
        result = agent.run()

        assert result.is_success()

        transform_result = agent.context.get_result("transform_data")
        assert transform_result["names"] == ["item1", "item2", "item3"]
        # The data key contains the same transformed result
        assert transform_result["data"]["names"] == ["item1", "item2", "item3"]


class TestValidationErrors:
    """Test error handling in validation tasks."""

    def test_validation_failure(self) -> None:
        """Test validation task fails when conditions not met."""
        agent = ValidationTaskAgent(
            "test",
            {"item_count": 2, "min_required": 5},
        )
        result = agent.run()

        # Should fail due to validation - check failed tasks count
        assert result.failed_tasks > 0
        # Or check that completed tasks is less than total
        assert result.completed_tasks < 3

    def test_validation_success(self) -> None:
        """Test validation passes when conditions met."""
        agent = ValidationTaskAgent(
            "test",
            {"item_count": 10, "min_required": 5},
        )
        result = agent.run()

        assert result.is_success()

        process_result = agent.context.get_result("process_items")
        assert process_result["processed"] is True
        assert process_result["item_count"] == 10


class TestCircularDependencyDetection:
    """Test circular dependency detection."""

    def test_circular_dependency_detected(self) -> None:
        """Test that circular dependencies result in failed agent run.

        Note: The framework logs a warning for circular dependencies during
        setup but doesn't raise an exception. The run will fail with no tasks.
        """

        # Define class inside test to avoid polluting module scope
        class CircularAgent(PipelineAgent):
            @task
            def task_a(
                self, task_b: Annotated[Dict[str, Any], Depends("task_b.data")]
            ) -> Dict[str, Any]:
                return {"from": "a"}

            @task
            def task_b(
                self, task_a: Annotated[Dict[str, Any], Depends("task_a.data")]
            ) -> Dict[str, Any]:
                return {"from": "b"}

        # Circular dependencies are detected during setup but logged as warning
        # The agent will be created but have no tasks
        agent = CircularAgent("test", {})
        # Run should complete but with 0 tasks executed (none could be scheduled)
        result = agent.run()
        # The agent should have 0 completed tasks due to circular dependency
        assert result.completed_tasks == 0


class TestTaskResultReturn:
    """Test returning TaskResult directly from tasks."""

    def test_task_result_passthrough(self) -> None:
        """Test TaskResult is passed through correctly."""

        class TaskResultAgent(PipelineAgent):
            @task
            def returns_task_result(self) -> TaskResult:
                return TaskResult.success(
                    data={"custom_key": "custom_value"},
                    metadata={"source": "manual_result"},
                )

        agent = TaskResultAgent("test", {})
        result = agent.run()

        assert result.is_success()

        task_output = agent.context.get_result("returns_task_result")
        assert task_output["custom_key"] == "custom_value"


class TestPydanticModelReturn:
    """Test returning Pydantic models from tasks."""

    def test_pydantic_model_converted_to_dict(self) -> None:
        """Test Pydantic model is converted to dict."""
        from pydantic import BaseModel

        class OutputModel(BaseModel):
            name: str
            count: int
            items: List[str]

        class PydanticAgent(PipelineAgent):
            @task
            def returns_pydantic(self) -> OutputModel:
                return OutputModel(
                    name="test",
                    count=3,
                    items=["a", "b", "c"],
                )

        agent = PydanticAgent("test", {})
        result = agent.run()

        assert result.is_success()

        output = agent.context.get_result("returns_pydantic")
        assert output["name"] == "test"
        assert output["count"] == 3
        assert output["items"] == ["a", "b", "c"]


class TestPrimitiveReturns:
    """Test returning primitive types from tasks."""

    def test_string_return(self) -> None:
        """Test string return is wrapped correctly."""

        class StringAgent(PipelineAgent):
            @task
            def returns_string(self) -> str:
                return "hello world"

        agent = StringAgent("test", {})
        result = agent.run()

        assert result.is_success()

        output = agent.context.get_result("returns_string")
        assert output["output"] == "hello world"
        assert output["data"] == "hello world"

    def test_list_return(self) -> None:
        """Test list return is wrapped correctly."""

        class ListAgent(PipelineAgent):
            @task
            def returns_list(self) -> List[int]:
                return [1, 2, 3, 4, 5]

        agent = ListAgent("test", {})
        result = agent.run()

        assert result.is_success()

        output = agent.context.get_result("returns_list")
        assert output["output"] == [1, 2, 3, 4, 5]
        assert output["data"] == [1, 2, 3, 4, 5]


class TestMultipleZeroDependencyTasks:
    """Test handling of multiple tasks with no dependencies."""

    def test_multiple_first_tasks(self) -> None:
        """Test multiple tasks with no dependencies all execute."""

        class MultiFirstAgent(PipelineAgent):
            @task
            def first_a(self) -> Dict[str, Any]:
                return {"from": "a"}

            @task
            def first_b(self) -> Dict[str, Any]:
                return {"from": "b"}

            @task
            def first_c(self) -> Dict[str, Any]:
                return {"from": "c"}

            @task
            def combine(
                self,
                first_a: Dict[str, Any],
                first_b: Dict[str, Any],
                first_c: Dict[str, Any],
            ) -> Dict[str, Any]:
                return {
                    "sources": [
                        first_a["from"],
                        first_b["from"],
                        first_c["from"],
                    ]
                }

        agent = MultiFirstAgent("test", {})
        result = agent.run()

        assert result.is_success()
        assert result.completed_tasks == 4

        combine_result = agent.context.get_result("combine")
        assert set(combine_result["sources"]) == {"a", "b", "c"}
