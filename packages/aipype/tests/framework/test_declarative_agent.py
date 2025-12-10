"""Integration tests for PipelineAgent (declarative syntax)."""

from typing import Annotated, Any, Dict, List

from pydantic import BaseModel

from aipype import PipelineAgent, task, Depends


class TestSimpleLinearPipeline:
    """Tests for simple linear task pipelines."""

    def test_linear_pipeline_execution(self) -> None:
        """Test a simple linear pipeline with 3 tasks."""

        class SimpleAgent(PipelineAgent):
            @task
            def step_one(self) -> Dict[str, Any]:
                return {"value": 1}

            @task
            def step_two(self, step_one: Dict[str, Any]) -> Dict[str, Any]:
                return {"value": step_one["value"] + 1}

            @task
            def step_three(self, step_two: Dict[str, Any]) -> Dict[str, Any]:
                return {"value": step_two["value"] + 1}

        agent = SimpleAgent("test", {})
        result = agent.run()

        assert result.is_success()
        assert result.completed_tasks == 3

        # Check final result
        final = agent.context.get_result("step_three")
        assert final is not None
        assert final["value"] == 3

    def test_task_receives_correct_data(self) -> None:
        """Test that tasks receive the correct data from dependencies."""

        class DataAgent(PipelineAgent):
            @task
            def producer(self) -> Dict[str, Any]:
                return {"message": "hello", "count": 42}

            @task
            def consumer(self, producer: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "received_message": producer["message"],
                    "received_count": producer["count"],
                }

        agent = DataAgent("test", {})
        result = agent.run()

        assert result.is_success()

        consumer_result = agent.context.get_result("consumer")
        assert consumer_result is not None
        assert consumer_result["received_message"] == "hello"
        assert consumer_result["received_count"] == 42


class TestParallelBranches:
    """Tests for parallel task execution."""

    def test_parallel_branches_execution(self) -> None:
        """Test pipeline with parallel branches."""

        class ParallelAgent(PipelineAgent):
            @task
            def source(self) -> Dict[str, Any]:
                return {"data": "source"}

            @task
            def branch_a(self, source: Dict[str, Any]) -> Dict[str, Any]:
                return {"branch": "a", "from": source["data"]}

            @task
            def branch_b(self, source: Dict[str, Any]) -> Dict[str, Any]:
                return {"branch": "b", "from": source["data"]}

            @task
            def merge(
                self, branch_a: Dict[str, Any], branch_b: Dict[str, Any]
            ) -> Dict[str, Any]:
                return {"merged": [branch_a["branch"], branch_b["branch"]]}

        agent = ParallelAgent("test", {"enable_parallel": True})
        result = agent.run()

        assert result.is_success()
        assert result.completed_tasks == 4

        # Verify execution plan has 3 phases: source -> [a,b] -> merge
        assert agent.execution_plan is not None
        assert agent.execution_plan.total_phases() == 3

        # Verify merge result
        merge_result = agent.context.get_result("merge")
        assert merge_result is not None
        assert set(merge_result["merged"]) == {"a", "b"}

    def test_multiple_zero_dependency_tasks(self) -> None:
        """Test pipeline with multiple tasks that have no dependencies."""

        class MultiSourceAgent(PipelineAgent):
            @task
            def source_a(self) -> Dict[str, Any]:
                return {"source": "a"}

            @task
            def source_b(self) -> Dict[str, Any]:
                return {"source": "b"}

            @task
            def combine(
                self, source_a: Dict[str, Any], source_b: Dict[str, Any]
            ) -> Dict[str, Any]:
                return {"combined": [source_a["source"], source_b["source"]]}

        agent = MultiSourceAgent("test", {"enable_parallel": True})
        result = agent.run()

        assert result.is_success()
        assert result.completed_tasks == 3

        # First phase should have both source tasks
        assert agent.execution_plan is not None
        assert agent.execution_plan.total_phases() == 2

        combine_result = agent.context.get_result("combine")
        assert combine_result is not None
        assert set(combine_result["combined"]) == {"a", "b"}


class TestCircularDependencyDetection:
    """Tests for circular dependency detection."""

    def test_circular_dependency_results_in_no_tasks(self) -> None:
        """Test that circular dependencies result in no tasks being set up.

        Circular dependencies are detected during setup_tasks() which raises
        a ValueError. The base PipelineAgent._auto_setup_tasks() catches this
        and logs a warning, leaving the agent with no tasks.
        """

        class CircularAgent(PipelineAgent):
            @task
            def task_a(self, task_b: Dict[str, Any]) -> Dict[str, Any]:
                return {}

            @task
            def task_b(self, task_a: Dict[str, Any]) -> Dict[str, Any]:
                return {}

        agent = CircularAgent("test", {})
        # Agent is created but has no tasks due to circular dependency
        assert len(agent.tasks) == 0

    def test_three_way_circular_dependency_results_in_no_tasks(self) -> None:
        """Test detection of circular dependencies with 3 tasks."""

        class ThreeWayCircularAgent(PipelineAgent):
            @task
            def task_a(self, task_c: Dict[str, Any]) -> Dict[str, Any]:
                return {}

            @task
            def task_b(self, task_a: Dict[str, Any]) -> Dict[str, Any]:
                return {}

            @task
            def task_c(self, task_b: Dict[str, Any]) -> Dict[str, Any]:
                return {}

        agent = ThreeWayCircularAgent("test", {})
        # Agent is created but has no tasks due to circular dependency
        assert len(agent.tasks) == 0


class TestConfigAccess:
    """Tests for accessing agent config from tasks."""

    def test_config_access_in_task(self) -> None:
        """Test that tasks can access self.config."""

        class ConfigAgent(PipelineAgent):
            @task
            def fetch(self) -> Dict[str, Any]:
                topic = self.config["topic"]
                max_results = self.config.get("max_results", 10)
                return {"topic": topic, "max_results": max_results}

        agent = ConfigAgent("test", {"topic": "AI", "max_results": 5})
        result = agent.run()

        assert result.is_success()

        fetch_result = agent.context.get_result("fetch")
        assert fetch_result is not None
        assert fetch_result["topic"] == "AI"
        assert fetch_result["max_results"] == 5

    def test_config_access_with_dependency(self) -> None:
        """Test that tasks can access both config and dependencies."""

        class MixedAgent(PipelineAgent):
            @task
            def fetch(self) -> Dict[str, Any]:
                return {"data": "fetched"}

            @task
            def process(self, fetch: Dict[str, Any]) -> Dict[str, Any]:
                prefix = self.config.get("prefix", "default")
                return {"result": f"{prefix}: {fetch['data']}"}

        agent = MixedAgent("test", {"prefix": "PROCESSED"})
        result = agent.run()

        assert result.is_success()

        process_result = agent.context.get_result("process")
        assert process_result is not None
        assert process_result["result"] == "PROCESSED: fetched"


class TestExplicitDependsPath:
    """Tests for explicit Depends() path specification."""

    def test_explicit_depends_path(self) -> None:
        """Test using Annotated[T, Depends()] for explicit paths."""

        class ExplicitDepsAgent(PipelineAgent):
            @task
            def search(self) -> Dict[str, Any]:
                return {"content": "article text", "url": "http://example.com"}

            @task
            def process(
                self, content: Annotated[str, Depends("search.content")]
            ) -> Dict[str, Any]:
                return {"processed": f"Processed: {content}"}

        agent = ExplicitDepsAgent("test", {})
        result = agent.run()

        assert result.is_success()

        process_result = agent.context.get_result("process")
        assert process_result is not None
        assert process_result["processed"] == "Processed: article text"

    def test_mixed_implicit_and_explicit_deps(self) -> None:
        """Test mixing implicit parameter names and explicit Depends()."""

        class MixedDepsAgent(PipelineAgent):
            @task
            def task_a(self) -> Dict[str, Any]:
                return {"value": 10}

            @task
            def task_b(self) -> Dict[str, Any]:
                return {"content": "text", "extra": "ignored"}

            @task
            def task_c(
                self,
                task_a: Dict[str, Any],  # Implicit: task_a.data
                text: Annotated[str, Depends("task_b.content")],  # Explicit
            ) -> Dict[str, Any]:
                return {"combined": f"{task_a['value']}:{text}"}

        agent = MixedDepsAgent("test", {})
        result = agent.run()

        assert result.is_success()

        task_c_result = agent.context.get_result("task_c")
        assert task_c_result is not None
        assert task_c_result["combined"] == "10:text"


class TestReturnTypes:
    """Tests for various return types from tasks."""

    def test_dict_return(self) -> None:
        """Test task returning dict."""

        class DictAgent(PipelineAgent):
            @task
            def produce(self) -> Dict[str, Any]:
                return {"key": "value", "count": 42}

        agent = DictAgent("test", {})
        result = agent.run()

        assert result.is_success()

        produce_result = agent.context.get_result("produce")
        assert produce_result is not None
        # Original keys are preserved at top level
        assert produce_result["key"] == "value"
        assert produce_result["count"] == 42
        # Full dict is also available under 'data'
        assert produce_result["data"] == {"key": "value", "count": 42}

    def test_string_return(self) -> None:
        """Test task returning string."""

        class StringAgent(PipelineAgent):
            @task
            def produce(self) -> str:
                return "hello world"

        agent = StringAgent("test", {})
        result = agent.run()

        assert result.is_success()

        produce_result = agent.context.get_result("produce")
        assert produce_result is not None
        assert produce_result["output"] == "hello world"
        assert produce_result["data"] == "hello world"

    def test_list_return(self) -> None:
        """Test task returning list."""

        class ListAgent(PipelineAgent):
            @task
            def produce(self) -> List[int]:
                return [1, 2, 3]

        agent = ListAgent("test", {})
        result = agent.run()

        assert result.is_success()

        produce_result = agent.context.get_result("produce")
        assert produce_result is not None
        assert produce_result["output"] == [1, 2, 3]

    def test_pydantic_return(self) -> None:
        """Test task returning Pydantic model."""

        class OutputModel(BaseModel):
            value: int
            items: List[str]

        class PydanticAgent(PipelineAgent):
            @task
            def produce(self) -> OutputModel:
                return OutputModel(value=10, items=["a", "b"])

        agent = PydanticAgent("test", {})
        result = agent.run()

        assert result.is_success()

        produce_result = agent.context.get_result("produce")
        assert produce_result is not None
        # Original fields are preserved at top level
        assert produce_result["value"] == 10
        assert produce_result["items"] == ["a", "b"]
        # Full dict is also available under 'data'
        assert produce_result["data"] == {"value": 10, "items": ["a", "b"]}


class TestNoTasksAgent:
    """Tests for agents with no @task methods."""

    def test_no_tasks_warning(self) -> None:
        """Test that agent with no @task methods logs warning."""

        class EmptyAgent(PipelineAgent):
            pass

        agent = EmptyAgent("test", {})
        assert len(agent.tasks) == 0


class TestAgentStr:
    """Tests for agent string representation."""

    def test_str_representation(self) -> None:
        """Test __str__ method."""

        class SimpleAgent(PipelineAgent):
            @task
            def task_one(self) -> Dict[str, Any]:
                return {"value": 1}

        agent = SimpleAgent("my_agent", {})
        result = str(agent)

        assert "PipelineAgent" in result
        assert "my_agent" in result
        assert "tasks=1" in result


class TestErrorHandling:
    """Tests for error handling in tasks."""

    def test_task_exception_captured(self) -> None:
        """Test that exceptions in tasks are captured as failures."""

        class FailingAgent(PipelineAgent):
            @task
            def failing_task(self) -> Dict[str, Any]:
                raise ValueError("Something went wrong")

        agent = FailingAgent("test", {})
        result = agent.run()

        # Pipeline should report error
        assert result.is_error() or result.is_partial()
        assert result.failed_tasks >= 1

    def test_dependency_failure_propagation(self) -> None:
        """Test that dependent tasks don't run if dependency fails."""

        class DependencyFailAgent(PipelineAgent):
            @task
            def failing_task(self) -> Dict[str, Any]:
                raise ValueError("Fail")

            @task
            def dependent_task(self, failing_task: Dict[str, Any]) -> Dict[str, Any]:
                return {"should_not_run": True}

        agent = DependencyFailAgent("test", {"stop_on_failure": True})
        result = agent.run()

        assert result.is_error() or result.is_partial()

        # Dependent task should not have completed
        dependent_result = agent.context.get_result("dependent_task")
        assert dependent_result is None


class TestTaskDiscovery:
    """Tests for task method discovery."""

    def test_private_methods_not_discovered(self) -> None:
        """Test that private methods are not discovered as tasks."""

        class PrivateAgent(PipelineAgent):
            @task
            def public_task(self) -> Dict[str, Any]:
                return {"public": True}

            def _private_method(self) -> Dict[str, Any]:
                return {"private": True}

        agent = PrivateAgent("test", {})
        assert len(agent.tasks) == 1

        task_names = [t.name for t in agent.tasks]
        assert "public_task" in task_names
        assert "_private_method" not in task_names

    def test_non_task_methods_not_discovered(self) -> None:
        """Test that regular methods are not discovered as tasks."""

        class MixedAgent(PipelineAgent):
            @task
            def task_method(self) -> Dict[str, Any]:
                return {"task": True}

            def helper_method(self) -> str:
                return "helper"

        agent = MixedAgent("test", {})
        assert len(agent.tasks) == 1

        task_names = [t.name for t in agent.tasks]
        assert "task_method" in task_names
        assert "helper_method" not in task_names
