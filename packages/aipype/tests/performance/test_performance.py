"""Performance and scalability tests for the pipeline framework."""

import pytest
import time
import psutil
import os
from typing import Any, Dict, List, Optional

from typing import override
from unittest.mock import Mock, patch
from aipype import (
    PipelineAgent,
    BaseTask,
    TaskContext,
    TaskResult,
    TaskDependency,
    DependencyType,
    DependencyResolver,
    TaskExecutionPlan,
    AgentRunStatus,
)


class PerformanceTestTask(BaseTask):
    """Task designed for performance testing."""

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        dependencies: Optional[List[TaskDependency]] = None,
        execution_time: float = 0.01,
        memory_usage: int = 1024,
    ):
        super().__init__(name, config, dependencies)
        self.execution_time = execution_time  # Simulated execution time in seconds
        self.memory_usage = memory_usage  # Simulated memory usage in bytes
        self._memory_allocated: Optional[bytearray] = None

    @override
    def run(self) -> TaskResult:
        # Simulate work with controlled timing and memory usage
        start_time = time.time()

        # Allocate memory to simulate processing
        self._memory_allocated = bytearray(self.memory_usage)

        # Simulate processing time
        if self.execution_time > 0:
            time.sleep(self.execution_time)

        actual_time = time.time() - start_time

        return TaskResult.success(
            {
                "task_name": self.name,
                "simulated_time": self.execution_time,
                "actual_time": actual_time,
                "memory_used": self.memory_usage,
                "timestamp": time.time(),
            }
        )


class TestDependencyResolutionPerformance:
    """Test dependency resolution performance with various scenarios."""

    def test_dependency_resolution_performance(self) -> None:
        """Dependency resolution scales with number of tasks."""
        context = TaskContext()
        resolver = DependencyResolver(context)

        # Create cascading dependencies: task1 -> task2 -> task3 -> ... -> taskN
        num_tasks = 50
        tasks: List[PerformanceTestTask] = []

        for i in range(num_tasks):
            task_name = f"task_{i}"
            dependencies: List[TaskDependency] = []

            if i > 0:
                # Each task depends on the previous task
                dependencies.append(
                    TaskDependency(
                        "prev_data", f"task_{i - 1}.result", DependencyType.REQUIRED
                    )
                )

            task = PerformanceTestTask(task_name, {}, dependencies)
            tasks.append(task)

            # Store mock result for previous tasks
            if i > 0:
                context.store_result(
                    f"task_{i - 1}", {"result": f"data_from_task_{i - 1}"}
                )

        # Measure dependency resolution time
        resolution_times: List[float] = []

        for task in tasks:
            start_time = time.time()
            resolved_config = resolver.resolve_dependencies(task)
            resolution_time = time.time() - start_time

            resolution_times.append(resolution_time)

            # Verify resolution succeeded
            assert isinstance(resolved_config, dict)
            if task.get_dependencies():
                assert len(resolved_config) > 0

        # Verify resolution time doesn't grow excessively
        avg_resolution_time = sum(resolution_times) / len(resolution_times)
        max_resolution_time: float = max(resolution_times)

        assert avg_resolution_time < 0.01, (
            f"Average resolution time too slow: {avg_resolution_time:.5f}s"
        )
        assert max_resolution_time < 0.05, (
            f"Max resolution time too slow: {max_resolution_time:.5f}s"
        )

        # Resolution time should not grow linearly with task position
        # (i.e., later tasks shouldn't take much longer than earlier tasks)
        first_half_avg = sum(resolution_times[: num_tasks // 2]) / (num_tasks // 2)
        second_half_avg = sum(resolution_times[num_tasks // 2 :]) / (num_tasks // 2)

        # Second half should not be more than 3x slower than first half
        assert second_half_avg < first_half_avg * 3, (
            f"Resolution time grows too much: {first_half_avg:.5f}s -> {second_half_avg:.5f}s"
        )

    def test_complex_dependency_graph_performance(self) -> None:
        """Performance with complex dependency graphs (many-to-many relationships)."""
        context = TaskContext()
        resolver = DependencyResolver(context)

        # Create fan-out then fan-in dependency pattern
        # 1 root -> 10 processors -> 1 aggregator

        # Store root result
        context.store_result("root", {"data": "root_data"})

        # Create processor tasks (all depend on root)
        processor_tasks: List[BaseTask] = []
        for i in range(10):
            task = PerformanceTestTask(
                f"processor_{i}",
                {},
                [TaskDependency("root_data", "root.data", DependencyType.REQUIRED)],
            )
            processor_tasks.append(task)

            # Store processor result
            context.store_result(f"processor_{i}", {"processed": f"processed_data_{i}"})

        # Create aggregator task (depends on all processors)
        aggregator_dependencies: List[TaskDependency] = []
        for i in range(10):
            aggregator_dependencies.append(
                TaskDependency(
                    f"data_{i}", f"processor_{i}.processed", DependencyType.REQUIRED
                )
            )

        aggregator_task = PerformanceTestTask("aggregator", {}, aggregator_dependencies)

        # Measure resolution performance for complex task
        start_time = time.time()
        resolved_config = resolver.resolve_dependencies(aggregator_task)
        resolution_time = time.time() - start_time

        # Verify resolution succeeded
        assert len(resolved_config) == 10  # Should have all 10 dependencies resolved
        for i in range(10):
            assert f"data_{i}" in resolved_config
            assert resolved_config[f"data_{i}"] == f"processed_data_{i}"

        # Resolution should be fast even with many dependencies
        assert resolution_time < 0.1, (
            f"Complex dependency resolution too slow: {resolution_time:.4f}s"
        )

    def test_dependency_resolution_with_transformations(self) -> None:
        """Performance of dependency resolution with transformation functions."""
        context = TaskContext()
        resolver = DependencyResolver(context)

        # Store source data
        source_data = {
            "articles": [
                {"title": f"Article {i}", "content": f"Content {i}" * 100}
                for i in range(100)
            ]
        }
        context.store_result("source", source_data)

        # Create task with expensive transformation
        def expensive_transformation(articles: List[Dict[str, Any]]) -> str:
            # Simulate expensive processing
            result: List[str] = []
            for article in articles:
                # Some string processing
                processed: str = article["content"].upper().replace(" ", "_")
                title_processed: str = article["title"].upper()
                result.append(f"{title_processed}: {processed[:50]}...")
            return "\n".join(result)

        task = PerformanceTestTask(
            "transform_test",
            {},
            [
                TaskDependency(
                    "transformed_content",
                    "source.articles",
                    DependencyType.REQUIRED,
                    transform_func=expensive_transformation,
                )
            ],
        )

        # Measure transformation performance
        start_time = time.time()
        resolved_config = resolver.resolve_dependencies(task)
        transformation_time = time.time() - start_time

        # Verify transformation succeeded
        assert "transformed_content" in resolved_config
        transformed = resolved_config["transformed_content"]
        assert "ARTICLE 0:" in transformed
        assert "ARTICLE 99:" in transformed

        # Transformation should complete in reasonable time
        assert transformation_time < 1.0, (
            f"Transformation too slow: {transformation_time:.3f}s"
        )


class TestPipelineAgentPerformance:
    """Test PipelineAgent performance and scalability."""

    def test_parallel_execution_benefits(self) -> None:
        """Parallel task execution improves overall performance."""

        class ParallelTestAgent(PipelineAgent):
            @override
            def setup_tasks(self) -> List[BaseTask]:
                # Create 5 independent tasks that can run in parallel
                independent_tasks: List[BaseTask] = []
                for i in range(5):
                    task = PerformanceTestTask(
                        f"independent_{i}", {}, [], execution_time=0.1
                    )
                    independent_tasks.append(task)

                # Add a final task that depends on all independent tasks
                final_deps: List[TaskDependency] = []
                for i in range(5):
                    final_deps.append(
                        TaskDependency(
                            f"data_{i}",
                            f"independent_{i}.task_name",
                            DependencyType.REQUIRED,
                        )
                    )

                final_task = PerformanceTestTask(
                    "final", {}, final_deps, execution_time=0.05
                )

                return independent_tasks + [final_task]

        # Test with parallel execution enabled
        agent_parallel = ParallelTestAgent("parallel", {"enable_parallel": True})

        with patch(
            "aipype.task_dependencies.DependencyResolver"
        ) as mock_resolver_class:
            mock_resolver = Mock()
            mock_resolver_class.return_value = mock_resolver

            def resolve_deps(task: Any) -> Dict[str, Any]:
                if task.name.startswith("independent"):
                    return {}
                else:  # final task
                    return {f"data_{i}": f"independent_{i}" for i in range(5)}

            mock_resolver.resolve_dependencies.side_effect = resolve_deps

            start_time = time.time()
            result_parallel = agent_parallel.run()
            parallel_time = time.time() - start_time

        # Test with sequential execution (parallel disabled)
        agent_sequential = ParallelTestAgent("sequential", {"enable_parallel": False})

        with patch(
            "aipype.task_dependencies.DependencyResolver"
        ) as mock_resolver_class:
            mock_resolver = Mock()
            mock_resolver_class.return_value = mock_resolver
            mock_resolver.resolve_dependencies.side_effect = resolve_deps

            start_time = time.time()
            result_sequential = agent_sequential.run()
            sequential_time = time.time() - start_time

        # Verify both completed successfully
        assert result_parallel.status == AgentRunStatus.SUCCESS
        assert result_sequential.status == AgentRunStatus.SUCCESS
        assert result_parallel.completed_tasks == 6
        assert result_sequential.completed_tasks == 6

        # Parallel execution should be significantly faster
        # Independent tasks: 5 * 0.1s = 0.5s sequential, ~0.1s parallel
        # Final task: 0.05s in both cases
        # Expected: sequential ~0.55s, parallel ~0.15s

        speedup = sequential_time / parallel_time
        assert speedup > 2.0, (
            f"Parallel execution not sufficiently faster: {speedup:.2f}x speedup"
        )
        assert parallel_time < 0.3, f"Parallel execution too slow: {parallel_time:.3f}s"

    def test_execution_plan_optimization(self) -> None:
        """TaskExecutionPlan optimizes for maximum parallelism."""

        # Create complex dependency graph
        # Level 1: A, B (independent)
        # Level 2: C (depends on A), D (depends on B), E (independent)
        # Level 3: F (depends on C, D), G (depends on E)
        # Level 4: H (depends on F, G)

        task_a = PerformanceTestTask("A", {})
        task_b = PerformanceTestTask("B", {})
        task_c = PerformanceTestTask(
            "C", {}, [TaskDependency("a_data", "A.result", DependencyType.REQUIRED)]
        )
        task_d = PerformanceTestTask(
            "D", {}, [TaskDependency("b_data", "B.result", DependencyType.REQUIRED)]
        )
        task_e = PerformanceTestTask("E", {})
        task_f = PerformanceTestTask(
            "F",
            {},
            [
                TaskDependency("c_data", "C.result", DependencyType.REQUIRED),
                TaskDependency("d_data", "D.result", DependencyType.REQUIRED),
            ],
        )
        task_g = PerformanceTestTask(
            "G", {}, [TaskDependency("e_data", "E.result", DependencyType.REQUIRED)]
        )
        task_h = PerformanceTestTask(
            "H",
            {},
            [
                TaskDependency("f_data", "F.result", DependencyType.REQUIRED),
                TaskDependency("g_data", "G.result", DependencyType.REQUIRED),
            ],
        )

        tasks: List[BaseTask] = [
            task_a,
            task_b,
            task_c,
            task_d,
            task_e,
            task_f,
            task_g,
            task_h,
        ]

        # Measure execution plan creation time
        start_time = time.time()
        plan = TaskExecutionPlan(tasks)
        plan_time = time.time() - start_time

        # Plan creation should be fast
        assert plan_time < 0.1, f"Execution plan creation too slow: {plan_time:.4f}s"

        # Verify optimal parallelism
        assert plan.total_phases() == 4  # Should have 4 phases for optimal parallelism

        # Phase 1: A, B, E (3 tasks in parallel)
        assert len(plan.phases[0]) == 3
        phase_1_names = {task.name for task in plan.phases[0]}
        assert phase_1_names == {"A", "B", "E"}

        # Phase 2: C, D, G (3 tasks in parallel)
        assert len(plan.phases[1]) == 3
        phase_2_names = {task.name for task in plan.phases[1]}
        assert phase_2_names == {"C", "D", "G"}

        # Phase 3: F (1 task)
        assert len(plan.phases[2]) == 1
        assert plan.phases[2][0].name == "F"

        # Phase 4: H (1 task)
        assert len(plan.phases[3]) == 1
        assert plan.phases[3][0].name == "H"

    def test_large_scale_agent_performance(self) -> None:
        """Framework performance with large number of tasks."""

        class LargeScaleAgent(PipelineAgent):
            @override
            def setup_tasks(self) -> List[BaseTask]:
                tasks: List[BaseTask] = []
                num_chains = 10
                chain_length = 20

                # Create multiple independent chains of dependencies
                for chain_id in range(num_chains):
                    for pos in range(chain_length):
                        task_name = f"chain_{chain_id}_task_{pos}"
                        dependencies: List[TaskDependency] = []

                        if pos > 0:
                            # Depend on previous task in chain
                            prev_task = f"chain_{chain_id}_task_{pos - 1}"
                            dependencies.append(
                                TaskDependency(
                                    "prev_data",
                                    f"{prev_task}.result",
                                    DependencyType.REQUIRED,
                                )
                            )

                        task = PerformanceTestTask(
                            task_name, {}, dependencies, execution_time=0.001
                        )
                        tasks.append(task)

                # Add final aggregation task
                final_deps: List[TaskDependency] = []
                for chain_id in range(num_chains):
                    final_task_name = f"chain_{chain_id}_task_{chain_length - 1}"
                    final_deps.append(
                        TaskDependency(
                            f"chain_{chain_id}",
                            f"{final_task_name}.result",
                            DependencyType.REQUIRED,
                        )
                    )

                final_task = PerformanceTestTask(
                    "final_aggregate", {}, final_deps, execution_time=0.01
                )
                tasks.append(final_task)

                return tasks

        agent = LargeScaleAgent("large_scale", {"enable_parallel": True})
        total_tasks = 10 * 20 + 1  # 201 tasks

        # Mock dependency resolution for performance by directly setting the instance
        mock_resolver = Mock()

        def fast_resolve(task: Any) -> Dict[str, str]:
            return {"mock_data": "resolved"}

        mock_resolver.resolve_dependencies.side_effect = fast_resolve
        agent.dependency_resolver = mock_resolver

        # Measure total execution time
        start_time = time.time()
        result = agent.run()
        execution_time = time.time() - start_time

        # Verify successful completion
        assert result.status == AgentRunStatus.SUCCESS
        assert result.total_tasks == total_tasks
        assert result.completed_tasks == total_tasks

        # Execution should complete efficiently
        # With parallelism, should take roughly chain_length * 0.001 + 0.01 = ~0.03s
        assert execution_time < 1.0, (
            f"Large scale execution too slow: {execution_time:.3f}s"
        )

        # Verify dependency resolver was called for each task
        assert mock_resolver.resolve_dependencies.call_count == total_tasks

    @pytest.mark.performance
    def test_memory_usage_with_large_pipelines(self) -> None:
        """Memory usage remains reasonable with large pipelines."""

        class MemoryTestAgent(PipelineAgent):
            @override
            def setup_tasks(self) -> List[BaseTask]:
                # Create tasks that use significant memory
                tasks: List[BaseTask] = []
                for i in range(50):
                    # Each task allocates 1MB of memory
                    task = PerformanceTestTask(
                        f"memory_task_{i}",
                        {},
                        [],
                        execution_time=0.01,
                        memory_usage=1024 * 1024,
                    )
                    tasks.append(task)

                return tasks

        # Monitor memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        agent = MemoryTestAgent("memory_test", {})

        with patch(
            "aipype.task_dependencies.DependencyResolver"
        ) as mock_resolver_class:
            mock_resolver = Mock()
            mock_resolver_class.return_value = mock_resolver
            mock_resolver.resolve_dependencies.return_value = {}

            result = agent.run()

            # Check peak memory usage
            peak_memory = process.memory_info().rss
            memory_increase = peak_memory - initial_memory

        # Clean up
        import gc

        gc.collect()

        final_memory = process.memory_info().rss

        # Verify execution succeeded
        assert result.status == AgentRunStatus.SUCCESS
        assert result.completed_tasks == 50

        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024, (
            f"Memory usage too high: {memory_increase / 1024 / 1024:.2f} MB"
        )

        # Memory should be mostly released after completion
        memory_retained = final_memory - initial_memory
        assert memory_retained < 100 * 1024 * 1024, (
            f"Too much memory retained: {memory_retained / 1024 / 1024:.2f} MB"
        )
