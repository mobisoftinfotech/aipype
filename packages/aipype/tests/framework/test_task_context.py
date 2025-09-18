"""Tests for simplified TaskContext system - shared data store for inter-task communication."""

from typing import List, Any
from aipype import TaskContext


class TestTaskContext:
    """Test suite for simplified TaskContext functionality."""

    def test_context_stores_task_results(self) -> None:
        """Tasks can store results in shared context."""
        context = TaskContext()

        # Store simple result
        result = {"status": "success", "data": "some text data"}
        context.store_result("task1", result)

        # Verify storage
        assert context.has_result("task1")
        stored = context.get_result("task1")
        assert stored == result
        assert stored is not None
        assert stored["status"] == "success"
        assert stored["data"] == "some text data"

    def test_context_stores_text_based_results(self) -> None:
        """Context can store and retrieve text-based data structures."""
        context = TaskContext()

        text_result = {
            "query": "machine learning trends",
            "results": ["Article 1", "Article 2", "Article 3"],
            "content": "This is the main content text...",
            "metadata": "Search completed in 1.5s",
        }

        context.store_result("search_task", text_result)

        # Verify text structure is preserved
        stored = context.get_result("search_task")
        assert stored is not None
        assert stored["query"] == "machine learning trends"
        assert stored["content"] == "This is the main content text..."
        assert len(stored["results"]) == 3
        assert stored["results"][0] == "Article 1"

    def test_context_retrieves_previous_results(self) -> None:
        """Tasks can access results from previous tasks."""
        context = TaskContext()

        # Store results from multiple tasks
        context.store_result("task1", {"output": "data1"})
        context.store_result("task2", {"output": "data2"})
        context.store_result("task3", {"output": "data3"})

        # Verify access to any previous result
        task1_result = context.get_result("task1")
        task2_result = context.get_result("task2")
        task3_result = context.get_result("task3")
        assert task1_result is not None and task1_result["output"] == "data1"
        assert task2_result is not None and task2_result["output"] == "data2"
        assert task3_result is not None and task3_result["output"] == "data3"

    def test_context_supports_simple_path_access(self) -> None:
        """Context supports simple dot notation paths like 'task.field'."""
        context = TaskContext()

        # Store nested text data
        search_result = {
            "query": "AI trends",
            "content": "This is the article content...",
            "metadata": "Search completed successfully",
            "count": "5 results found",
        }
        context.store_result("search", search_result)

        # Test simple path access
        query = context.get_path_value("search.query")
        assert query == "AI trends"

        content = context.get_path_value("search.content")
        assert content == "This is the article content..."

        metadata = context.get_path_value("search.metadata")
        assert metadata == "Search completed successfully"

        count = context.get_path_value("search.count")
        assert count == "5 results found"

    def test_context_handles_missing_dependencies(self) -> None:
        """Context gracefully handles missing or invalid paths."""
        context = TaskContext()

        # Test missing task
        assert context.get_result("nonexistent_task") is None
        assert not context.has_result("nonexistent_task")

        # Store some data for path testing
        context.store_result("task1", {"data": "some value", "nested": "text"})

        # Test valid paths
        assert context.get_path_value("task1.data") == "some value"
        assert context.get_path_value("task1.nested") == "text"

        # Test invalid paths
        assert context.get_path_value("task1.nonexistent") is None
        assert context.get_path_value("nonexistent_task.data") is None

        # Test invalid path formats
        assert context.get_path_value("invalid_path") is None
        assert context.get_path_value("") is None
        assert context.get_path_value("task1") is None  # Missing field

    def test_context_maintains_execution_history(self) -> None:
        """Context tracks which tasks ran and their status."""
        context = TaskContext()

        # Record task execution
        context.record_task_started("task1")
        context.record_task_completed("task1", {"result": "success"})

        context.record_task_started("task2")
        context.record_task_failed("task2", "Some error occurred")

        context.record_task_started("task3")
        context.record_task_completed("task3", {"result": "done"})

        # Verify execution history
        history = context.get_execution_history()
        assert len(history) == 3

        # Check task1
        task1_history = next(h for h in history if h["task_name"] == "task1")
        assert task1_history["status"] == "completed"
        assert task1_history["result"]["result"] == "success"
        assert task1_history["error"] is None

        # Check task2
        task2_history = next(h for h in history if h["task_name"] == "task2")
        assert task2_history["status"] == "failed"
        assert task2_history["error"] == "Some error occurred"
        assert task2_history["result"] is None

        # Check task3
        task3_history = next(h for h in history if h["task_name"] == "task3")
        assert task3_history["status"] == "completed"

    def test_context_get_completed_tasks(self) -> None:
        """Context can filter for completed tasks only."""
        context = TaskContext()

        context.record_task_started("task1")
        context.record_task_completed("task1", {"data": "result1"})

        context.record_task_started("task2")
        context.record_task_failed("task2", "Error")

        context.record_task_started("task3")
        context.record_task_completed("task3", {"data": "result3"})

        completed = context.get_completed_tasks()
        assert len(completed) == 2
        assert "task1" in completed
        assert "task3" in completed
        assert "task2" not in completed

    def test_context_get_failed_tasks(self) -> None:
        """Context can filter for failed tasks only."""
        context = TaskContext()

        context.record_task_started("task1")
        context.record_task_completed("task1", {"data": "result1"})

        context.record_task_started("task2")
        context.record_task_failed("task2", "Error 1")

        context.record_task_started("task3")
        context.record_task_failed("task3", "Error 2")

        failed = context.get_failed_tasks()
        assert len(failed) == 2
        assert "task2" in failed
        assert "task3" in failed
        assert "task1" not in failed

    def test_context_clear_and_reset(self) -> None:
        """Context can be cleared and reset."""
        context = TaskContext()

        # Add some data
        context.store_result("task1", {"data": "value"})
        context.record_task_started("task1")
        context.record_task_completed("task1", {"data": "value"})

        assert context.has_result("task1")
        assert len(context.get_execution_history()) == 1

        # Clear context
        context.clear()

        assert not context.has_result("task1")
        assert len(context.get_execution_history()) == 0
        assert context.get_result("task1") is None

    def test_context_data_aliases(self) -> None:
        """Context provides convenient aliases for data operations."""
        context = TaskContext()

        # Test set_data / get_data aliases
        data = {"message": "Hello World", "count": "42"}
        context.set_data("test_task", data)

        retrieved = context.get_data("test_task")
        assert retrieved == data
        assert retrieved is not None
        assert retrieved["message"] == "Hello World"

        # Verify aliases work with main methods
        assert context.has_result("test_task")
        assert context.get_result("test_task") == data

    def test_context_string_representation(self) -> None:
        """Context provides useful string representation."""
        context = TaskContext()

        # Empty context
        context_str = str(context)
        assert "TaskContext" in context_str
        assert "results=0" in context_str
        assert "completed=0" in context_str
        assert "failed=0" in context_str

        # Add some data
        context.store_result("task1", {"output": "data"})
        context.record_task_started("task1")
        context.record_task_completed("task1", {"output": "data"})

        context.record_task_started("task2")
        context.record_task_failed("task2", "Error")

        context_str = str(context)
        assert "results=1" in context_str
        assert "completed=1" in context_str
        assert "failed=1" in context_str

    def test_context_thread_safety(self) -> None:
        """Context operations are thread-safe for concurrent access."""
        import threading
        import time

        context = TaskContext()
        results: List[int] = []
        errors: List[str] = []

        def worker(task_id: int) -> None:
            try:
                # Each thread stores and retrieves data
                data = {"task_id": f"task_{task_id}", "timestamp": str(time.time())}
                context.store_result(f"task_{task_id}", data)

                # Small delay to increase chance of race conditions
                time.sleep(0.001)

                # Retrieve and verify
                retrieved = context.get_result(f"task_{task_id}")
                if retrieved and retrieved["task_id"] == f"task_{task_id}":
                    results.append(task_id)
            except Exception as e:
                errors.append(str(e))

        # Start multiple threads
        threads: List[Any] = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Verify all operations succeeded
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 10
        assert sorted(results) == list(range(10))

    def test_context_text_based_workflow(self) -> None:
        """Test a complete text-based workflow."""
        context = TaskContext()

        # Simulate a text processing pipeline
        # Task 1: Input processing
        input_data = {
            "original_text": "The quick brown fox jumps over the lazy dog",
            "language": "en",
            "word_count": "9",
        }
        context.store_result("input_processor", input_data)

        # Task 2: Text analysis
        analysis_data = {
            "sentiment": "neutral",
            "keywords": "fox, dog, jump",
            "summary": "A sentence about animals",
        }
        context.store_result("text_analyzer", analysis_data)

        # Task 3: Output formatting (uses previous results)
        original = context.get_path_value("input_processor.original_text")
        sentiment = context.get_path_value("text_analyzer.sentiment")
        keywords = context.get_path_value("text_analyzer.keywords")

        output_data = {
            "formatted_result": f"Text: {original}\nSentiment: {sentiment}\nKeywords: {keywords}",
            "final_status": "processing_complete",
        }
        context.store_result("output_formatter", output_data)

        # Verify the workflow
        assert original == "The quick brown fox jumps over the lazy dog"
        assert sentiment == "neutral"
        assert keywords == "fox, dog, jump"

        final_result = context.get_result("output_formatter")
        assert final_result is not None
        assert "The quick brown fox" in final_result["formatted_result"]
        assert "neutral" in final_result["formatted_result"]
        assert final_result["final_status"] == "processing_complete"

    def test_context_array_path_support(self) -> None:
        """Test that basic array path access works for compatibility."""
        context = TaskContext()

        # Store data with arrays (common in search results)
        search_result = {
            "query": "test query",
            "results": [
                {"url": "http://example1.com", "title": "Article 1"},
                {"url": "http://example2.com", "title": "Article 2"},
                {"url": "http://example3.com", "title": "Article 3"},
            ],
        }
        context.store_result("search", search_result)

        # Test array extraction (results[].url)
        urls = context.get_path_value("search.results[].url")
        assert urls == [
            "http://example1.com",
            "http://example2.com",
            "http://example3.com",
        ]

        # Test array indexing (results[0].title)
        first_title = context.get_path_value("search.results[0].title")
        assert first_title == "Article 1"

        # Test second item
        second_url = context.get_path_value("search.results[1].url")
        assert second_url == "http://example2.com"

        # Test simple field access still works
        query = context.get_path_value("search.query")
        assert query == "test query"
