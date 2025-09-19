"""Integration tests for tutorial agents from aipype-examples.

These tests run the actual tutorial agents to ensure they work correctly.
"""

import os
import importlib
import pytest
from aipype import AgentRunResult


class TestTutorialAgents:
    """Integration tests for tutorial agents."""

    def test_basic_print_agent(self) -> None:
        """Test tutorial 01 - basic print agent executes successfully."""
        # Import the module using importlib due to numeric prefix in filename
        basic_print_module = importlib.import_module(
            "aipype_examples.tutorial.01_basic_print_agent"
        )
        BasicAgent = basic_print_module.BasicAgent

        agent = BasicAgent(name="test-basic", config={"message": "Test message"})
        result = agent.run()

        assert isinstance(result, AgentRunResult)
        assert result.is_success()
        assert result.completed_tasks == 1
        assert result.failed_tasks == 0
        assert result.agent_name == "test-basic"

        # Verify the task result exists
        task_result = agent.context.get_result("print_message")
        assert task_result is not None
        assert task_result["message"] == "Test message"

    @pytest.mark.ollama
    def test_llm_outline_agent(
        self, skip_if_ollama_unavailable, skip_if_test_model_unavailable
    ) -> None:
        """Test tutorial 02 - LLM outline generation."""
        # Import the module using importlib due to numeric prefix in filename
        llm_task_module = importlib.import_module(
            "aipype_examples.tutorial.02_llm_task"
        )
        OutlineAgent = llm_task_module.OutlineAgent

        test_model = os.getenv("INTEGRATION_TEST_MODEL", "gemma3:1b")

        agent = OutlineAgent(
            name="test-outline", config={"topic": "integration testing in Python"}
        )

        # Update the agent's LLM task to use the test model
        agent.setup_tasks()
        for task in agent.tasks:
            if hasattr(task, "config") and "llm_model" in task.config:
                task.config["llm_model"] = test_model

        result = agent.run()

        assert isinstance(result, AgentRunResult)
        assert result.is_success()
        assert result.completed_tasks == 1
        assert result.agent_name == "test-outline"

        # Verify outline was generated
        outline_result = agent.context.get_result("outline_article")
        assert outline_result is not None
        assert "content" in outline_result
        assert len(outline_result["content"]) > 20  # Has some content

    @pytest.mark.ollama
    def test_dependent_tasks_agent(
        self, skip_if_ollama_unavailable, skip_if_test_model_unavailable
    ) -> None:
        """Test tutorial 03 - dependent tasks with outline → article flow."""
        # Import the module using importlib due to numeric prefix in filename
        dependent_tasks_module = importlib.import_module(
            "aipype_examples.tutorial.03_dependent_tasks"
        )
        OutlineToArticleAgent = dependent_tasks_module.OutlineToArticleAgent

        test_model = os.getenv("INTEGRATION_TEST_MODEL", "gemma3:1b")

        agent = OutlineToArticleAgent(
            name="test-dependent", config={"topic": "integration testing fundamentals"}
        )

        # Update the agent's LLM tasks to use the test model
        agent.setup_tasks()
        for task in agent.tasks:
            if hasattr(task, "config") and "llm_model" in task.config:
                task.config["llm_model"] = test_model

        result = agent.run()

        assert isinstance(result, AgentRunResult)
        assert result.is_success()
        assert result.completed_tasks == 2
        assert result.total_phases == 2  # Tasks run in sequence due to dependency
        assert result.agent_name == "test-dependent"

        # Verify both tasks completed
        outline_result = agent.context.get_result("outline_article")
        article_result = agent.context.get_result("write_article")

        assert outline_result is not None
        assert article_result is not None
        assert "content" in outline_result
        assert "content" in article_result
        assert len(outline_result["content"]) > 20
        assert len(article_result["content"]) > 20

    @pytest.mark.ollama
    def test_conditional_task_agent_success_path(
        self, skip_if_ollama_unavailable, skip_if_test_model_unavailable
    ) -> None:
        """Test tutorial 04 - conditional task with condition that passes."""
        # Import the module using importlib due to numeric prefix in filename
        conditional_task_module = importlib.import_module(
            "aipype_examples.tutorial.04_conditional_task"
        )
        QualityCheckerAgent = conditional_task_module.QualityCheckerAgent

        test_model = os.getenv("INTEGRATION_TEST_MODEL", "gemma3:1b")

        agent = QualityCheckerAgent(
            name="test-quality-pass",
            config={
                "topic": "automated testing practices",
                "min_word_count": 10,  # Low threshold to ensure pass
            },
        )

        # Update the agent's LLM task to use the test model
        agent.setup_tasks()
        for task in agent.tasks:
            if hasattr(task, "config") and "llm_model" in task.config:
                task.config["llm_model"] = test_model

        result = agent.run()

        assert isinstance(result, AgentRunResult)
        assert result.is_success()
        assert result.completed_tasks == 2
        assert result.agent_name == "test-quality-pass"

        # Verify outline was generated
        outline_result = agent.context.get_result("generate_outline")
        assert outline_result is not None
        assert "content" in outline_result

        # Verify quality check completed (condition should pass)
        quality_result = agent.context.get_result("quality_check")
        assert quality_result is not None

    @pytest.mark.ollama
    def test_conditional_task_agent_failure_path(
        self, skip_if_ollama_unavailable, skip_if_test_model_unavailable
    ) -> None:
        """Test tutorial 04 - conditional task with condition that fails."""
        # Import the module using importlib due to numeric prefix in filename
        conditional_task_module = importlib.import_module(
            "aipype_examples.tutorial.04_conditional_task"
        )
        QualityCheckerAgent = conditional_task_module.QualityCheckerAgent

        test_model = os.getenv("INTEGRATION_TEST_MODEL", "gemma3:1b")

        agent = QualityCheckerAgent(
            name="test-quality-fail",
            config={
                "topic": "test",
                "min_word_count": 10000,  # Very high threshold to ensure fail
            },
        )

        # Update the agent's LLM task to use the test model
        agent.setup_tasks()
        for task in agent.tasks:
            if hasattr(task, "config") and "llm_model" in task.config:
                task.config["llm_model"] = test_model

        result = agent.run()

        assert isinstance(result, AgentRunResult)
        # Should still complete successfully even if condition fails
        assert result.is_success() or result.is_partial()
        assert result.completed_tasks >= 1  # At least outline task should complete
        assert result.agent_name == "test-quality-fail"

        # Verify outline was generated
        outline_result = agent.context.get_result("generate_outline")
        assert outline_result is not None
        assert "content" in outline_result

        # Quality check task should exist (may have failed condition)
        # Note: quality_result might be None if condition failed and else_function raised exception
