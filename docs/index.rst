aipype Documentation
====================

.. note::
   **Version:** |release|

Modular AI agent framework with pipeline-based task orchestration for automation heavy AI workflows.

aipype is a powerful Python framework for building AI workflows with automatic dependency resolution and parallel execution. It provides a clean interface for orchestrating tasks while handling complex dependency management behind the scenes.

Key Features
============

* **Declarative Syntax**: Use ``@task`` decorators for clean, Pythonic agent definitions
* **Automatic Dependencies**: Dependencies inferred from method parameter names
* **Helper Functions**: ``llm()``, ``search()``, ``mcp_server()`` for intuitive task creation
* **Pipeline System**: Declarative pipeline with automatic dependency resolution
* **Task Context**: Shared data with path-based access ("search_results.data")
* **Template Substitution**: ``${variable}`` syntax in task configurations
* **Multiple Built in Tasks**: LLM, Search, Transform, and Conditional tasks
* **Parallel Execution**: Automatic optimization of task execution order
* **Error Handling**: Graceful error propagation with TaskResult pattern

Quick Start
===========

Install aipype:

.. code-block:: bash

   pip install aipype

Declarative Syntax (Recommended)
--------------------------------

Create agents using the ``@task`` decorator for clean, Pythonic code:

.. code-block:: python

   from aipype import PipelineAgent, task, llm, search, Depends
   from typing import Annotated

   class ResearchAgent(PipelineAgent):

       @task
       def find_sources(self) -> dict:
           """Search for articles - no dependencies, runs first."""
           return search(self.config["topic"], max_results=5)

       @task
       def analyze(self, find_sources: dict) -> str:
           """Analyze sources - find_sources parameter auto-injected."""
           return llm(
               prompt=f"Analyze these sources: {find_sources}",
               model="gpt-4o",
               temperature=0.3
           )

       @task
       def write_summary(
           self,
           content: Annotated[str, Depends("analyze.content")]
       ) -> str:
           """Use Depends() for explicit field extraction."""
           return llm(f"Write a summary based on: {content}", model="gpt-4o")

   # Usage
   agent = ResearchAgent("research", {"topic": "AI trends"})
   result = agent.run()
   agent.display_results()

Key Concepts:

* ``@task`` decorator marks methods as pipeline tasks
* Parameter names matching task names create automatic dependencies
* ``Annotated[T, Depends("task.field")]`` extracts specific fields from task output
* ``self.config`` provides access to agent configuration
* Tasks with no dependencies run first (topologically sorted)

Legacy Syntax
-------------

For backwards compatibility or complex scenarios:

.. code-block:: python

   from aipype import BasePipelineAgent, LLMTask, SearchTask, TaskDependency, DependencyType

   class ArticleWriterAgent(BasePipelineAgent):
       def setup_tasks(self):
           return [
               SearchTask("search", {"query": "${topic}", "max_results": 5},
                         [TaskDependency("topic", "user_input.topic", DependencyType.REQUIRED)]),
               LLMTask("write", {"prompt": "Write about ${topic}: ${results}"},
                      [TaskDependency("topic", "user_input.topic", DependencyType.REQUIRED),
                       TaskDependency("results", "search.results", DependencyType.REQUIRED)])
           ]

   agent = ArticleWriterAgent("writer", {})
   result = agent.run({"topic": "AI trends"})

Packages
--------

.. toctree::
   :maxdepth: 2

   api/aipype
   api/aipype_extras
   api/aipype_g

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`