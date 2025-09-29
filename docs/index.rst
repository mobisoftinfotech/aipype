aipype Documentation
====================

Modular AI agent framework with pipeline-based task orchestration for automation heavy AI workflows.

aipype is a powerful Python framework for building AI workflows with automatic dependency resolution and parallel execution. It provides a clean interface for orchestrating tasks while handling complex dependency management behind the scenes.

Key Features
============

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

Create a simple pipeline:

.. code-block:: python

   from aipype import PipelineAgent, LLMTask, SearchTask, TaskDependency, DependencyType

   class ArticleWriterAgent(PipelineAgent):
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