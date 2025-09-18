"""
Basic print agent tutorial
-------------------------

This minimal example demonstrates core framework concepts from the local
`aipype` package:

- BaseTask: the unit of work. You subclass it and implement `run()` to perform
  side effects and return a `TaskResult` (success/failure with structured data).
- PipelineAgent: orchestrates a pipeline (ordered list) of tasks. You subclass
  it and implement `setup_tasks()` to declare which tasks should run.
- TaskResult: a typed container for outcomes, making task outputs consistent
  and easy to pass to downstream tasks.

It also shows how configuration flows from an agent into its tasks via the
`config` dictionaries.
"""

from typing import List
from typing import override
from aipype import PipelineAgent, BaseTask, TaskResult
from aipype import print_header


DEFAULT_MESSAGE = "Hello from my first agent!"


class PrintMessageTask(BaseTask):
    """A simple task that reads a message from config and prints it.

    Key framework concepts:
    - Inherit from `BaseTask` to implement a unit of work.
    - Implement `run()` to perform the work and return a `TaskResult`.
    - Access task-specific configuration via `self.config`.
    """

    @override
    def run(self) -> TaskResult:
        # Pull input from the task's config. Agents typically pass this down.
        message = self.config.get("message", DEFAULT_MESSAGE)

        # Perform the side-effect this task is responsible for.
        print(message)

        # Return a structured outcome indicating success. Downstream tasks can
        # consume this `data` if needed.
        return TaskResult.success(data={"message": message})


class BasicAgent(PipelineAgent):
    """An agent that orchestrates one or more tasks as a pipeline.

    Key framework concepts:
    - Inherit from `PipelineAgent` to manage task orchestration and lifecycle.
    - Implement `setup_tasks()` to declare an ordered list of tasks.
    - Agent-level `config` can be used to parameterize tasks.
    """

    @override
    def setup_tasks(self) -> List[BaseTask]:
        # Read input from the agent's config and pass it to the task(s).
        message = self.config.get("message", DEFAULT_MESSAGE)

        # Return the pipeline: an ordered list of tasks to run.
        return [PrintMessageTask(name="print_message", config={"message": message})]


if __name__ == "__main__":
    print_header("BASIC PRINT AGENT TUTORIAL")

    # Instantiate the agent with a name and configuration.
    agent = BasicAgent(name="basic-agent", config={"message": DEFAULT_MESSAGE})

    # Run executes the pipeline by calling `setup_tasks()` and running tasks in
    # order, collecting `TaskResult`s along the way.
    agent.run()
    agent.display_results()
