"""
Basic print agent tutorial
-------------------------

This minimal example demonstrates core framework concepts from the local
`aipype` package using the recommended declarative syntax:

- PipelineAgent: The primary agent class for building AI pipelines.
- @task decorator: Marks methods as pipeline tasks with automatic dependency inference.
- Return values: Tasks return dicts or primitives (auto-wrapped in TaskResult).

Key concepts:
- Use `@task` decorator to mark methods as pipeline tasks
- Access agent configuration via `self.config`
- Return dicts or primitive values (automatically wrapped in TaskResult)
- Dependencies are inferred from parameter names matching other task names
"""

from aipype import PipelineAgent, task
from aipype import print_header


DEFAULT_MESSAGE = "Hello from my first agent!"


class BasicAgent(PipelineAgent):
    """An agent that prints a message using the declarative @task syntax.

    Key framework concepts:
    - Inherit from `PipelineAgent` for declarative task definition
    - Use `@task` decorator to mark methods as pipeline tasks
    - Access configuration via `self.config`
    - Return dicts or primitives (auto-wrapped in TaskResult)
    """

    @task
    def print_message(self) -> dict[str, str]:
        """A simple task that reads a message from config and prints it.

        This task has no dependencies (no parameters matching other task names),
        so it runs first in the pipeline.

        Returns:
            dict: Contains the message that was printed
        """
        # Pull input from the agent's config
        message = self.config.get("message", DEFAULT_MESSAGE)

        # Perform the side-effect this task is responsible for
        print(message)

        # Return a dict - automatically wrapped in TaskResult.success()
        return {"message": message}


if __name__ == "__main__":
    print_header("BASIC PRINT AGENT TUTORIAL")

    # Instantiate the agent with a name and configuration.
    agent = BasicAgent(name="basic-agent", config={"message": DEFAULT_MESSAGE})

    # Run discovers @task methods, builds the execution plan, and runs tasks
    # in dependency order, collecting results along the way.
    agent.run()
    agent.display_results()
