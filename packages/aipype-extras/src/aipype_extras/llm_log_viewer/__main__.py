"""Entry point for running LLM Log Viewer as a module."""

from .llm_log_viewer import main as _main


def main() -> None:
    """CLI entry point for aipype-log-viewer command."""
    _main()


if __name__ == "__main__":
    main()
