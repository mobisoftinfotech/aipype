#!/usr/bin/env python3
"""LLM Log Viewer - A browser-based tool for reviewing LLM call logs."""

import argparse
import json
import os
import webbrowser
from typing import Any, Dict, List, Optional
from flask import Flask, jsonify, render_template, request, Response
from dataclasses import dataclass


@dataclass
class LogEntry:
    """Represents a single LLM log entry."""

    timestamp: str
    agent_name: Optional[str]
    task_name: str
    provider: str
    model: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    raw_line: str
    index: int


class LLMLogReader:
    """Handles reading and parsing LLM log files."""

    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path

    def read_logs(
        self, start_index: int = 0, count: int = 50, reverse: bool = True
    ) -> List[LogEntry]:
        """Read log entries from file."""
        if not os.path.exists(self.log_file_path):
            return []

        logs: List[LogEntry] = []

        try:
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Parse all lines first
            all_entries: List[LogEntry] = []
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                try:
                    log_data = json.loads(line)
                    entry = LogEntry(
                        timestamp=log_data.get("timestamp", ""),
                        agent_name=log_data.get("agent_name"),
                        task_name=log_data.get("task_name", ""),
                        provider=log_data.get("provider", ""),
                        model=log_data.get("model", ""),
                        input=log_data.get("input", {}),
                        output=log_data.get("output", {}),
                        raw_line=line,
                        index=i,
                    )
                    all_entries.append(entry)
                except json.JSONDecodeError:
                    continue

            # Apply filtering and pagination
            if reverse:
                all_entries.reverse()

            # Get slice based on start_index and count
            end_index = start_index + count
            logs = all_entries[start_index:end_index]

        except Exception as e:
            print(f"Error reading log file: {e}")

        return logs

    def get_total_log_count(self) -> int:
        """Get total number of log entries."""
        if not os.path.exists(self.log_file_path):
            return 0

        try:
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return 0


def create_app(log_reader: LLMLogReader) -> Flask:
    """Create Flask application."""
    app = Flask(__name__)

    # Add security headers for Chrome compatibility
    @app.after_request
    def after_request(response: Response) -> Response:
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = (
            "GET, POST, PUT, DELETE, OPTIONS"
        )
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        return response

    @app.route("/")
    def index() -> str:
        """Serve the main page."""
        return render_template("index.html")

    @app.route("/api/logs")
    def get_logs() -> Any:
        """API endpoint to get logs."""
        offset = int(request.args.get("offset", 0))
        count = int(request.args.get("count", 50))

        logs = log_reader.read_logs(start_index=offset, count=count, reverse=True)
        total = log_reader.get_total_log_count()

        # Convert LogEntry objects to dictionaries
        log_dicts: List[Dict[str, Any]] = []
        for log in logs:
            log_dicts.append(
                {
                    "timestamp": log.timestamp,
                    "agent_name": log.agent_name,
                    "task_name": log.task_name,
                    "provider": log.provider,
                    "model": log.model,
                    "input": log.input,
                    "output": log.output,
                    "index": log.index,
                }
            )

        return jsonify(
            {
                "logs": log_dicts,
                "total": total,
                "offset": offset,
                "count": len(log_dicts),
            }
        )

    return app


def main() -> None:
    """Main entry point for the LLM log viewer."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="LLM Log Viewer - A browser-based tool for reviewing LLM call logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m framework.extras.llm_log_viewer
  python -m framework.extras.llm_log_viewer /path/to/logs.jsonl
  python -m framework.extras.llm_log_viewer --port 8080 /custom/logs.jsonl
        """,
    )

    parser.add_argument(
        "log_file",
        nargs="?",  # Optional positional argument
        help="Path to the JSONL log file (default: output/llm_logs.jsonl or MI_AGENT_LLM_LOGS_FILE env var)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the web server on (default: 5000)",
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the web server to (default: 127.0.0.1)",
    )

    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the browser",
    )

    args = parser.parse_args()

    # Determine log file path
    if args.log_file:
        log_file_path = args.log_file
    else:
        # Fall back to environment variable or default
        log_file_path = os.environ.get(
            "MI_AGENT_LLM_LOGS_FILE",
            "output/llm_logs.jsonl",  # Default relative to project root
        )

    print("LLM Log Viewer")
    print(f"Log file: {log_file_path}")

    if not os.path.exists(log_file_path):
        print(f"Warning: Log file not found at {log_file_path}")
        print(
            "The viewer will still start, but no logs will be displayed until the file is created."
        )

    # Create log reader
    log_reader = LLMLogReader(log_file_path)

    # Create Flask app
    app = create_app(log_reader)

    # Start server and open browser
    port = args.port
    host = args.host
    url = f"http://{host}:{port}"

    print(f"Starting LLM Log Viewer on {url}")
    print("Press Ctrl+C to stop the server")
    print("")

    if host == "localhost":
        print("If you experience issues with Chrome, try:")
        print(f"  - Opening {url.replace('localhost', '127.0.0.1')} manually")
        print("  - Using Safari or Firefox instead")
        print("  - Checking Chrome's localhost blocking settings")

    # Open browser unless disabled
    if not args.no_browser:
        webbrowser.open(url)
    else:
        print(f"Browser auto-open disabled. Navigate to {url} manually.")

    try:
        app.run(host=host, port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down LLM Log Viewer...")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\nError: Port {port} is already in use.")
            print(
                f"Try using a different port: python -m framework.extras.llm_log_viewer --port {port + 1} {log_file_path if args.log_file else ''}"
            )
        else:
            print(f"\nError starting server: {e}")


if __name__ == "__main__":
    main()
