#!/bin/bash

# LLM Log Viewer Runner Script
# This script sets up the environment and runs the LLM log viewer
#
# Recommended usage: python -m framework.extras.llm_log_viewer
# This script is provided as an alternative method.

echo "🔍 Starting LLM Log Viewer..."
echo "💡 Tip: You can also run this with: python -m framework.extras.llm_log_viewer"

# Check if we're in the right directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Set default log file path if not already set
if [ -z "$MI_AGENT_LLM_LOGS_FILE" ]; then
    DEFAULT_LOG_PATH="$(dirname "$SCRIPT_DIR")/../../../output/llm_logs.jsonl"
    export MI_AGENT_LLM_LOGS_FILE="$DEFAULT_LOG_PATH"
    echo "📁 Using default log file: $MI_AGENT_LLM_LOGS_FILE"
else
    echo "📁 Using log file: $MI_AGENT_LLM_LOGS_FILE"
fi

# Check if log file exists
if [ ! -f "$MI_AGENT_LLM_LOGS_FILE" ]; then
    echo "⚠️  Warning: Log file not found at $MI_AGENT_LLM_LOGS_FILE"
    echo "   The viewer will still start, but no logs will be displayed until the file is created."
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python is not installed or not in PATH"
    exit 1
fi

# Check if Flask is available
if ! python -c "import flask" &> /dev/null; then
    echo "❌ Error: Flask is not installed"
    echo "   Please install Flask: pip install flask"
    echo "   Or run: uv sync (to install all project dependencies)"
    exit 1
fi

echo "🚀 Launching LLM Log Viewer..."
echo "   Press Ctrl+C to stop the server"
echo ""

# Run the log viewer
python llm_log_viewer.py