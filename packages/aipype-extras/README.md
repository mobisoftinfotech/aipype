# aipype-extras

Extra tools for the aipype framework.

## Installation

```bash
pip install aipype-extras
```

## Components

### LLM Log Viewer

Web interface for viewing LLM conversation logs.

**Important Security Note**

**This feature is designed for local use only for the developer to check the LLM calls being made from the agent. It should not be exposed as a remote service over public internet.**


```bash
# Start log viewer
python -m aipype_extras.llm_log_viewer

# Custom log file
python -m aipype_extras.llm_log_viewer /path/to/logs.jsonl

# Custom port
python -m aipype_extras.llm_log_viewer --port 8080

# Development mode
python -m aipype_extras.llm_log_viewer --debug
```

## Development

### Requirements
- Python ≥3.12
- aipype (core framework)
- Flask ≥3.0.0
