#!/bin/bash
set -e

# Build HTML documentation using Sphinx for all packages in the workspace

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCS_DIR="$PROJECT_ROOT/docs"

# Function to show help
show_help() {
    cat << EOF
Build HTML documentation for aipype packages using Sphinx

Usage: $0 [OPTIONS]

Options:
    --serve             Start local HTTP server after building docs
    --clean             Remove existing docs before building
    --help              Show this help message

Examples:
    $0                              # Build all packages
    $0 --clean --serve             # Clean, build all, and serve
EOF
}

# Default values
SERVE=false
CLEAN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --serve)
            SERVE=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Clean existing docs if requested
if [ "$CLEAN" = true ]; then
    echo "🧹 Cleaning existing documentation..."
    rm -rf "$DOCS_DIR/_build"
fi

# Change to project root
cd "$PROJECT_ROOT"

# Install/update dependencies
echo "📦 Syncing dependencies..."
uv sync --quiet

# Build documentation using Sphinx
echo "🚀 Building documentation with Sphinx..."
if uv run sphinx-build -b html docs "$DOCS_DIR/_build/html" --keep-going; then
    echo "✅ Documentation build successful"
else
    echo "❌ Documentation build failed"
    exit 1
fi

echo "📝 Sphinx documentation generated at docs/_build/html/index.html"

# Serve documentation if requested
if [ "$SERVE" = true ]; then
    echo "🌐 Starting local documentation server..."
    echo "📱 Open http://localhost:8000 in your browser"
    echo "🔄 Press Ctrl+C to stop the server"
    cd "$DOCS_DIR/_build/html"
    python3 -m http.server 8000
fi

echo "🎉 Documentation build complete!"
echo "📁 Documentation available in: $DOCS_DIR/_build/html"
echo "🔗 Open $DOCS_DIR/_build/html/index.html in your browser"