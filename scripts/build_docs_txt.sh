#!/bin/bash
set -e

# Build single-file text documentation (llms.txt) using Sphinx for all packages

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCS_DIR="$PROJECT_ROOT/docs"
BUILD_DIR="$DOCS_DIR/_build/text"

# Function to show help
show_help() {
    cat << EOF
Build single-file text documentation (llms.txt) for aipype packages using Sphinx

Usage: $0 [OPTIONS]

Options:
    --output FILE       Output file path (default: llms.txt in project root)
    --clean             Remove existing text build before building
    --help              Show this help message

Examples:
    $0                              # Build llms.txt in project root
    $0 --output docs/llms.txt       # Build to custom location
    $0 --clean                      # Clean build first
EOF
}

# Default values
OUTPUT_FILE="$PROJECT_ROOT/llms.txt"
CLEAN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output)
            OUTPUT_FILE="$2"
            shift 2
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

# Clean existing text build if requested
if [ "$CLEAN" = true ]; then
    echo "🧹 Cleaning existing text documentation..."
    rm -rf "$BUILD_DIR"
fi

# Change to project root
cd "$PROJECT_ROOT"

# Install/update dependencies
echo "📦 Syncing dependencies..."
uv sync --quiet

# Build text documentation using Sphinx
echo "🚀 Building text documentation with Sphinx..."
if uv run sphinx-build -b text docs "$BUILD_DIR" --keep-going; then
    echo "✅ Text documentation build successful"
else
    echo "❌ Text documentation build failed"
    exit 1
fi

# Concatenate all text files into single llms.txt
echo "📝 Combining documentation into single file..."

# Start with a header
cat > "$OUTPUT_FILE" << 'EOF'
================================================================================
aipype Documentation
================================================================================

This file contains the complete documentation for the aipype framework and
all its packages, generated from Sphinx documentation sources.

Last generated: $(date)

================================================================================

EOF

# Add index/overview
if [ -f "$BUILD_DIR/index.txt" ]; then
    cat "$BUILD_DIR/index.txt" >> "$OUTPUT_FILE"
    echo -e "\n\n" >> "$OUTPUT_FILE"
fi

# Add separator before API docs
cat >> "$OUTPUT_FILE" << 'EOF'
================================================================================
API DOCUMENTATION
================================================================================

EOF

# Add each API documentation file
for api_file in "$BUILD_DIR/api"/*.txt; do
    if [ -f "$api_file" ]; then
        # Add separator with filename
        echo "================================================================================" >> "$OUTPUT_FILE"
        echo "$(basename "$api_file" .txt)" >> "$OUTPUT_FILE"
        echo "================================================================================" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"

        cat "$api_file" >> "$OUTPUT_FILE"
        echo -e "\n\n" >> "$OUTPUT_FILE"
    fi
done

echo "✅ Documentation combined successfully"
echo "📁 Output file: $OUTPUT_FILE"
echo "📊 File size: $(du -h "$OUTPUT_FILE" | cut -f1)"

echo "🎉 llms.txt generation complete!"
