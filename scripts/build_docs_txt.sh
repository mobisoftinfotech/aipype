#!/bin/bash
set -e

# Build single-file text documentation (llms.txt) using Sphinx for specific packages

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCS_DIR="$PROJECT_ROOT/docs"
BUILD_DIR="$DOCS_DIR/_build/text"
OUTPUT_DIR="$DOCS_DIR/_build"

# Function to show help
show_help() {
    cat << EOF
Build single-file text documentation (llms.txt) for specific aipype packages using Sphinx

Usage: $0 [OPTIONS]

Options:
    --packages LIST     Comma-separated list of packages (default: aipype,aipype-g)
                        Available: aipype, aipype-extras, aipype-g
    --clean             Remove existing text build before building
    --help              Show this help message

Output:
    Files are created in docs/_build/ directory:
    - llms-aipype.txt
    - llms-aipype-g.txt

Examples:
    $0                                    # Build llms files in docs/_build/
    $0 --packages aipype                  # Build only llms-aipype.txt
    $0 --packages aipype,aipype-extras    # Build specified packages
    $0 --clean                            # Clean build first
EOF
}

# Default values
PACKAGES="aipype,aipype-g"
CLEAN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --packages)
            PACKAGES="$2"
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

# Convert packages string to array
IFS=',' read -ra PACKAGE_ARRAY <<< "$PACKAGES"

echo "📝 Generating separate llms.txt files for packages: ${PACKAGES}..."

# Process each package
for package in "${PACKAGE_ARRAY[@]}"; do
    # Trim whitespace
    package=$(echo "$package" | xargs)

    # Convert package name: aipype-g -> aipype_g for file lookup
    package_file="${package//-/_}"
    api_file="$BUILD_DIR/api/${package_file}.txt"
    output_file="$OUTPUT_DIR/llms-${package}.txt"

    if [ ! -f "$api_file" ]; then
        echo "⚠️  Warning: API documentation not found for package '$package' at $api_file"
        echo "   Skipping..."
        continue
    fi

    echo "  📄 Creating $output_file..."

    # Start with a header
    cat > "$output_file" << EOF
================================================================================
${package} Documentation
================================================================================

This file contains the documentation for the ${package} package,
generated from Sphinx documentation sources.

Last generated: $(date)

================================================================================

EOF

    # Add index/overview (optional - comment out if too large)
    if [ -f "$BUILD_DIR/index.txt" ]; then
        cat "$BUILD_DIR/index.txt" >> "$output_file"
        echo -e "\n\n" >> "$output_file"
    fi

    # Add separator before API docs
    cat >> "$output_file" << 'EOF'
================================================================================
API DOCUMENTATION
================================================================================

EOF

    # Add the specific package API documentation
    cat "$api_file" >> "$output_file"

    echo "  ✅ Created $output_file ($(du -h "$output_file" | cut -f1))"
done

echo ""
echo "🎉 llms.txt generation complete!"
echo "📁 Output directory: $OUTPUT_DIR"
echo "📦 Packages processed: ${PACKAGES}"
