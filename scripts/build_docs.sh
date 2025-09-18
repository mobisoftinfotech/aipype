#!/bin/bash
set -e

# Build HTML documentation using pdoc3 for all packages in the workspace

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCS_DIR="$PROJECT_ROOT/docs"

# Function to show help
show_help() {
    cat << EOF
Build HTML documentation for aipype packages

Usage: $0 [OPTIONS]

Options:
    --package NAME      Build docs for specific package only (aipype, aipype-extras, aipype-g)
    --serve             Start local HTTP server after building docs
    --clean             Remove existing docs before building
    --help              Show this help message

Examples:
    $0                              # Build all packages
    $0 --package aipype            # Build only aipype package
    $0 --clean --serve             # Clean, build all, and serve
EOF
}

# Default values
SPECIFIC_PACKAGE=""
SERVE=false
CLEAN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --package)
            SPECIFIC_PACKAGE="$2"
            shift 2
            ;;
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
    rm -rf "$DOCS_DIR"
    mkdir -p "$DOCS_DIR"
fi

# Change to project root
cd "$PROJECT_ROOT"

# Function to build documentation for a package
build_package_docs() {
    local package_name="$1"
    local src_path="$2"
    local module_name="$3"

    echo "📚 Building documentation for $package_name..."

    if [ ! -d "$src_path" ]; then
        echo "⚠️  Skipping $package_name: source directory not found at $src_path"
        return
    fi

    # Create output directory
    mkdir -p "$DOCS_DIR/$package_name"

    # Generate documentation
    if uv run pdoc --html --output-dir "docs/$package_name" "$src_path/$module_name" 2>/dev/null; then
        echo "✅ Generated docs for $package_name"
    else
        echo "❌ Failed to generate docs for $package_name (skipping)"
    fi
}

# Package definitions: name, source_path, module_name
declare -a packages=(
    "aipype|packages/aipype/src|aipype"
    "aipype-extras|packages/aipype-extras/src|aipype_extras"
    "aipype-g|packages/aipype-g/src|aipype_g"
    # Note: aipype-examples has import issues, skipping for now
)

# Build specific package or all packages
if [ -n "$SPECIFIC_PACKAGE" ]; then
    found=false
    for package_def in "${packages[@]}"; do
        IFS='|' read -r name src_path module_name <<< "$package_def"
        if [ "$name" = "$SPECIFIC_PACKAGE" ]; then
            build_package_docs "$name" "$src_path" "$module_name"
            found=true
            break
        fi
    done

    if [ "$found" = false ]; then
        echo "❌ Package '$SPECIFIC_PACKAGE' not found. Available packages:"
        for package_def in "${packages[@]}"; do
            IFS='|' read -r name src_path module_name <<< "$package_def"
            echo "   - $name"
        done
        exit 1
    fi
else
    echo "🚀 Building documentation for all packages..."
    for package_def in "${packages[@]}"; do
        IFS='|' read -r name src_path module_name <<< "$package_def"
        build_package_docs "$name" "$src_path" "$module_name"
    done
fi

# Create main index page
echo "📝 Creating main index page..."
cat > "$DOCS_DIR/index.html" << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>aipype Documentation</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 800px;
            margin: 2rem auto;
            padding: 2rem;
            line-height: 1.6;
            color: #333;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 0.5rem;
        }
        .package {
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 4px;
        }
        .package h3 {
            margin-top: 0;
            color: #2c3e50;
        }
        .package a {
            color: #3498db;
            text-decoration: none;
            font-weight: 500;
        }
        .package a:hover {
            text-decoration: underline;
        }
        .description {
            color: #666;
            margin: 0.5rem 0;
        }
        .footer {
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid #eee;
            text-align: center;
            color: #666;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <h1>🤖 aipype Documentation</h1>
    <p>Modular AI agent framework with declarative pipeline-based task orchestration.</p>

    <div class="package">
        <h3><a href="aipype/aipype/index.html">aipype</a></h3>
        <p class="description">Core framework containing PipelineAgent, LLMTask, SearchTask, and TaskResult components.</p>
    </div>

    <div class="package">
        <h3><a href="aipype-extras/aipype_extras/index.html">aipype-extras</a></h3>
        <p class="description">Optional tools and utilities including LLM log viewer.</p>
    </div>

    <div class="package">
        <h3><a href="aipype-g/aipype_g/index.html">aipype-g</a></h3>
        <p class="description">Google API integrations for Gmail and Google Sheets services.</p>
    </div>

    <div class="footer">
        <p>Generated with <a href="https://pdoc3.github.io/pdoc/">pdoc3</a></p>
    </div>
</body>
</html>
EOF

echo "✅ Created main index page at docs/index.html"

# Serve documentation if requested
if [ "$SERVE" = true ]; then
    echo "🌐 Starting local documentation server..."
    echo "📱 Open http://localhost:8000 in your browser"
    echo "🔄 Press Ctrl+C to stop the server"
    cd "$DOCS_DIR"
    python3 -m http.server 8000
fi

echo "🎉 Documentation build complete!"
echo "📁 Documentation available in: $DOCS_DIR"
echo "🔗 Open $DOCS_DIR/index.html in your browser"