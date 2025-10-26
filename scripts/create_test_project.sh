#!/bin/bash

# Create Test Project Script for aipype packages
# Generates a UV-based project to verify published packages on PyPI
# How to run:
# ./scripts/create_test_project.sh /tmp/verify-publish

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
USE_TEST_PYPI=false
RUN_VERIFICATION=true
PROJECT_NAME="aipype-test"

# Parse command line arguments
TARGET_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --test-pypi|-t)
            USE_TEST_PYPI=true
            shift
            ;;
        --no-verify)
            RUN_VERIFICATION=false
            shift
            ;;
        --name|-n)
            PROJECT_NAME="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [TARGET_DIR] [OPTIONS]"
            echo ""
            echo "Arguments:"
            echo "  TARGET_DIR          Directory where test project will be created"
            echo "                      (default: ./test-project)"
            echo ""
            echo "Options:"
            echo "  --test-pypi, -t     Use TestPyPI instead of production PyPI"
            echo "  --no-verify         Skip import verification after installation"
            echo "  --name, -n NAME     Custom project name (default: aipype-test)"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 /tmp/mytest              # Create test project in /tmp/mytest"
            echo "  $0 --test-pypi              # Test from TestPyPI"
            echo "  $0 ~/test --name my-test    # Custom location and name"
            exit 0
            ;;
        -*)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            TARGET_DIR="$1"
            shift
            ;;
    esac
done

# Set default target directory if not provided
if [[ -z "$TARGET_DIR" ]]; then
    TARGET_DIR="./test-project"
fi

echo -e "${BLUE}🧪 aipype Test Project Creator${NC}"
echo -e "${BLUE}==============================${NC}"
echo ""

echo -e "${BLUE}📋 Configuration:${NC}"
echo -e "  Project name: ${YELLOW}$PROJECT_NAME${NC}"
echo -e "  Target directory: ${YELLOW}$TARGET_DIR${NC}"
echo -e "  PyPI source: ${YELLOW}$(if [[ "$USE_TEST_PYPI" == true ]]; then echo "TestPyPI"; else echo "Production PyPI"; fi)${NC}"
echo -e "  Run verification: ${YELLOW}$RUN_VERIFICATION${NC}"
echo ""

# Check if target directory already exists
if [[ -d "$TARGET_DIR" ]]; then
    echo -e "${YELLOW}⚠️  Warning: Directory $TARGET_DIR already exists${NC}"
    read -p "Do you want to remove it and continue? (y/n): " confirm
    if [[ "$confirm" != "y" ]]; then
        echo -e "${YELLOW}❌ Operation cancelled${NC}"
        exit 0
    fi
    rm -rf "$TARGET_DIR"
fi

# Create directory
echo -e "${BLUE}📁 Creating project directory...${NC}"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"
echo -e "${GREEN}  ✓ Directory created: $TARGET_DIR${NC}"
echo ""

# Initialize UV project
echo -e "${BLUE}🔧 Initializing UV project...${NC}"
uv init --name "$PROJECT_NAME" --no-readme
echo -e "${GREEN}  ✓ UV project initialized${NC}"
echo ""

# Add packages
echo -e "${BLUE}📦 Installing aipype packages...${NC}"

if [[ "$USE_TEST_PYPI" == true ]]; then
    echo -e "${YELLOW}  Installing from TestPyPI...${NC}"
    # For TestPyPI, we need to add both TestPyPI and PyPI sources
    # because dependencies might not be on TestPyPI
    uv add --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ aipype aipype-extras aipype-g
else
    echo -e "${YELLOW}  Installing from production PyPI...${NC}"
    uv add aipype aipype-extras aipype-g
fi

echo -e "${GREEN}  ✓ Packages installed${NC}"
echo ""

# Create verification script
if [[ "$RUN_VERIFICATION" == true ]]; then
    echo -e "${BLUE}🧪 Creating verification script...${NC}"

    cat > verify_imports.py << 'EOF'
#!/usr/bin/env python3
"""Verify that all aipype packages can be imported successfully."""

import sys

def test_import(module_name: str, from_items: list[str] | None = None) -> bool:
    """Test if a module or specific items can be imported."""
    try:
        if from_items:
            module = __import__(module_name, fromlist=from_items)
            for item in from_items:
                if not hasattr(module, item):
                    print(f"✗ {item} not found in {module_name}")
                    return False
            print(f"✓ from {module_name} import {', '.join(from_items)}")
        else:
            __import__(module_name)
            print(f"✓ import {module_name}")
        return True
    except ImportError as e:
        if from_items:
            print(f"✗ Failed to import from {module_name}: {e}")
        else:
            print(f"✗ Failed to import {module_name}: {e}")
        return False

def main() -> int:
    """Run all import tests."""
    print("Testing aipype package imports...")
    print("=" * 60)

    results = []

    # Test aipype core package
    print("\n📦 Testing aipype core package:")
    results.append(test_import("aipype"))
    results.append(test_import("aipype", [
        "PipelineAgent", "BaseTask", "TaskResult", "TaskContext",
        "TaskDependency", "DependencyType", "LLMTask", "SearchTask",
        "ConditionalTask", "TransformTask"
    ]))

    # Test aipype tasklib
    print("\n📦 Testing aipype tasklib:")
    results.append(test_import("aipype", [
        "BatchArticleSummarizeTask", "FileSaveTask", "URLFetchTask",
        "ExtractAudioFromVideoTask", "AudioTranscriptTask"
    ]))

    # Test aipype utils
    print("\n📦 Testing aipype utils:")
    results.append(test_import("aipype", [
        "setup_logger", "SearchResult", "SerperSearcher",
        "fetch_main_text", "URLFetcher"
    ]))

    # Test aipype-extras
    print("\n📦 Testing aipype-extras package:")
    results.append(test_import("aipype_extras"))
    results.append(test_import("aipype_extras.llm_log_viewer"))
    results.append(test_import("aipype_extras", [
        "LogEntry", "LLMLogReader"
    ]))

    # Test aipype-g
    print("\n📦 Testing aipype-g package:")
    results.append(test_import("aipype_g"))
    results.append(test_import("aipype_g", [
        "GoogleOAuthTask", "GmailListEmailsTask", "GmailReadEmailTask",
        "ReadGoogleSheetTask"
    ]))
    results.append(test_import("aipype_g", [
        "GmailService", "GoogleSheetsService", "GoogleAuthService"
    ]))
    results.append(test_import("aipype_g", [
        "GmailMessage", "SheetData", "SpreadsheetInfo"
    ]))

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"\n📊 Results: {passed}/{total} tests passed")

    if all(results):
        print("✅ All imports successful! Packages are correctly installed.")
    else:
        print("❌ Some imports failed. Please check the output above.")

    return 0 if all(results) else 1

if __name__ == "__main__":
    sys.exit(main())
EOF

    chmod +x verify_imports.py
    echo -e "${GREEN}  ✓ Verification script created${NC}"
    echo ""

    # Run verification
    echo -e "${BLUE}🔍 Running import verification...${NC}"
    if uv run python verify_imports.py; then
        echo ""
        echo -e "${GREEN}✅ All imports successful!${NC}"
    else
        echo ""
        echo -e "${RED}❌ Some imports failed${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}🎉 Test project created successfully!${NC}"
echo ""
echo -e "${BLUE}📂 Project location: ${YELLOW}$TARGET_DIR${NC}"
echo -e "${BLUE}📝 Next steps:${NC}"
echo "  1. cd $TARGET_DIR"
echo "  2. Edit hello.py to test aipype functionality"
echo "  3. Run: uv run python hello.py"
echo ""
echo -e "${BLUE}💡 Quick test example:${NC}"
echo "  # Edit hello.py and add:"
echo "  from aipype import TaskResult"
echo "  print(TaskResult.success({'message': 'Hello from aipype!'}))"
echo ""
