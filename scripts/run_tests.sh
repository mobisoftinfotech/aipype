#!/bin/bash

# run_tests.sh - Test runner script for aipype workspace
#
# This script runs unit and/or integration tests for all packages in the workspace.
# All packages with test directories are included.
#
# Usage:
#   ./scripts/run_tests.sh              # Run unit tests only (default)
#   ./scripts/run_tests.sh --unit       # Run unit tests only
#   ./scripts/run_tests.sh --integration # Run integration tests only
#   ./scripts/run_tests.sh --all        # Run both unit and integration tests
#   ./scripts/run_tests.sh --verbose    # Run with verbose output
#   ./scripts/run_tests.sh -v           # Short form of --verbose
#
# Requirements:
#   - pytest must be installed
#   - Run from workspace root directory
#   - For integration tests: external services may be required (Ollama, etc.)

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to workspace root
cd "$WORKSPACE_ROOT"

# Default settings
RUN_UNIT=true
RUN_INTEGRATION=false
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            RUN_UNIT=true
            RUN_INTEGRATION=false
            shift
            ;;
        --integration)
            RUN_UNIT=false
            RUN_INTEGRATION=true
            shift
            ;;
        --all)
            RUN_UNIT=true
            RUN_INTEGRATION=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --unit         Run unit tests only (default)"
            echo "  --integration  Run integration tests only"
            echo "  --all          Run both unit and integration tests"
            echo "  --verbose, -v  Run with verbose output"
            echo "  --help, -h     Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Run unit tests"
            echo "  $0 --all --verbose    # Run all tests with verbose output"
            echo "  $0 --integration      # Run integration tests only"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}     Running Tests${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Check if pytest is available
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv is not installed or not in PATH${NC}"
    echo "This script requires uv to run pytest"
    exit 1
fi

echo -e "${YELLOW}Workspace root:${NC} $WORKSPACE_ROOT"

# Show what tests will be run
echo -e "${YELLOW}Test configuration:${NC}"
if [[ "$RUN_UNIT" == true ]]; then
    echo "  ✓ Unit tests"
fi
if [[ "$RUN_INTEGRATION" == true ]]; then
    echo "  ✓ Integration tests"
fi
if [[ "$VERBOSE" == true ]]; then
    echo "  ✓ Verbose output"
fi
echo ""

# Packages with tests (excludes aipype-examples)
PACKAGES_WITH_TESTS=("aipype" "aipype-extras" "aipype-g" "aipype-examples")

# Build pytest arguments
PYTEST_ARGS=()
if [[ "$VERBOSE" == true ]]; then
    PYTEST_ARGS+=("-v")
fi

# Counters for summary
total_packages=0
passed_packages=0
failed_packages=()

# Function to run tests for a specific type
run_test_type() {
    local test_type=$1
    local test_suffix=$2

    echo -e "${BLUE}Running ${test_type} tests...${NC}"
    echo ""

    for package in "${PACKAGES_WITH_TESTS[@]}"; do
        test_dir="packages/$package/$test_suffix"

        if [[ -d "$test_dir" ]]; then
            echo -e "${YELLOW}Testing $package ($test_type):${NC} $test_dir"

            # Run pytest and capture exit code (temporarily disable exit on error)
            set +e
            uv run pytest "$test_dir" "${PYTEST_ARGS[@]}"
            exit_code=$?
            set -e

            if [[ $exit_code -eq 0 ]]; then
                echo -e "${GREEN}✓ $package ($test_type) passed${NC}"
                passed_packages=$((passed_packages + 1))
            elif [[ $exit_code -eq 5 ]]; then
                # Exit code 5 means "no tests collected" which is OK for some packages
                echo -e "${YELLOW}⚠ $package ($test_type): No tests found, marking as passed${NC}"
                passed_packages=$((passed_packages + 1))
            else
                echo -e "${RED}✗ $package ($test_type) failed (exit code: $exit_code)${NC}"
                failed_packages+=("$package ($test_type)")
            fi
        else
            echo -e "${YELLOW}⚠ $package: No $test_type tests found (skipping $test_dir)${NC}"
        fi

        total_packages=$((total_packages + 1))
        echo ""
    done
}

# Run the appropriate tests
if [[ "$RUN_UNIT" == true ]]; then
    run_test_type "unit" "tests"
fi

if [[ "$RUN_INTEGRATION" == true ]]; then
    run_test_type "integration" "integration_tests"
fi

# Print summary
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}     Test Results Summary${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

echo "Total test suites: $total_packages"
echo -e "Passed: ${GREEN}$passed_packages${NC}"

if [[ ${#failed_packages[@]} -eq 0 ]]; then
    echo -e "Failed: ${GREEN}0${NC}"
    echo ""
    echo -e "${GREEN}🎉 All tests passed!${NC}"
    exit 0
else
    echo -e "Failed: ${RED}${#failed_packages[@]}${NC}"
    echo ""
    echo -e "${RED}Failed test suites:${NC}"
    for package in "${failed_packages[@]}"; do
        echo -e "  ${RED}✗${NC} $package"
    done
    echo ""
    echo -e "${YELLOW}Please fix the failing tests before proceeding.${NC}"
    exit 1
fi