#!/bin/bash

# run_ruff.sh - Code formatting and linting script for aipype workspace
#
# This script formats and lints all packages in the aipype workspace using ruff.
# It first formats the code, then runs linting with automatic fixes.
#
# Usage:
#   ./scripts/run_ruff.sh
#
# Requirements:
#   - ruff must be installed (pip install ruff)
#   - Run from workspace root directory

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

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}  Ruff Code Formatting & Linting${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Check if ruff is available
if ! command -v ruff &> /dev/null; then
    echo -e "${RED}Error: ruff is not installed or not in PATH${NC}"
    echo "Install it with: pip install ruff"
    exit 1
fi

echo -e "${YELLOW}Workspace root:${NC} $WORKSPACE_ROOT"
echo ""

# Step 1: Format all packages
echo -e "${BLUE}Step 1: Formatting code...${NC}"
echo "Running: ruff format packages/"

if ruff format packages/; then
    echo -e "${GREEN}✓ Code formatting completed successfully${NC}"
else
    echo -e "${RED}✗ Code formatting failed${NC}"
    exit 1
fi

echo ""

# Step 2: Run linting with fixes
echo -e "${BLUE}Step 2: Running linter with automatic fixes...${NC}"
echo "Running: ruff check --fix packages/"

if ruff check --fix packages/; then
    echo -e "${GREEN}✓ Linting completed successfully${NC}"
else
    echo -e "${YELLOW}⚠ Linting found issues that couldn't be auto-fixed${NC}"
    echo "Please review the output above and fix any remaining issues manually."
    exit 1
fi

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}  All ruff checks completed!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "Summary:"
echo -e "  ${GREEN}✓${NC} Code formatted successfully"
echo -e "  ${GREEN}✓${NC} Linting completed with auto-fixes applied"
echo ""
echo "Next steps:"
echo "  - Run type checking: ./scripts/run_type_checks.sh"
echo "  - Run tests: ./scripts/run_tests.sh"