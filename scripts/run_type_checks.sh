#!/bin/bash

# run_type_checks.sh - Type checking script for aipype workspace
#
# This script runs type checking on all packages in the aipype workspace
# using the consolidated scripts/check_types.py script.
#
# Usage:
#   ./scripts/run_type_checks.sh
#
# Requirements:
#   - pyright must be installed (npm install -g pyright)
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
echo -e "${BLUE}     Type Checking All Packages${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Check if pyright is available
if ! command -v pyright &> /dev/null; then
    echo -e "${RED}Error: pyright is not installed or not in PATH${NC}"
    echo "Install it with: npm install -g pyright"
    exit 1
fi

echo -e "${YELLOW}Workspace root:${NC} $WORKSPACE_ROOT"
echo ""

# Package list in order
PACKAGES=("aipype" "aipype-extras" "aipype-g" "aipype-examples")

# Counters for summary
total_packages=0
failed_packages=()

# Type check each package using the consolidated script
for package in "${PACKAGES[@]}"; do
    package_dir="packages/$package"

    echo -e "${BLUE}Checking package: ${package}${NC}"
    echo "Directory: $package_dir"

    # Check if package directory exists
    if [[ ! -d "$package_dir" ]]; then
        echo -e "${RED}✗ Package directory not found: $package_dir${NC}"
        failed_packages+=("$package")
        continue
    fi

    # Run type checking using consolidated script
    echo "Running: python scripts/check_types.py --package $package --summary"

    if python scripts/check_types.py --package "$package" --summary; then
        echo -e "${GREEN}✓ Type checking passed for $package${NC}"
    else
        echo -e "${RED}✗ Type checking failed for $package${NC}"
        failed_packages+=("$package")
    fi

    total_packages=$((total_packages + 1))
    echo ""
done

# Print summary
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}     Type Checking Summary${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

successful_packages=$((total_packages - ${#failed_packages[@]}))

echo "Packages checked: $total_packages"
echo -e "Successful: ${GREEN}$successful_packages${NC}"

if [[ ${#failed_packages[@]} -eq 0 ]]; then
    echo -e "Failed: ${GREEN}0${NC}"
    echo ""
    echo -e "${GREEN}🎉 All packages passed type checking!${NC}"
    exit 0
else
    echo -e "Failed: ${RED}${#failed_packages[@]}${NC}"
    echo ""
    echo -e "${RED}Failed packages:${NC}"
    for package in "${failed_packages[@]}"; do
        echo -e "  ${RED}✗${NC} $package"
    done
    echo ""
    echo -e "${YELLOW}Please fix type issues in the failed packages.${NC}"
    exit 1
fi