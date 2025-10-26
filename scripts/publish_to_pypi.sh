#!/bin/bash

# PyPI Publishing Script for aipype workspace
# Default: Publishes to TestPyPI (safe)
# Use --prod flag for production PyPI

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
PYPI_URL="https://test.pypi.org/legacy/"
PYPI_NAME="TestPyPI"
PRODUCTION=false
DRY_RUN=false
CLEAN_DIST=true

# Packages to publish (excluding examples)
PACKAGES=("aipype" "aipype-extras" "aipype-g")

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prod|-p)
            PYPI_URL="https://upload.pypi.org/legacy/"
            PYPI_NAME="PyPI (Production)"
            PRODUCTION=true
            shift
            ;;
        --dry-run|-d)
            DRY_RUN=true
            shift
            ;;
        --no-clean)
            CLEAN_DIST=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --prod, -p      Publish to production PyPI (default: TestPyPI)"
            echo "  --dry-run, -d   Show what would be published without doing it"
            echo "  --no-clean      Don't clean dist directory before building"
            echo "  --help, -h      Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  UV_PUBLISH_TOKEN    PyPI API token for publishing"
            echo ""
            echo "Examples:"
            echo "  $0                  # Publish to TestPyPI"
            echo "  $0 --prod          # Publish to production PyPI"
            echo "  $0 --dry-run       # Preview what would be published"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}🚀 aipype PyPI Publishing Script${NC}"
echo -e "${BLUE}==================================${NC}"
echo ""

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]] || [[ ! -d "packages" ]]; then
    echo -e "${RED}❌ Error: This script must be run from the workspace root directory${NC}"
    exit 1
fi

# Check for API token
if [[ -z "$UV_PUBLISH_TOKEN" ]]; then
    echo -e "${RED}❌ Error: UV_PUBLISH_TOKEN environment variable not set${NC}"
    echo -e "${YELLOW}Please set your PyPI API token:${NC}"
    echo "export UV_PUBLISH_TOKEN=your-pypi-token"
    exit 1
fi

echo -e "${BLUE}📋 Configuration:${NC}"
echo -e "  Target: ${YELLOW}$PYPI_NAME${NC}"
echo -e "  Packages: ${YELLOW}${PACKAGES[*]}${NC}"
echo -e "  Dry run: ${YELLOW}$DRY_RUN${NC}"
echo -e "  Clean dist: ${YELLOW}$CLEAN_DIST${NC}"
echo ""

# Production confirmation
if [[ "$PRODUCTION" == true ]] && [[ "$DRY_RUN" == false ]]; then
    echo -e "${YELLOW}⚠️  WARNING: You are about to publish to PRODUCTION PyPI!${NC}"
    echo -e "${YELLOW}   This cannot be undone. Published versions are permanent.${NC}"
    echo ""
    read -p "Are you sure you want to continue? (type 'yes' to confirm): " confirm
    if [[ "$confirm" != "yes" ]]; then
        echo -e "${YELLOW}❌ Publishing cancelled${NC}"
        exit 0
    fi
    echo ""
fi

# Clean dist directory if requested
if [[ "$CLEAN_DIST" == true ]]; then
    echo -e "${BLUE}🧹 Cleaning dist directory...${NC}"
    if [[ "$DRY_RUN" == false ]]; then
        rm -rf dist/
        mkdir -p dist/
    else
        echo "  [DRY RUN] Would remove and recreate dist/ directory"
    fi
    echo ""
fi

# Build packages
echo -e "${BLUE}🔨 Building packages...${NC}"
for package in "${PACKAGES[@]}"; do
    echo -e "${BLUE}  Building $package...${NC}"
    if [[ "$DRY_RUN" == false ]]; then
        uv build --package "$package"
    else
        echo "  [DRY RUN] Would run: uv build --package $package"
    fi
done
echo ""

# Show what would be published
echo -e "${BLUE}📦 Packages to publish:${NC}"
if [[ "$DRY_RUN" == false ]]; then
    if [[ -d "dist" ]]; then
        ls -la dist/
    else
        echo -e "${YELLOW}  No dist directory found${NC}"
    fi
else
    echo "  [DRY RUN] Would list contents of dist/ directory"
fi
echo ""

# Publish packages
if [[ "$DRY_RUN" == false ]]; then
    echo -e "${BLUE}🚀 Publishing to $PYPI_NAME...${NC}"

    # Check if dist directory has files
    if [[ ! -d "dist" ]] || [[ -z "$(ls -A dist/)" ]]; then
        echo -e "${RED}❌ Error: No built packages found in dist/ directory${NC}"
        exit 1
    fi

    # Publish all built packages
    uv publish --publish-url "$PYPI_URL" dist/*

    echo ""
    echo -e "${GREEN}✅ Successfully published to $PYPI_NAME!${NC}"

    if [[ "$PRODUCTION" == false ]]; then
        echo -e "${YELLOW}📝 Note: Published to TestPyPI. Use --prod flag for production.${NC}"
        echo -e "${YELLOW}   View at: https://test.pypi.org/project/aipype/${NC}"
    else
        echo -e "${GREEN}🎉 Packages are now live on production PyPI!${NC}"
        echo -e "${GREEN}   View at: https://pypi.org/project/aipype/${NC}"
    fi
else
    echo -e "${YELLOW}[DRY RUN] Would publish packages to $PYPI_NAME${NC}"
    echo -e "${YELLOW}[DRY RUN] Command: uv publish --publish-url $PYPI_URL dist/*${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Publishing script completed!${NC}"