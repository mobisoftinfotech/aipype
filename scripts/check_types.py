#!/usr/bin/env python3
"""
Type checking script for all aipype packages.

This script provides convenient type checking using pyright (the same engine as Pylance).
Use this for periodic type checking from the command line.

Prerequisites:
    npm install -g pyright

Usage:
    python scripts/check_types.py                           # Check all packages
    python scripts/check_types.py --package aipype        # Check specific package
    python scripts/check_types.py --package aipype-extras # Check specific package
    python scripts/check_types.py --file <file>             # Check specific file
    python scripts/check_types.py --summary                 # Show only summary, not individual warnings
"""

import argparse
import subprocess
import sys
from pathlib import Path


# All packages in the workspace
PACKAGES = [
    "packages/aipype/src/aipype/",
    "packages/aipype-extras/src/aipype_extras/",
    "packages/aipype-g/src/aipype_g/",
    "packages/aipype-examples/src/aipype_examples/",
]


def run_pyright(
    target: str,
    summary_only: bool = False,
) -> int:
    """Run pyright type checking on the specified target."""
    cmd = ["pyright"]

    if summary_only:
        cmd.append("--outputjson")

    cmd.append(target)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if summary_only and result.stdout:
            import json

            try:
                data = json.loads(result.stdout)
                summary = data.get("summary", {})
                error_count = summary.get("errorCount", 0)
                warning_count = summary.get("warningCount", 0)
                info_count = summary.get("informationCount", 0)

                print(f"Type checking results for {target}:")
                print(f"  Errors: {error_count}")
                print(f"  Warnings: {warning_count}")
                print(f"  Information: {info_count}")

                if error_count == 0 and warning_count == 0 and info_count == 0:
                    print("  [SUCCESS] No type issues found!")
                    return 0
                else:
                    # Return non-zero exit code if any issues found
                    return 1

            except json.JSONDecodeError:
                print("Failed to parse pyright JSON output")
                print(result.stdout)
        else:
            # Regular output
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)

        return result.returncode

    except FileNotFoundError:
        print("Error: pyright not found. Install it with: npm install -g pyright")
        return 1
    except Exception as e:
        print(f"Error running pyright: {e}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Type check aipype packages")
    parser.add_argument(
        "--package",
        type=str,
        choices=["aipype", "aipype-extras", "aipype-g", "aipype-examples"],
        help="Check specific package only",
    )
    parser.add_argument("--file", type=str, help="Check specific file or directory")
    parser.add_argument(
        "--summary", action="store_true", help="Show only summary of results"
    )

    args = parser.parse_args()

    # Determine targets
    targets = []
    if args.file:
        targets = [args.file]
    elif args.package:
        package_map = {
            "aipype": "packages/aipype/src/aipype/",
            "aipype-extras": "packages/aipype-extras/src/aipype_extras/",
            "aipype-g": "packages/aipype-g/src/aipype_g/",
            "aipype-examples": "packages/aipype-examples/src/aipype_examples/",
        }
        targets = [package_map[args.package]]
    else:
        targets = PACKAGES

    print(f"Running type check on {len(targets)} target(s)")
    print("Mode: Standard (errors and warnings)")
    print("-" * 50)

    total_exit_code = 0
    for target in targets:
        if len(targets) > 1:
            print(f"\nChecking: {target}")
            print("-" * 30)

        # Check if target exists
        if not Path(target).exists():
            print(f"Warning: Target {target} does not exist, skipping...")
            continue

        exit_code = run_pyright(target, args.summary)
        if exit_code != 0:
            total_exit_code = exit_code

    if total_exit_code == 0:
        print("\n[SUCCESS] All type checking completed successfully!")
    else:
        print(f"\n[ERROR] Type checking found issues (exit code: {total_exit_code})")

    return total_exit_code


if __name__ == "__main__":
    sys.exit(main())