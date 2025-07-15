#!/usr/bin/env python3
"""Script to run enhanced evaluation tests with LLM Judge."""

import argparse
import subprocess
import sys
from pathlib import Path


def run_tests(test_type: str = "all", verbose: bool = True, parallel: bool = False):
    """
    Run enhanced evaluation tests.

    Args:
        test_type: Type of tests to run (all, e2e, tool_agent, etc.)
        verbose: Enable verbose output
        parallel: Run tests in parallel
    """

    # Base pytest command
    cmd = ["uv", "run", "python", "-m", "pytest"]

    # Add verbose flag
    if verbose:
        cmd.append("-v")

    # Add parallel execution if requested
    if parallel:
        cmd.extend(["-n", "auto"])

    # Map test types to test files
    test_files = {
        "e2e": "test_e2e_with_judge.py",
        "e2e_optimized": "test_e2e_optimized.py",
        "tool_agent": "test_tool_agent_with_judge.py",
        "all": "test_*_with_judge.py test_*_optimized.py",
    }

    # Get test file pattern
    if test_type in test_files:
        test_pattern = test_files[test_type]
    else:
        print(f"Unknown test type: {test_type}")
        print(f"Available types: {', '.join(test_files.keys())}")
        return 1

    # Add test file pattern
    cmd.append(test_pattern)

    # Add color output
    cmd.append("--color=yes")

    # Add current directory
    cmd.append(".")

    print(f"Running command: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    # Run tests
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        return result.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Run enhanced evaluation tests with LLM Judge"
    )

    parser.add_argument(
        "test_type",
        nargs="?",
        default="all",
        choices=["all", "e2e", "e2e_optimized", "tool_agent"],
        help="Type of tests to run (default: all)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose output (default: True)",
    )

    parser.add_argument(
        "-p", "--parallel", action="store_true", help="Run tests in parallel"
    )

    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Disable verbose output"
    )

    args = parser.parse_args()

    # Handle quiet flag
    verbose = not args.quiet

    # Run tests
    exit_code = run_tests(
        test_type=args.test_type, verbose=verbose, parallel=args.parallel
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
