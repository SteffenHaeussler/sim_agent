"""Minimal utilities for evaluation tests."""

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.agent.adapters.database import BaseDatabaseAdapter
from src.agent.adapters.notifications import AbstractNotifications
from src.agent.domain import events


class CollectingNotifications(AbstractNotifications):
    def __init__(self):
        self.sent = defaultdict(list)

    def send(self, destination, event: events.Event):
        self.sent[destination].append(event)


def load_yaml_fixtures(
    test_dir: Path, subdirectory: str, recursive: bool = True
) -> Dict[str, Any]:
    """
    Load YAML test fixtures from a subdirectory.

    Args:
        test_dir: Base test directory
        subdirectory: Subdirectory to load fixtures from
        recursive: Whether to search recursively in subdirectories

    Returns:
        Dict with test_name as key and test data as value (without subdirectory nesting)
    """
    fixtures = {}
    fixtures_dir = test_dir / subdirectory

    if not fixtures_dir.exists():
        return fixtures

    # Get all YAML files (recursive or not)
    yaml_files = (
        fixtures_dir.rglob("*.yaml") if recursive else fixtures_dir.glob("*.yaml")
    )

    for yaml_file in yaml_files:
        with open(yaml_file, "r") as f:
            suite_data = yaml.safe_load(f)

        # Calculate relative path for nested directories
        rel_path = yaml_file.relative_to(fixtures_dir).parent
        path_prefix = str(rel_path).replace("/", "_") if str(rel_path) != "." else ""

        # Extract tests from the suite
        for test in suite_data.get("tests", []):
            # Create unique test name including path if nested
            test_name = (
                f"{path_prefix}_{yaml_file.stem}_{test['name']}"
                if path_prefix
                else f"{yaml_file.stem}_{test['name']}"
            )
            test_name = test_name.strip("_")

            # Merge suite defaults with test-specific criteria
            test_data = test.copy()
            if "default_judge_criteria" in suite_data:
                test_data["judge_criteria"] = suite_data[
                    "default_judge_criteria"
                ].copy()
                if "judge_criteria" in test:
                    test_data["judge_criteria"].update(test["judge_criteria"])
            elif "judge_criteria" not in test:
                test_data["judge_criteria"] = {}

            # Return flat structure - just the test data
            fixtures[test_name] = test_data

    return fixtures


def get_report_dir() -> Path:
    """Get the report directory from environment variable or use default."""
    # Check environment variable first
    report_dir = os.environ.get("EVALS_REPORT_DIR")
    if report_dir:
        return Path(report_dir)

    # Default: evals/reports relative to this file
    return Path(__file__).parent / "reports"


def save_test_report(results: List[Dict[str, Any]], test_name: str) -> None:
    """Save test results to a JSON report file with timestamp and optionally to database."""
    # Always save to JSON
    report_dir = get_report_dir()
    report_dir.mkdir(exist_ok=True, parents=True)

    timestamp = int(time.time())
    # Create run_id from test_name and timestamp (same as filename without .json)
    run_id = f"{test_name}_report_{timestamp}"
    filename = f"{run_id}.json"

    with open(report_dir / filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Report saved to: {report_dir / filename}")

    # Try to save to database if configured
    if os.environ.get("EVALS_DB_CONNECTION"):
        try:
            save_to_database(results, test_name, run_id)
        except Exception as e:
            print(f"Failed to save to database: {e}")


def save_to_database(
    results: List[Dict[str, Any]], test_suite: str, run_id: str
) -> Optional[str]:
    """Save test results to database."""
    connection_string = os.environ.get("EVALS_DB_CONNECTION")
    if not connection_string:
        return None

    db = BaseDatabaseAdapter({"connection_string": connection_string})

    try:
        db.connect()

        # Create test run record
        run_data = {
            "run_id": run_id,
            "test_suite": test_suite,
            "total_tests": len(results),
            "passed_tests": sum(1 for r in results if r.get("passed", False)),
            "failed_tests": sum(1 for r in results if not r.get("passed", False)),
        }

        if not db.insert_data("test_runs", run_data):
            print("Failed to insert test run")
            return None

        # Prepare all test results for batch insert
        test_results = []
        for result in results:
            test_data = {
                "run_id": run_id,
                "test_name": result.get("test_name") or result.get("test", ""),
                "question": result.get("question", ""),
                "expected": str(result.get("expected", "")),
                "actual": str(result.get("actual", "")),
                "passed": result.get("passed", False),
                "execution_time_ms": result.get("execution_time_ms"),
                "overall_score": result.get("overall_score"),
                "accuracy_score": result.get("accuracy"),
                "relevance_score": result.get("relevance"),
                "completeness_score": result.get("completeness"),
                "hallucination_score": result.get("hallucination"),
                "judge_assessment": result.get("judge_assessment"),
            }

            # Remove None values
            test_data = {k: v for k, v in test_data.items() if v is not None}
            test_results.append(test_data)

        # Insert all test results in a single transaction
        if not db.insert_batch("test_results", test_results):
            print("Failed to insert test results")
            return None

        print(f"Results saved to database with run_id: {run_id}")
        return run_id

    finally:
        db.disconnect()
