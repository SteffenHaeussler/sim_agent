"""Minimal utilities for evaluation tests."""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import yaml

from src.agent.adapters.adapter import AbstractNotifications
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
