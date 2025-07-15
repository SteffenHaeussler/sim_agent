"""Simple SQL E2E tests that call the actual CLI."""

import json
import os
from pathlib import Path

import pytest
import yaml

from evals.base_sql_eval import BaseSQLEvalTest

current_path = Path(__file__).parent


def load_yaml_fixtures(test_dir):
    """Load YAML test fixtures from the sql_e2e directory."""
    fixtures = {}
    sql_e2e_dir = test_dir / "sql_e2e"

    if not sql_e2e_dir.exists():
        print(f"WARNING: sql_e2e directory not found at {sql_e2e_dir}")
        return fixtures

    # Load all YAML files in the sql_e2e directory
    for yaml_file in sql_e2e_dir.glob("*.yaml"):
        with open(yaml_file, "r") as f:
            suite_data = yaml.safe_load(f)

        # Extract tests from the suite
        for test in suite_data.get("tests", []):
            test_name = f"{yaml_file.stem}_{test['name']}"

            # Merge suite defaults with test-specific criteria
            judge_criteria = suite_data.get("default_judge_criteria", {}).copy()
            if "judge_criteria" in test:
                judge_criteria.update(test["judge_criteria"])

            # Convert to expected format
            fixtures[test_name] = {
                "sql_e2e": {
                    "question": test["question"],
                    "expected_sql": test["sql"],
                    "judge_criteria": judge_criteria,
                }
            }

    return fixtures


# Load fixtures from YAML files
fixtures = load_yaml_fixtures(current_path)

# Load database schema
with open(current_path / "sql_e2e/schema.json", "r") as f:
    schema_data = json.load(f)


class TestSQLEndToEndSimple(BaseSQLEvalTest):
    """Simple SQL End-to-End evaluation tests using CLI."""

    RUN_TYPE = "sql_e2e"
    TEST_TYPE = "sql_e2e"

    def setup_method(self):
        """Setup for each test."""
        super().setup_method()
        self.current_path = current_path
        self.schema = schema_data

        # Set environment to use test database
        env = os.environ.copy()
        env["IS_TESTING"] = "true"

    def extract_sql_from_output(self, output: str) -> str:
        """Extract SQL query from CLI output."""
        # Look for SQL between ```sql markers or after "Generated SQL:" line
        lines = output.split("\n")

        # Try to find SQL block
        in_sql_block = False
        sql_lines = []

        for line in lines:
            if line.strip().startswith("```sql"):
                in_sql_block = True
                continue
            elif line.strip() == "```" and in_sql_block:
                break
            elif in_sql_block:
                sql_lines.append(line)
            elif "Generated SQL:" in line or "SQL Query:" in line:
                # Sometimes the SQL is on the next lines
                idx = lines.index(line)
                for next_line in lines[idx + 1 :]:
                    if next_line.strip() and not next_line.startswith(
                        "2025-"
                    ):  # Skip log lines
                        sql_lines.append(next_line)

        return "\n".join(sql_lines).strip()

    @pytest.mark.parametrize(
        "fixture_name, fixture",
        [
            pytest.param(fixture_name, fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_sql_e2e_via_cli(self, fixture_name, fixture):
        """Run SQL E2E test via CLI."""

        # Extract test data
        test_data = fixture["sql_e2e"]
        question = test_data["question"]
        expected_sql = test_data["expected_sql"]

        # Run CLI command
        output = self.run_cli_command(question)

        # Extract SQL from output
        actual_sql = self.extract_sql_from_output(output)

        # If we couldn't extract SQL, try to find it in the raw output
        if not actual_sql:
            # Sometimes the SQL is just printed directly
            for line in output.split("\n"):
                if "SELECT" in line.upper():
                    actual_sql = line.strip()
                    break

        # Evaluate the result using the base class method
        self.evaluate_with_judge(
            fixture_name=fixture_name,
            question=question,
            expected_response={"sql": expected_sql},
            actual_response={"sql": actual_sql},
            test_data=test_data,
            judge_question=question,
            sql_query=actual_sql,
            schema_context=schema_data,
        )

    @classmethod
    def teardown_class(cls):
        """Generate summary report after all tests."""
        super().teardown_class()
