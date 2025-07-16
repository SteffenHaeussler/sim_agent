"""SQL E2E tests using FastAPI endpoint."""

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from evals.base_eval import BaseEvaluationTest
from evals.utils import load_yaml_fixtures
from src.agent.entrypoints.app import app

current_path = Path(__file__).parent

# Load fixtures from YAML files using the utility function
fixtures = load_yaml_fixtures(current_path, "sql_e2e/temp")

# Load database schema
with open(current_path / "sql_e2e/schema.json", "r") as f:
    schema_data = json.load(f)

# Create test client
client = TestClient(app)


class TestSQLEndToEnd(BaseEvaluationTest):
    """SQL End-to-End evaluation tests using FastAPI endpoint."""

    RUN_TYPE = "sql_e2e"
    TEST_TYPE = "sql_e2e"

    def setup_method(self):
        """Setup for each test."""
        super().setup_method()
        self.current_path = current_path
        self.schema = schema_data

    # def extract_sql_from_response(self, session_id: str, max_retries: int = 10) -> str:
    #     """
    #     Poll the SSE endpoint to get the SQL response.

    #     Args:
    #         session_id: The session ID to poll
    #         max_retries: Maximum number of polling attempts

    #     Returns:
    #         The extracted SQL query
    #     """
    #     # Give the async processing some time to start

    #     # Try to get the response from SSE endpoint
    #     for retry in range(max_retries):
    #         try:
    #             # Get SSE stream
    #             response = client.get(f"/sse/{session_id}", stream=True)

    #             if response.status_code == 200:
    #                 sql_response = ""
    #                 for line in response.iter_lines():
    #                     if line and line.startswith("data: "):
    #                         try:
    #                             data = json.loads(line[6:])  # Skip "data: " prefix
    #                             if data.get("type") == "sql_response":
    #                                 sql_response = data.get("content", "")
    #                                 break
    #                         except json.JSONDecodeError:
    #                             continue

    #                 if sql_response:
    #                     # Extract SQL from the response
    #                     # Look for SQL between ```sql markers
    #                     if "```sql" in sql_response:
    #                         start = sql_response.find("```sql") + 6
    #                         end = sql_response.find("```", start)
    #                         if end > start:
    #                             return sql_response[start:end].strip()

    #                     # Otherwise look for SELECT statement
    #                     lines = sql_response.split("\n")
    #                     for line in lines:
    #                         if "SELECT" in line.upper():
    #                             # Find the full SQL query starting from SELECT
    #                             sql_start = line.upper().find("SELECT")
    #                             return line[sql_start:].strip()

    #                     return sql_response.strip()
    #         except Exception as e:
    #             print(f"Retry {retry + 1}: {e}")

    #         time.sleep(1)  # Wait before retry

    #     return ""

    @pytest.mark.parametrize(
        "fixture_name, fixture",
        [
            pytest.param(fixture_name, fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_sql_e2e(self, fixture_name, fixture):
        """Run SQL E2E test via FastAPI endpoint."""

        # breakpoint()
        # Direct access to test data - no nested structure
        question = fixture["question"]
        expected_sql = fixture["sql"]

        # Create session ID for this test
        session_id = f"test-sql-{fixture_name}"
        headers = {"X-Session-ID": session_id}

        # Start timing
        start_time = time.time()

        # Make API request
        response = client.get("/query", params={"question": question}, headers=headers)

        # Check initial response
        assert response.status_code == 200
        assert response.json()["status"] == "processing"

        # Extract SQL from SSE response
        actual_sql = self.extract_sql_from_response(session_id)

        # Calculate execution time
        execution_time_ms = int((time.time() - start_time) * 1000)

        # Normalize SQL for comparison (basic normalization)
        def normalize_sql(sql: str) -> str:
            """Basic SQL normalization for comparison."""
            # Remove extra whitespace and newlines
            return " ".join(sql.split()).strip().rstrip(";")

        # Evaluate the result
        self.evaluate_with_judge(
            fixture_name=fixture_name,
            question=question,
            expected_response=normalize_sql(expected_sql),
            actual_response=normalize_sql(actual_sql)
            if actual_sql
            else "NO SQL GENERATED",
            test_data=fixture,  # Pass the fixture directly as test_data
            execution_time_ms=execution_time_ms,
        )

    @classmethod
    def teardown_class(cls):
        """Generate summary report after all tests."""
        super().teardown_class()
