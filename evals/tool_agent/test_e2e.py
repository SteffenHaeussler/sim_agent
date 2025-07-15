import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.agent.entrypoints.app import app
from tests.utils import get_fixtures
from evals.base_eval_db import BaseEvaluationTest

current_path = Path(__file__).parent
fixtures = get_fixtures(current_path, keys=["e2e"])

# Create test client
client = TestClient(app)


class TestEvalE2E(BaseEvaluationTest):
    """End-to-End evaluation tests."""

    RUN_TYPE = "e2e"
    TEST_TYPE = "e2e"
    EVALUATION_CATEGORY = "e2e"

    @pytest.mark.parametrize(
        "fixture_name, fixture",
        [
            pytest.param(fixture_name, fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_eval_e2e(self, fixture_name, fixture):
        """Run E2E test with optional LLM judge evaluation."""

        # Extract test data
        test_data = fixture["e2e"]
        question = test_data["question"]
        expected_response = test_data["response"]

        # Start timing
        start_time = time.time()

        # Make API request
        params = {"question": question, "q_id": fixture_name}
        headers = {"X-Session-ID": f"test-{fixture_name}"}
        response = client.get("/answer", params=params, headers=headers)

        # Calculate execution time
        execution_time_ms = int((time.time() - start_time) * 1000)

        # Add delay to avoid rate limiting (E2E makes many API calls internally)
        time.sleep(60)

        # Extract actual response
        if response.status_code == 200:
            data = response.json()
            actual_response = data.get("response", "")
        else:
            actual_response = f"Error: {response.status_code}"

        # Evaluate with judge and record to database
        self.evaluate_with_judge(
            fixture_name=fixture_name,
            question=question,
            expected_response=expected_response,
            actual_response=actual_response,
            test_data=test_data,
            execution_time_ms=execution_time_ms,
            metadata={"status_code": response.status_code, "api_endpoint": "/answer"},
        )

    @classmethod
    def teardown_class(cls):
        """Generate summary report after all tests."""
        # Call parent teardown which handles database completion and summary
        super().teardown_class()
