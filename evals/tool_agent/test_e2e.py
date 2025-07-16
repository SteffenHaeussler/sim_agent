import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from evals.llm_judge import JudgeCriteria, LLMJudge
from evals.utils import load_yaml_fixtures
from src.agent.entrypoints.app import app

current_path = Path(__file__).parent
# Load fixtures from YAML file
fixtures = load_yaml_fixtures(current_path, "")

# Create test client
client = TestClient(app)


class TestEvalE2E:
    """End-to-End evaluation tests."""

    def setup_method(self):
        """Initialize LLM Judge for evaluation."""
        self.judge = LLMJudge()

    @pytest.mark.parametrize(
        "fixture_name, fixture",
        [
            pytest.param(fixture_name, fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_eval_e2e(self, fixture_name, fixture):
        """Run E2E test with optional LLM judge evaluation."""

        # Extract test data - fixture is now the test data directly
        question = fixture["question"]
        expected_response = fixture["response"]

        # Start timing
        # start_time = time.time()

        # Make API request
        params = {"question": question, "q_id": fixture_name}
        headers = {"X-Session-ID": f"test-{fixture_name}"}
        response = client.get("/answer", params=params, headers=headers)

        # Calculate execution time
        # execution_time_ms = int((time.time() - start_time) * 1000)

        # Add delay to avoid rate limiting (E2E makes many API calls internally)
        time.sleep(60)

        # Extract actual response
        if response.status_code == 200:
            data = response.json()
            actual_response = data.get("response", "")
        else:
            actual_response = f"Error: {response.status_code}"

        # Use LLM Judge for evaluation
        criteria = JudgeCriteria(**fixture.get("judge_criteria", {}))
        judge_result = self.judge.evaluate(
            question=question,
            expected=str(expected_response),
            actual=str(actual_response),
            criteria=criteria,
            test_type="e2e",
        )

        # Add delay to avoid rate limiting
        time.sleep(1)

        # Assert judge passed
        assert judge_result.passed, f"Judge failed: {judge_result.overall_assessment}"
