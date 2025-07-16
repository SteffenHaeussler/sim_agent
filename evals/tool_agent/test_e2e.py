import time
from pathlib import Path

import pytest

from evals.llm_judge import JudgeCriteria, LLMJudge
from evals.utils import load_yaml_fixtures, save_test_report
from src.agent.domain import events

current_path = Path(__file__).parent
# Load fixtures from YAML file
fixtures = load_yaml_fixtures(current_path, "e2e")


class TestEvalE2E:
    """End-to-End evaluation tests."""

    def setup_method(self):
        """Initialize LLM Judge for evaluation."""
        self.judge = LLMJudge()

    def setup_class(self):
        """Setup report file."""
        self.results = []

    def teardown_class(self):
        """Save results to report file."""
        save_test_report(self.results, "tool_e2e")

    def extract_final_response(self, session_id: str, test_notifications) -> str:
        """Extract the final response from collected notifications."""
        # Get all events sent to this session
        session_events = test_notifications.sent.get(session_id, [])

        # Find the final response event
        for event in reversed(session_events):  # Check from most recent
            if isinstance(event, events.Response):
                return event.response

        return ""

    @pytest.mark.parametrize(
        "fixture_name, fixture",
        [
            pytest.param(fixture_name, fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_eval_e2e(self, fixture_name, fixture, test_client, test_notifications):
        """Run E2E test with optional LLM judge evaluation."""

        # Clear all previous notifications
        # Since sent is a defaultdict, we need to clear its contents properly
        for key in list(test_notifications.sent.keys()):
            del test_notifications.sent[key]

        # Extract test data - fixture is now the test data directly
        question = fixture["question"]
        expected_response = fixture["response"]
        session_id = f"test-{fixture_name}"

        # Start timing
        start_time = time.time()

        # Make API request
        params = {"question": question, "q_id": fixture_name}
        headers = {"X-Session-ID": session_id}
        response = test_client.get("/answer", params=params, headers=headers)

        # API should return processing status
        assert response.status_code == 200
        assert response.json()["status"] == "processing"

        # Wait for async processing to complete
        # You may need to adjust this based on typical processing time
        max_wait_time = 30  # Maximum wait time in seconds
        wait_interval = 1  # Check interval in seconds
        elapsed_time = 0

        while elapsed_time < max_wait_time:
            # Check if we have received a response event
            session_events = test_notifications.sent.get(session_id, [])
            has_response = any(
                isinstance(event, events.Response) or hasattr(event, "response")
                for event in session_events
            )

            if has_response:
                break

            time.sleep(wait_interval)
            elapsed_time += wait_interval

        # Extract actual response from collected notifications
        actual_response = self.extract_final_response(session_id, test_notifications)

        # Calculate execution time
        execution_time_ms = int((time.time() - start_time) * 1000)

        # Add delay to avoid rate limiting (E2E makes many API calls internally)
        time.sleep(55)  # Reduced since we already waited 5 seconds

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

        # Record result
        result = {
            "test_name": fixture_name,
            "question": question,
            "expected": str(expected_response),
            "actual": str(actual_response),
            "passed": judge_result.passed,
            "execution_time_ms": execution_time_ms,
            "overall_score": (
                judge_result.scores.accuracy
                + judge_result.scores.relevance
                + judge_result.scores.completeness
                + judge_result.scores.hallucination
            )
            / 4,
            "accuracy": judge_result.scores.accuracy,
            "relevance": judge_result.scores.relevance,
            "completeness": judge_result.scores.completeness,
            "hallucination": judge_result.scores.hallucination,
            "judge_assessment": judge_result.overall_assessment,
        }
        self.__class__.results.append(result)

        # Assert judge passed
        assert judge_result.passed, f"Judge failed: {judge_result.overall_assessment}"
