import time
from pathlib import Path

import pytest

from evals.utils import load_yaml_fixtures, save_test_report
from src.agent import config
from src.agent.adapters import rag

current_path = Path(__file__).parent
# Load fixtures from YAML file
fixtures = load_yaml_fixtures(current_path, "")
rag_adapter = rag.BaseRAG(config.get_rag_config())


class TestIR:
    """Information Retrieval evaluation tests."""

    def setup_class(self):
        """Setup report file."""
        self.results = []

    def teardown_class(self):
        """Save results to report file."""
        save_test_report(self.results, "ir")

    @pytest.mark.parametrize(
        "fixture_name, fixture",
        [
            pytest.param(fixture_name, fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_ir(self, fixture_name, fixture):
        """Run IR test with optional LLM judge evaluation."""

        # Extract test data - fixture is now the test data directly
        question = fixture["question"]
        expected_response = fixture["response"]

        # Start timing
        start_time = time.time()

        # Execute retrieval and ranking
        response = rag_adapter.embed(question)
        response = rag_adapter.retrieve(response["embedding"])

        candidates = []
        if response and "results" in response:
            for candidate in response["results"]:
                temp = rag_adapter.rerank(question, candidate["description"])
                candidate["score"] = temp["score"] if temp else 0.0
                candidates.append(candidate)

        candidates = sorted(candidates, key=lambda x: -x["score"])

        # Calculate execution time
        execution_time_ms = int((time.time() - start_time) * 1000)

        # Extract top result - handle both string and dict expected responses
        if isinstance(expected_response, dict):
            # For dict responses, we should return the top candidate as a dict
            actual_response = candidates[0] if candidates else {}
        else:
            # For string responses (empty string case), return name or empty string
            actual_response = candidates[0]["name"] if candidates else ""

        # Add delay to avoid rate limiting
        time.sleep(1)

        # Record result
        result = {
            "test": fixture_name,
            "question": question,
            "expected": expected_response,
            "actual": actual_response,
            "passed": actual_response == expected_response,
            "execution_time_ms": execution_time_ms,
        }
        self.__class__.results.append(result)

        # Simple assert for exact match
        assert actual_response == expected_response
