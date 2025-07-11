import time
from pathlib import Path

import pytest

from src.agent import config
from src.agent.adapters import rag
from tests.utils import get_fixtures
from tests.evals.base_eval_db import BaseEvaluationTest

current_path = Path(__file__).parent
fixtures = get_fixtures(current_path, keys=["ir"])
rag_adapter = rag.BaseRAG(config.get_rag_config())


class TestIR(BaseEvaluationTest):
    """Information Retrieval evaluation tests."""

    RUN_TYPE = "ir"
    TEST_TYPE = "information_retrieval"
    EVALUATION_CATEGORY = "retrieval"

    @pytest.mark.parametrize(
        "fixture_name, fixture",
        [
            pytest.param(fixture_name, fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_ir(self, fixture_name, fixture):
        """Run IR test with optional LLM judge evaluation."""

        # Extract test data
        test_data = fixture["ir"]
        question = test_data["question"]
        expected_response = test_data["response"]

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

        # Extract top result
        actual_response = candidates[0]["name"] if candidates else ""

        # Add delay to avoid rate limiting
        time.sleep(1)

        # Evaluate with judge and record to database
        self.evaluate_with_judge(
            fixture_name=fixture_name,
            question=question,
            expected_response=expected_response,
            actual_response=actual_response,
            test_data=test_data,
            execution_time_ms=execution_time_ms,
            metadata={
                "candidates_retrieved": len(candidates),
                "top_5_results": [r["name"] for r in candidates[:5]]
                if len(candidates) >= 5
                else [r["name"] for r in candidates],
                "top_scores": [r["score"] for r in candidates[:5]]
                if len(candidates) >= 5
                else [r["score"] for r in candidates],
            },
        )

    @classmethod
    def teardown_class(cls):
        """Generate summary report after all tests."""
        # Call parent teardown which handles database completion and summary
        super().teardown_class()
