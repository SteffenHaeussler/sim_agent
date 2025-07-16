import time
import uuid
from pathlib import Path

import pytest

from src.agent.adapters.llm import LLM
from src.agent.config import get_agent_config, get_llm_config
from src.agent.domain import commands, model
from evals.base_eval import BaseEvaluationTest
from evals.utils import load_yaml_fixtures

current_path = Path(__file__).parent
# Load fixtures from YAML file
fixtures = load_yaml_fixtures(current_path, "")


class TestEvalPreCheck(BaseEvaluationTest):
    """Pre-check guardrails evaluation tests."""

    RUN_TYPE = "pre_check"
    TEST_TYPE = "guardrails"
    EVALUATION_CATEGORY = "guardrails"

    @pytest.mark.parametrize(
        "fixture_name, fixture",
        [
            pytest.param(fixture_name, fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_eval_guardrails(self, fixture_name, fixture):
        """Run pre-check guardrails test with optional LLM judge evaluation."""

        # Extract test data - fixture is now the test data directly
        question_text = fixture["question"]
        expected_response = fixture["approved"]

        q_id = str(uuid.uuid4())
        question = commands.Question(question=question_text, q_id=q_id)

        llm = LLM(get_llm_config())
        agent = model.BaseAgent(
            question=question,
            kwargs=get_agent_config(),
        )

        # Start timing
        start_time = time.time()

        # Prepare guardrails check
        check = agent.prepare_guardrails_check(question)
        response = llm.use(
            check.question, response_model=commands.GuardrailPreCheckModel
        )

        # Calculate execution time
        execution_time_ms = int((time.time() - start_time) * 1000)

        # Add delay to avoid rate limiting
        time.sleep(1)

        # Extract response
        actual_response = response.approved
        parsed_response = response.model_dump()

        # Evaluate with judge and record to database
        self.evaluate_with_judge(
            fixture_name=fixture_name,
            question=question_text,
            expected_response=expected_response,
            actual_response=actual_response,
            test_data=fixture,
            execution_time_ms=execution_time_ms,
            metadata={
                "check_type": "pre_check",
                "response_json": parsed_response
                if "parsed_response" in locals()
                else None,
            },
        )

    @classmethod
    def teardown_class(cls):
        """Generate summary report after all tests."""
        # Call parent teardown which handles database completion and summary
        super().teardown_class()
