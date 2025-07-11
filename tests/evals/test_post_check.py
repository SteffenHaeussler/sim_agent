import time
import uuid
from pathlib import Path

import pytest

from src.agent.adapters.llm import LLM
from src.agent.config import get_agent_config, get_llm_config
from src.agent.domain import commands, model
from tests.utils import get_fixtures
from tests.evals.base_eval_db import BaseEvaluationTest

current_path = Path(__file__).parent
fixtures = get_fixtures(current_path, keys=["post_check"])


class TestEvalPostCheck(BaseEvaluationTest):
    """Post-check guardrails evaluation tests."""

    RUN_TYPE = "post_check"
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
        """Run post-check guardrails test with optional LLM judge evaluation."""

        # Extract test data
        test_data = fixture["post_check"]
        question_text = test_data["question"]
        response_text = test_data["response"]
        memory = test_data["memory"]
        expected_response = test_data["approved"]

        q_id = str(uuid.uuid4())
        question = commands.Question(question=question_text, q_id=q_id)

        llm = LLM(get_llm_config())
        agent = model.BaseAgent(
            question=question,
            kwargs=get_agent_config(),
        )

        # Create LLMResponse command for post-check
        llm_response = commands.LLMResponse(
            question=question_text,
            q_id=q_id,
            response=response_text,
        )

        # Start timing
        start_time = time.time()

        # Prepare post-check
        prompt = agent.create_prompt(llm_response, memory)
        response = llm.use(prompt, response_model=commands.GuardrailPostCheckModel)

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
            test_data=test_data,
            execution_time_ms=execution_time_ms,
            metadata={
                "check_type": "post_check",
                "response_being_checked": response_text,
                "memory": memory,
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
