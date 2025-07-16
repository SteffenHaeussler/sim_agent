import time
import uuid
from pathlib import Path

import pytest

from evals.utils import load_yaml_fixtures, save_test_report
from src.agent.adapters.llm import LLM
from src.agent.config import get_agent_config, get_llm_config
from src.agent.domain import commands, model

current_path = Path(__file__).parent
# Load fixtures from YAML file
fixtures = load_yaml_fixtures(current_path, "post_check")


class TestEvalPostCheck:
    """Post-check guardrails evaluation tests."""

    def setup_class(self):
        """Setup report file."""
        self.results = []

    def teardown_class(self):
        """Save results to report file."""
        save_test_report(self.results, "post_check")

    @pytest.mark.parametrize(
        "fixture_name, fixture",
        [
            pytest.param(fixture_name, fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_eval_guardrails(self, fixture_name, fixture):
        """Run post-check guardrails test with optional LLM judge evaluation."""

        # Extract test data - fixture is now the test data directly
        question_text = fixture["question"]
        response_text = fixture["response"]
        memory = fixture["memory"]
        expected_response = fixture["approved"]

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

        # Record result
        result = {
            "test_name": fixture_name,
            "question": question_text,
            "expected": expected_response,
            "actual": actual_response,
            "passed": actual_response == expected_response,
            "execution_time_ms": execution_time_ms,
        }
        self.__class__.results.append(result)

        # Simple assert for exact match
        assert actual_response == expected_response
