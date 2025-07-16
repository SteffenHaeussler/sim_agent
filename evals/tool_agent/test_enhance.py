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


class TestEvalEnhance(BaseEvaluationTest):
    """Enhancement evaluation tests."""

    RUN_TYPE = "enhance"
    TEST_TYPE = "enhance"
    EVALUATION_CATEGORY = "enhancement"

    @pytest.mark.parametrize(
        "fixture_name, fixture",
        [
            pytest.param(fixture_name, fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_eval_enhance(self, fixture_name, fixture):
        """Run enhancement test with optional LLM judge evaluation."""

        # Extract test data - fixture is now the test data directly
        question_text = fixture["question"]
        candidates = fixture["candidates"]
        expected_response = fixture["response"]

        q_id = str(uuid.uuid4())
        question = commands.Question(question=question_text, q_id=q_id)

        llm = LLM(get_llm_config())
        agent = model.BaseAgent(
            question=question,
            kwargs=get_agent_config(),
        )

        # Convert candidates to KBResponse objects
        kb_candidates = []
        for candidate in candidates:
            kb_candidate = commands.KBResponse(
                description=candidate.get("text", ""),
                score=candidate.get("score", 0.0),
                id=candidate.get("id", ""),
                tag=candidate.get("tag", ""),
                name=candidate.get("name", ""),
            )
            kb_candidates.append(kb_candidate)

        # Create Rerank command
        rerank_command = commands.Rerank(
            question=question_text,
            candidates=kb_candidates,
            q_id=q_id,
        )

        # Start timing
        start_time = time.time()

        # Prepare enhancement
        enhance_command = agent.prepare_enhancement(rerank_command)

        # Execute enhancement
        response = llm.use(
            enhance_command.question, response_model=commands.LLMResponseModel
        )
        actual_response = response.response

        # Calculate execution time
        execution_time_ms = int((time.time() - start_time) * 1000)

        # Add delay to avoid rate limiting
        time.sleep(1)

        # Evaluate with judge and record to database
        self.evaluate_with_judge(
            fixture_name=fixture_name,
            question=question_text,
            expected_response=expected_response,
            actual_response=actual_response,
            test_data=fixture,
            execution_time_ms=execution_time_ms,
            metadata={
                "candidates_count": len(candidates),
                "enhance_prompt": enhance_command.question[:200] + "..."
                if len(enhance_command.question) > 200
                else enhance_command.question,
            },
        )

    @classmethod
    def teardown_class(cls):
        """Generate summary report after all tests."""
        # Call parent teardown which handles database completion and summary
        super().teardown_class()
