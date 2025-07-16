import time
import uuid
from pathlib import Path

import pytest

from evals.llm_judge import JudgeCriteria, LLMJudge
from evals.utils import load_yaml_fixtures, save_test_report
from src.agent.adapters.llm import LLM
from src.agent.config import get_agent_config, get_llm_config
from src.agent.domain import commands, model

current_path = Path(__file__).parent
# Load fixtures from YAML file
fixtures = load_yaml_fixtures(current_path, "enhance")


class TestEvalEnhance:
    """Enhancement evaluation tests."""

    def setup_method(self):
        """Initialize LLM Judge for evaluation."""
        self.judge = LLMJudge()

    def setup_class(self):
        """Setup report file."""
        self.results = []

    def teardown_class(self):
        """Save results to report file."""
        save_test_report(self.results, "enhance")

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

        # Use LLM Judge for evaluation
        criteria = JudgeCriteria(**fixture.get("judge_criteria", {}))
        judge_result = self.judge.evaluate(
            question=question_text,
            expected=str(expected_response),
            actual=str(actual_response),
            criteria=criteria,
            test_type="enhance",
        )

        # Record result
        result = {
            "test_name": fixture_name,
            "question": question_text,
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
