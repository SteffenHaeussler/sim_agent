import json
import os
import uuid
from pathlib import Path
from typing import Dict, List

import pytest

from src.agent.adapters.llm import LLM
from src.agent.config import get_agent_config, get_llm_config
from src.agent.domain import commands, model
from tests.utils import get_fixtures

# Import judge components only if enabled
USE_LLM_JUDGE = os.getenv("USE_LLM_JUDGE", "true").lower() == "true"
if USE_LLM_JUDGE:
    from tests.evals.llm_judge import JudgeCriteria, JudgeResult, LLMJudge

current_path = Path(__file__).parent

fixtures = get_fixtures(current_path, keys=["pre_check"])
results: List[Dict] = []
judge_results: Dict[str, "JudgeResult"] = {}

# Create judge if enabled
judge = LLMJudge() if USE_LLM_JUDGE else None


class TestEvalGuardrails:
    @pytest.mark.parametrize(
        "fixture_name, fixture",
        [
            pytest.param(fixture_name, fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_eval_guardrails(self, fixture_name, fixture):
        question, expected_response = (
            fixture["pre_check"]["question"],
            fixture["pre_check"]["approved"],
        )
        q_id = str(uuid.uuid4())
        question = commands.Question(question=question, q_id=q_id)

        llm = LLM(get_llm_config())
        agent = model.BaseAgent(
            question=question,
            kwargs=get_agent_config(),
        )

        command = agent.update(question)

        response = llm.use(
            command.question, response_model=commands.GuardrailPreCheckModel
        )
        actual_approved = response.approved

        # Create base report
        report = {
            "test_id": fixture_name,
            "question": question.question,
            "expected_approved": expected_response,
            "actual_approved": actual_approved,
        }

        # If judge is enabled, use it for evaluation
        if USE_LLM_JUDGE and judge is not None:
            # Extract judge criteria if present, otherwise use defaults
            test_data = fixture["pre_check"]
            if "judge_criteria" in test_data:
                criteria = JudgeCriteria(**test_data["judge_criteria"])
            else:
                criteria = JudgeCriteria()

            # Use LLM Judge to evaluate the guardrail decision
            # For pre-check, we evaluate if the guardrail made the correct decision
            judge_result = judge.evaluate(
                question=f"Should the guardrail approve this question?\nQuestion: {question.question}",
                expected=f"Guardrail Decision: Approved={expected_response}",
                actual=f"Guardrail Decision: Approved={actual_approved}",
                criteria=criteria,
                test_type="pre_check",
            )

            # Add judge results to report
            report["judge_result"] = judge_result.model_dump()
            report["passed"] = judge_result.passed
            judge_results[fixture_name] = judge_result

            # Write results with judge information
            results.append(report)
            with open(
                current_path / "reports" / "pre_check_judge_report.json", "w"
            ) as f:
                json.dump(results, f, indent=2)

            # Assert based on judge evaluation
            assert judge_result.passed, (
                f"LLM Judge evaluation failed:\n"
                f"Scores: {judge_result.scores.model_dump()}\n"
                f"Reasoning: {judge_result.reasoning}\n"
                f"Assessment: {judge_result.overall_assessment}"
            )
        else:
            # Standard evaluation without judge
            passed = actual_approved == expected_response
            report["passed"] = passed
            results.append(report)
            with open(current_path / "reports" / "pre_check_report.json", "w") as f:
                json.dump(results, f)

            assert passed, (
                f"Test failed: expected approved={expected_response}, got approved={actual_approved}"
            )

    @classmethod
    def teardown_class(cls):
        """Generate summary report after all tests."""

        # Only generate judge summary if judge was used
        if not USE_LLM_JUDGE or not judge_results:
            return

        # Calculate aggregate metrics
        total_tests = len(judge_results)
        passed_tests = sum(1 for r in judge_results.values() if r.passed)

        # Calculate average scores
        avg_accuracy = (
            sum(r.scores.accuracy for r in judge_results.values()) / total_tests
        )
        avg_relevance = (
            sum(r.scores.relevance for r in judge_results.values()) / total_tests
        )
        avg_completeness = (
            sum(r.scores.completeness for r in judge_results.values()) / total_tests
        )
        avg_hallucination = (
            sum(r.scores.hallucination for r in judge_results.values()) / total_tests
        )

        summary = {
            "test_type": "pre_check",
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "pass_rate": f"{(passed_tests / total_tests * 100):.1f}%",
            "average_scores": {
                "accuracy": round(avg_accuracy, 2),
                "relevance": round(avg_relevance, 2),
                "completeness": round(avg_completeness, 2),
                "hallucination": round(avg_hallucination, 2),
            },
            "failed_tests_details": [
                {
                    "test_id": test_id,
                    "scores": result.scores.model_dump(),
                    "assessment": result.overall_assessment,
                }
                for test_id, result in judge_results.items()
                if not result.passed
            ],
        }

        # Write summary report
        with open(current_path / "reports" / "pre_check_judge_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("\nPre-Check Test Summary with LLM Judge:")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Pass Rate: {summary['pass_rate']}")
        print("\nAverage Scores:")
        for metric, score in summary["average_scores"].items():
            print(f"  {metric.capitalize()}: {score}/10")
