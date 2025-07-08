import json
import os
import time
from pathlib import Path
from typing import Dict, List

import pytest

from src.agent import config
from src.agent.adapters import agent_tools
from tests.utils import get_fixtures

# Import judge components only if enabled
USE_LLM_JUDGE = os.getenv("USE_LLM_JUDGE", "true").lower() == "true"
if USE_LLM_JUDGE:
    from tests.evals.llm_judge import JudgeCriteria, JudgeResult, LLMJudge

current_path = Path(__file__).parent

fixtures = get_fixtures(current_path, keys=["tool_agent"])
results: List[Dict] = []
judge_results: Dict[str, "JudgeResult"] = {}

tools = agent_tools.Tools(config.get_tools_config())

# Create judge if enabled
judge = LLMJudge() if USE_LLM_JUDGE else None


class TestEvalPlanning:
    @pytest.mark.parametrize(
        "fixture_name, fixture",
        [
            pytest.param(fixture_name, fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_eval_tool_agent(self, fixture_name, fixture):
        """Run tool agent test with optional LLM judge evaluation."""

        # Extract test data
        test_data = fixture["tool_agent"]
        question = test_data["question"]
        expected_response = test_data["response"]

        # Execute tool agent
        response, _ = tools.use(question)

        # Add delay to avoid rate limiting (tool agent makes many small API calls)
        time.sleep(60)

        if isinstance(response, list):
            response = sorted(response)

        # Create base report
        report = {
            "test_id": fixture_name,
            "question": question,
            "expected_response": expected_response,
            "actual_response": response,
        }

        # Handle different response types
        passed = False
        if isinstance(expected_response, dict) and "plot" in expected_response:
            passed = len(response) > 0
        elif isinstance(expected_response, dict) and "comparison" in expected_response:
            passed = len(response) > 0
        else:
            passed = expected_response in response

        # If judge is enabled, use it for evaluation
        if USE_LLM_JUDGE and judge is not None:
            # Extract judge criteria if present, otherwise use defaults
            if "judge_criteria" in test_data:
                criteria = JudgeCriteria(**test_data["judge_criteria"])
            else:
                criteria = JudgeCriteria()

            # Format responses for judge evaluation
            expected_str = (
                json.dumps(expected_response)
                if isinstance(expected_response, dict)
                else str(expected_response)
            )
            actual_str = (
                json.dumps(response)
                if isinstance(response, (list, dict))
                else str(response)
            )

            # Use LLM Judge to evaluate the response
            judge_result = judge.evaluate(
                question=question,
                expected=expected_str,
                actual=actual_str,
                criteria=criteria,
                test_type="tool_agent",
            )

            # Add delay after judge evaluation to avoid rate limiting
            time.sleep(1)

            # Add judge results to report
            report["judge_result"] = judge_result.model_dump()
            report["passed"] = judge_result.passed
            judge_results[fixture_name] = judge_result

            # Write results with judge information
            results.append(report)
            with open("tool_agent_judge_report.json", "w") as f:
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
            report["passed"] = passed
            results.append(report)
            with open("tool_agent_report.json", "w") as f:
                json.dump(results, f)

            assert passed, f"Test failed: expected {expected_response}, got {response}"

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
            "test_type": "tool_agent",
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
        with open("tool_agent_judge_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("\nTool Agent Test Summary with LLM Judge:")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Pass Rate: {summary['pass_rate']}")
        print("\nAverage Scores:")
        for metric, score in summary["average_scores"].items():
            print(f"  {metric.capitalize()}: {score}/10")
