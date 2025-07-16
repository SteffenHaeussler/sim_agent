"""Simplified base evaluation class - bare minimum functionality."""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

from evals.llm_judge import LLMJudge, JudgeCriteria


USE_LLM_JUDGE = os.getenv("USE_LLM_JUDGE", "true").lower() == "true"
RATE_LIMIT_DELAY = 1  # seconds between LLM calls


class BaseEvaluationTest:
    """Simplified base class for evaluation tests."""

    def setup_method(self):
        """Setup method called before each test method."""
        self.judge = LLMJudge() if USE_LLM_JUDGE else None
        self.report_dir = Path(__file__).parent / "reports"
        self.report_dir.mkdir(exist_ok=True)
        self.results = []

    @classmethod
    def teardown_class(cls):
        """Write summary after all tests."""
        # Simple summary printed to console
        test_name = getattr(cls, "RUN_TYPE", cls.__name__)
        print(f"\n{test_name} evaluation complete")

    def evaluate_with_judge(
        self,
        fixture_name: str,
        question: str,
        expected_response: Any,
        actual_response: Any,
        test_data: Dict[str, Any],
        **kwargs,
    ):
        """Evaluate a test with optional LLM judge."""

        # Convert responses to strings
        expected_str = (
            json.dumps(expected_response)
            if isinstance(expected_response, dict)
            else str(expected_response)
        )
        actual_str = (
            json.dumps(actual_response)
            if isinstance(actual_response, dict)
            else str(actual_response)
        )

        if USE_LLM_JUDGE and self.judge is not None:
            # Use LLM Judge
            criteria = JudgeCriteria(**test_data.get("judge_criteria", {}))
            judge_result = self.judge.evaluate(
                question=question,
                expected=expected_str,
                actual=actual_str,
                criteria=criteria,
                test_type=getattr(self, "TEST_TYPE", "general"),
            )

            time.sleep(RATE_LIMIT_DELAY)

            # Save result
            self._save_result(
                fixture_name=fixture_name,
                question=question,
                expected=expected_str,
                actual=actual_str,
                passed=judge_result.passed,
                judge_result=judge_result.model_dump(),
            )

            # Assert
            assert judge_result.passed, (
                f"Judge failed: {judge_result.overall_assessment}"
            )
        else:
            # Simple comparison
            passed = actual_response == expected_response

            # Save result
            self._save_result(
                fixture_name=fixture_name,
                question=question,
                expected=expected_str,
                actual=actual_str,
                passed=passed,
            )

            # Assert
            assert passed, f"Expected: {expected_response}, Got: {actual_response}"

    def _save_result(
        self,
        fixture_name: str,
        question: str,
        expected: str,
        actual: str,
        passed: bool,
        judge_result: Optional[Dict] = None,
    ):
        """Save result to JSON file."""
        result = {
            "test_id": fixture_name,
            "question": question,
            "expected": expected,
            "actual": actual,
            "passed": passed,
        }

        if judge_result:
            result["judge_result"] = judge_result

        # Append to report file
        run_type = getattr(self, "RUN_TYPE", self.__class__.__name__.lower())
        report_file = self.report_dir / f"{run_type}_report.json"

        # Load existing results
        if report_file.exists():
            with open(report_file, "r") as f:
                results = json.load(f)
        else:
            results = []

        results.append(result)

        # Write back
        with open(report_file, "w") as f:
            json.dump(results, f, indent=2)
