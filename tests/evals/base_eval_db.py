"""Base evaluation class with database integration."""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from uuid import UUID

from loguru import logger

from src.agent.adapters.evaluation_repository import EvaluationRepository
from src.agent.domain.evaluation_model import TestResult, JudgeScores
from src.agent.config import get_llm_config
from tests.evals.llm_judge import JudgeCriteria, JudgeResult, LLMJudge


USE_LLM_JUDGE = os.getenv("USE_LLM_JUDGE", "true").lower() == "true"
RATE_LIMIT_DELAY = 1  # seconds between LLM calls


class BaseEvaluationTest:
    """Base class for evaluation tests with database integration."""

    # Class-level storage for run information
    _current_run_id: Optional[UUID] = None
    _repository: Optional[EvaluationRepository] = None
    _fixtures_used: List[str] = []

    def setup_method(self):
        """Setup method called before each test method."""
        self.judge = LLMJudge() if USE_LLM_JUDGE else None
        self.current_path = Path(__file__).parent

        # Initialize repository if not already done
        if BaseEvaluationTest._repository is None:
            BaseEvaluationTest._repository = EvaluationRepository()

    @classmethod
    def setup_class(cls):
        """Setup class - create evaluation run."""
        # Initialize repository
        cls._repository = EvaluationRepository()

        # Get run type from class name or override
        run_type = getattr(cls, "RUN_TYPE", cls.__name__.lower())
        evaluation_category = getattr(cls, "EVALUATION_CATEGORY", None)
        stage = getattr(cls, "STAGE", None)

        # Get model configuration
        llm_config = get_llm_config()
        model_name = llm_config.get("model_id", "unknown")
        model_temperature = float(llm_config.get("temperature", 0.0))

        # Create evaluation run
        run = cls._repository.create_evaluation_run(
            run_type=run_type,
            evaluation_category=evaluation_category,
            stage=stage,
            model_name=model_name,
            model_temperature=model_temperature,
            metadata={"test_class": cls.__name__, "use_llm_judge": USE_LLM_JUDGE},
        )

        cls._current_run_id = run.id
        cls._fixtures_used = []
        logger.info(f"Created evaluation run {run.id} for {run_type}")

    @classmethod
    def teardown_class(cls):
        """Teardown class - complete evaluation run and generate summary."""
        if cls._current_run_id and cls._repository:
            # Complete the run
            cls._repository.complete_evaluation_run(
                cls._current_run_id, fixtures_used=cls._fixtures_used
            )

            # Retrieve and display summary
            run = cls._repository.get_evaluation_run(cls._current_run_id)
            if run:
                summary = run.to_summary_dict()

                # Print summary
                print(f"\n{cls.__name__} Summary:")
                print(f"Total Tests: {summary['total_tests']}")
                print(f"Passed: {summary['passed_tests']}")
                print(f"Failed: {summary['failed_tests']}")
                print(f"Pass Rate: {summary['pass_rate']}")

                if "average_scores" in summary and summary["average_scores"]:
                    print("\nAverage Scores:")
                    for metric, score in summary["average_scores"].items():
                        print(f"  {metric.capitalize()}: {score}/10")

                # Also write summary to JSON for compatibility
                report_dir = Path(__file__).parent / "reports"
                report_dir.mkdir(exist_ok=True)

                report_name = f"{run.run_type}_summary.json"
                import json

                with open(report_dir / report_name, "w") as f:
                    json.dump(summary, f, indent=2)

    def record_test_result(
        self,
        fixture_name: str,
        question: str,
        expected_response: Any,
        actual_response: Any,
        passed: bool,
        execution_time_ms: Optional[int] = None,
        judge_result: Optional[JudgeResult] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,  # For test-specific fields
    ) -> TestResult:
        """Record a test result to the database."""

        # Track fixture usage
        if fixture_name not in self._fixtures_used:
            self._fixtures_used.append(fixture_name)

        # Convert responses to strings
        if isinstance(expected_response, dict):
            expected_str = json.dumps(expected_response, indent=2)
        else:
            expected_str = str(expected_response)

        if isinstance(actual_response, dict):
            actual_str = json.dumps(actual_response, indent=2)
        else:
            actual_str = str(actual_response)

        # Create test result
        test_result = TestResult(
            test_name=fixture_name,
            test_type=getattr(self, "TEST_TYPE", None),
            question=question,
            expected_response=expected_str,
            actual_response=actual_str,
            passed=passed,
            execution_time_ms=execution_time_ms,
            judge_scores=JudgeScores(**judge_result.scores.model_dump())
            if judge_result
            else None,
            judge_reasoning=json.dumps(judge_result.reasoning)
            if judge_result and isinstance(judge_result.reasoning, dict)
            else judge_result.reasoning
            if judge_result
            else None,
            error_message=error_message,
            metadata=metadata,
            **kwargs,  # Pass through test-specific fields
        )

        # Save to database
        self._repository.add_test_result(self._current_run_id, test_result)

        # Also write individual result to JSON for compatibility
        self._write_json_report(fixture_name, test_result, judge_result)

        return test_result

    def _write_json_report(
        self,
        fixture_name: str,
        test_result: TestResult,
        judge_result: Optional[JudgeResult] = None,
    ):
        """Write JSON report for backwards compatibility."""
        report_dir = self.current_path / "reports"
        report_dir.mkdir(exist_ok=True)

        # Build report dict
        report = {
            "test_id": fixture_name,
            "question": test_result.question,
            "expected_response": test_result.expected_response,
            "actual_response": test_result.actual_response,
            "passed": test_result.passed,
        }

        if test_result.execution_time_ms:
            report["execution_time_ms"] = test_result.execution_time_ms

        if judge_result:
            report["judge_result"] = judge_result.model_dump()

        # Append to results file
        run_type = getattr(self, "RUN_TYPE", self.__class__.__name__.lower())
        report_file = report_dir / f"{run_type}_report.json"

        # Load existing results
        if report_file.exists():
            with open(report_file, "r") as f:
                results = json.load(f)
        else:
            results = []

        results.append(report)

        # Write back
        with open(report_file, "w") as f:
            json.dump(results, f, indent=2)

    def evaluate_with_judge(
        self,
        fixture_name: str,
        question: str,
        expected_response: Any,
        actual_response: Any,
        test_data: Dict[str, Any],
        judge_question: Optional[str] = None,
        execution_time_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> TestResult:
        """Evaluate a test with optional LLM judge and record to database."""

        if USE_LLM_JUDGE and self.judge is not None:
            # Extract judge criteria
            criteria = JudgeCriteria(**test_data.get("judge_criteria", {}))

            # Convert responses to strings for judge
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

            # Use LLM Judge
            judge_result = self.judge.evaluate(
                question=judge_question or question,
                expected=expected_str,
                actual=actual_str,
                criteria=criteria,
                test_type=getattr(self, "TEST_TYPE", "general"),
            )

            time.sleep(RATE_LIMIT_DELAY)

            # Record result with judge evaluation
            test_result = self.record_test_result(
                fixture_name=fixture_name,
                question=question,
                expected_response=expected_response,
                actual_response=actual_response,
                passed=judge_result.passed,
                execution_time_ms=execution_time_ms,
                judge_result=judge_result,
                metadata=metadata,
                **kwargs,
            )

            # Assert based on judge evaluation
            assert judge_result.passed, (
                f"LLM Judge evaluation failed:\n"
                f"Scores: {judge_result.scores.model_dump()}\n"
                f"Reasoning: {judge_result.reasoning}\n"
                f"Assessment: {judge_result.overall_assessment}"
            )
        else:
            # Simple pass/fail evaluation
            passed = actual_response == expected_response

            # Record result without judge
            test_result = self.record_test_result(
                fixture_name=fixture_name,
                question=question,
                expected_response=expected_response,
                actual_response=actual_response,
                passed=passed,
                execution_time_ms=execution_time_ms,
                metadata=metadata,
                **kwargs,
            )

            # Assert
            assert passed, f"Expected: {expected_response}, Got: {actual_response}"

        return test_result
