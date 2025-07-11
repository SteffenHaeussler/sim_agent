"""Base classes and utilities for SQL evaluation tests."""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.agent.domain import commands
from tests.evals.llm_judge import JudgeCriteria, JudgeResult, LLMJudge


USE_LLM_JUDGE = os.getenv("USE_LLM_JUDGE", "true").lower() == "true"
RATE_LIMIT_DELAY = 1  # seconds between LLM calls


class BaseSQLEvalTest:
    """Base class for SQL evaluation tests with common functionality."""

    # Class-level storage for results across all test instances
    _class_results: Dict[str, List[Dict]] = {}
    _class_judge_results: Dict[str, Dict[str, JudgeResult]] = {}

    def setup_method(self):
        """Setup method called before each test method."""
        self.results = BaseSQLEvalTest._class_results
        self.judge_results = BaseSQLEvalTest._class_judge_results
        self.judge = LLMJudge() if USE_LLM_JUDGE else None
        self.current_path = Path(__file__).parent

    @staticmethod
    def create_grounding_result(
        grounding_tables: List[str], grounding_columns: List[str]
    ) -> commands.GroundingResponse:
        """Create a GroundingResponse from table and column lists.

        Args:
            grounding_tables: List of table names
            grounding_columns: List of column names (can be "table.column" format)

        Returns:
            GroundingResponse with table and column mappings
        """
        # Create table mappings
        table_mappings = [
            commands.TableMapping(question_term=table, table_name=table, confidence=0.9)
            for table in grounding_tables
        ]

        # Create column mappings
        column_mappings = []
        for col in grounding_columns:
            parts = col.split(".", 1)
            table_name = (
                parts[0]
                if len(parts) > 1
                else grounding_tables[0]
                if grounding_tables
                else "unknown"
            )
            column_name = parts[1] if len(parts) > 1 else col

            column_mappings.append(
                commands.ColumnMapping(
                    question_term=column_name,
                    table_name=table_name,
                    column_name=column_name,
                    confidence=0.9,
                )
            )

        return commands.GroundingResponse(
            table_mapping=table_mappings,
            column_mapping=column_mappings,
        )

    def evaluate_with_judge(
        self,
        stage_name: str,
        fixture_name: str,
        question: str,
        expected_response: Dict[str, Any],
        actual_response_dict: Dict[str, Any],
        test_data: Dict[str, Any],
        judge_question: str,
    ) -> Dict[str, Any]:
        """Common pattern for judge evaluation and report generation.

        Args:
            stage_name: Name of the SQL stage being tested
            fixture_name: Name of the test fixture
            question: The original question
            expected_response: Expected response dict
            actual_response_dict: Actual response dict
            test_data: Test data containing judge criteria
            judge_question: Question to ask the judge

        Returns:
            Test report dictionary
        """
        # Create base report
        report = {
            "test_id": fixture_name,
            "question": question,
            "expected_response": expected_response,
            "actual_response": actual_response_dict,
        }

        # Initialize stage results if needed
        if stage_name not in self.results:
            self.results[stage_name] = []
        if stage_name not in self.judge_results:
            self.judge_results[stage_name] = {}

        # Evaluate with judge if enabled
        if USE_LLM_JUDGE and self.judge is not None:
            criteria = JudgeCriteria(**test_data.get("judge_criteria", {}))

            expected_str = json.dumps(expected_response, indent=2)
            actual_str = json.dumps(actual_response_dict, indent=2)

            judge_result = self.judge.evaluate(
                question=judge_question,
                expected=expected_str,
                actual=actual_str,
                criteria=criteria,
                test_type=f"sql_{stage_name}" if stage_name != "e2e" else "sql_e2e",
            )

            time.sleep(RATE_LIMIT_DELAY)

            # Add judge results to report
            report["judge_result"] = judge_result.model_dump()
            report["passed"] = judge_result.passed
            self.judge_results[stage_name][fixture_name] = judge_result

            # Append to results and write report
            self.results[stage_name].append(report)
            report_filename = (
                f"sql_{stage_name}_judge_report.json"
                if stage_name != "e2e"
                else "sql_e2e_judge_report.json"
            )
            self.write_report(
                self.current_path / "reports" / report_filename,
                self.results[stage_name],
            )

            # Assert based on judge evaluation
            self.assert_judge_passed(judge_result)
        else:
            # Basic validation without judge
            report["passed"] = actual_response_dict == expected_response
            self.results[stage_name].append(report)
            report_filename = (
                f"sql_{stage_name}_report.json"
                if stage_name != "e2e"
                else "sql_e2e_report.json"
            )
            self.write_report(
                self.current_path / "reports" / report_filename,
                self.results[stage_name],
            )

        return report

    @staticmethod
    def assert_judge_passed(judge_result: JudgeResult):
        """Common assertion for judge results."""
        assert judge_result.passed, (
            f"LLM Judge evaluation failed:\n"
            f"Scores: {judge_result.scores.model_dump()}\n"
            f"Reasoning: {judge_result.reasoning}\n"
            f"Assessment: {judge_result.overall_assessment}"
        )

    @staticmethod
    def write_report(report_path: Path, data: Any):
        """Write report to file, creating directory if needed."""
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(data, f, indent=2)

    def generate_stage_summary(
        self, stage_name: str, stage_results: Dict[str, JudgeResult]
    ) -> Optional[Dict[str, Any]]:
        """Generate summary for a single stage.

        Args:
            stage_name: Name of the stage
            stage_results: Dictionary of test results

        Returns:
            Summary dictionary or None if no results
        """
        if not stage_results:
            return None

        total_tests = len(stage_results)
        passed_tests = sum(1 for r in stage_results.values() if r.passed)

        # Calculate average scores
        score_types = ["accuracy", "relevance", "completeness", "hallucination"]
        avg_scores = {
            score: round(
                sum(getattr(r.scores, score) for r in stage_results.values())
                / total_tests,
                2,
            )
            for score in score_types
        }

        summary = {
            "test_type": f"sql_{stage_name}" if stage_name != "e2e" else "sql_e2e",
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "pass_rate": f"{(passed_tests / total_tests * 100):.1f}%",
            "average_scores": avg_scores,
            "failed_tests_details": [
                {
                    "test_id": test_id,
                    "scores": result.scores.model_dump(),
                    "assessment": result.overall_assessment,
                }
                for test_id, result in stage_results.items()
                if not result.passed
            ],
        }

        # Write summary
        summary_filename = (
            f"sql_{stage_name}_judge_summary.json"
            if stage_name != "e2e"
            else "sql_e2e_judge_summary.json"
        )
        self.write_report(self.current_path / "reports" / summary_filename, summary)

        # Print summary
        print(
            f"\nSQL {stage_name.upper() if stage_name == 'e2e' else stage_name.capitalize()} Test Summary with LLM Judge:"
        )
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Pass Rate: {summary['pass_rate']}")
        print("\nAverage Scores:")
        for metric, score in summary["average_scores"].items():
            print(f"  {metric.capitalize()}: {score}/10")

        return summary


# Command handler mapping for E2E tests
SQL_COMMAND_HANDLERS = {
    commands.SQLCheck: "check",
    commands.SQLGrounding: "grounding",
    commands.SQLFilter: "filter",
    commands.SQLJoinInference: "join_inference",
    commands.SQLAggregation: "aggregation",
    commands.SQLConstruction: "construction",
}
