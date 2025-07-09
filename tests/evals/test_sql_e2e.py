import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

import pytest

from src.agent.adapters.adapter import SQLAgentAdapter
from src.agent.config import get_agent_config
from src.agent.domain import commands
from src.agent.domain.sql_model import SQLBaseAgent
from tests.utils import get_fixtures

# Import judge components only if enabled
USE_LLM_JUDGE = os.getenv("USE_LLM_JUDGE", "true").lower() == "true"
if USE_LLM_JUDGE:
    from tests.evals.llm_judge import JudgeCriteria, JudgeResult, LLMJudge

current_path = Path(__file__).parent

fixtures = get_fixtures(current_path, keys=["sql_e2e"])
results: List[Dict] = []
judge_results: Dict[str, "JudgeResult"] = {}

# Create judge if enabled
judge = LLMJudge() if USE_LLM_JUDGE else None

# Load database schema
with open(current_path / "schema.json", "r") as f:
    schema_data = json.load(f)

# Create DatabaseSchema object
db_schema = commands.DatabaseSchema(**schema_data)


class TestEvalSQLEndToEnd:
    @pytest.mark.parametrize(
        "fixture_name, fixture",
        [
            pytest.param(fixture_name, fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_eval_sql_e2e(self, fixture_name, fixture):
        """Run SQL E2E test with optional LLM judge evaluation."""

        # Extract test data
        test_data = fixture["sql_e2e"]
        question = test_data["question"]
        expected_sql = test_data["expected_sql"]

        # Create SQL question command
        q_id = str(uuid.uuid4())
        sql_question = commands.SQLQuestion(question=question, q_id=q_id)

        # Initialize SQL adapter
        adapter = SQLAgentAdapter()

        # Mock the question method to return our schema
        def mock_question(command):
            command.schema_info = db_schema
            return command

        # Use patch to mock the question method
        with patch.object(adapter, "question", side_effect=mock_question):
            # Initialize SQL agent
            agent = SQLBaseAgent(
                question=sql_question,
                kwargs=get_agent_config(),
            )

            # Process the SQL question command to set up schema
            sql_question = adapter.question(sql_question)
            agent.construction = commands.SQLConstruction(
                question=question, q_id=q_id, schema_info=sql_question.schema_info
            )

            # Process through SQL pipeline
            current_command = sql_question

            # Run through the complete SQL generation pipeline
            while not isinstance(current_command, commands.SQLConstruction):
                current_command = agent.update(current_command)

                # Process each stage through the adapter
                if isinstance(current_command, commands.SQLCheck):
                    response = adapter.check(current_command)
                    agent.check_result = response
                elif isinstance(current_command, commands.SQLGrounding):
                    response = adapter.grounding(current_command)
                    agent.grounding_result = response
                elif isinstance(current_command, commands.SQLFilter):
                    response = adapter.filter(current_command)
                    agent.filter_result = response
                elif isinstance(current_command, commands.SQLJoinInference):
                    response = adapter.join_inference(current_command)
                    agent.join_inference_result = response
                elif isinstance(current_command, commands.SQLAggregation):
                    response = adapter.aggregation(current_command)
                    agent.aggregation_result = response
                elif isinstance(current_command, commands.SQLConstruction):
                    response = adapter.construction(current_command)
                    agent.construction_result = response
                    break

        # Get the final SQL query
        actual_sql = (
            agent.construction_result.sql_query
            if hasattr(agent, "construction_result")
            else ""
        )

        # Add delay to avoid rate limiting
        time.sleep(1)

        # Create base report
        report = {
            "test_id": fixture_name,
            "question": question,
            "expected_sql": expected_sql,
            "actual_sql": actual_sql,
        }

        # If judge is enabled, use it for evaluation
        if USE_LLM_JUDGE and judge is not None:
            # Extract judge criteria if present, otherwise use defaults
            if "judge_criteria" in test_data:
                criteria = JudgeCriteria(**test_data["judge_criteria"])
            else:
                criteria = JudgeCriteria()

            # Use LLM Judge to evaluate the SQL query
            judge_result = judge.evaluate(
                question=question,
                expected=expected_sql,
                actual=actual_sql,
                criteria=criteria,
                test_type="sql_e2e",
            )

            # Add delay after judge evaluation to avoid rate limiting
            time.sleep(1)

            # Add judge results to report
            report["judge_result"] = judge_result.model_dump()
            report["passed"] = judge_result.passed
            judge_results[fixture_name] = judge_result

            # Write results with judge information
            results.append(report)
            with open(current_path / "reports" / "sql_e2e_judge_report.json", "w") as f:
                json.dump(results, f, indent=2)

            # Assert based on judge evaluation
            assert judge_result.passed, (
                f"LLM Judge evaluation failed:\n"
                f"Scores: {judge_result.scores.model_dump()}\n"
                f"Reasoning: {judge_result.reasoning}\n"
                f"Assessment: {judge_result.overall_assessment}"
            )
        else:
            # Standard evaluation without judge - exact match
            passed = actual_sql.strip() == expected_sql.strip()
            report["passed"] = passed
            results.append(report)
            with open(current_path / "reports" / "sql_e2e_report.json", "w") as f:
                json.dump(results, f, indent=2)

            assert passed, (
                f"SQL mismatch:\nExpected:\n{expected_sql}\n\nActual:\n{actual_sql}"
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
            "test_type": "sql_e2e",
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
        with open(current_path / "reports" / "sql_e2e_judge_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("\nSQL E2E Test Summary with LLM Judge:")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Pass Rate: {summary['pass_rate']}")
        print("\nAverage Scores:")
        for metric, score in summary["average_scores"].items():
            print(f"  {metric.capitalize()}: {score}/10")
