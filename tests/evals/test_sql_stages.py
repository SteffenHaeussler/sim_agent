import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

from src.agent.adapters.adapter import SQLAgentAdapter
from src.agent.domain import commands
from tests.utils import get_fixtures

# Import judge components only if enabled
USE_LLM_JUDGE = os.getenv("USE_LLM_JUDGE", "true").lower() == "true"
if USE_LLM_JUDGE:
    from tests.evals.llm_judge import JudgeCriteria, JudgeResult, LLMJudge

current_path = Path(__file__).parent

# Load fixtures for different SQL stages
fixtures = get_fixtures(current_path, keys=["sql_stages"])
results: Dict[str, List[Dict]] = {
    "grounding": [],
    "filter": [],
    "aggregation": [],
    "join": [],
}
judge_results: Dict[str, Dict[str, "JudgeResult"]] = {
    "grounding": {},
    "filter": {},
    "aggregation": {},
    "join": {},
}

# Create judge if enabled
judge = LLMJudge() if USE_LLM_JUDGE else None

# Load database schema
with open(current_path / "schema.json", "r") as f:
    schema_data = json.load(f)

# Create DatabaseSchema object
db_schema = commands.DatabaseSchema(**schema_data)


class TestEvalSQLStages:
    def setup_method(self):
        """Setup mock database schema for each test."""
        self.schema = db_schema
        self.adapter = SQLAgentAdapter()
        # Mock the database adapter
        self.adapter.db = MagicMock()
        self.adapter.db.get_schema.return_value = self.schema

    @pytest.mark.parametrize(
        "fixture_name, fixture",
        [
            pytest.param(fixture_name, fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
            if "sql_stages" in fixture and "grounding" in fixture["sql_stages"]
        ],
    )
    def test_eval_sql_grounding(self, fixture_name, fixture):
        """Test SQL grounding stage - identifying relevant tables and columns."""
        test_data = fixture["sql_stages"]["grounding"]
        question = test_data["question"]
        expected_response = test_data["expected_response"]

        # Create grounding command
        q_id = str(uuid.uuid4())
        command = commands.SQLGrounding(
            question=question,
            q_id=q_id,
            tables=self.schema.tables,
        )

        # Execute grounding
        actual_response = self.adapter.grounding(command)

        # Add delay to avoid rate limiting
        time.sleep(1)

        # Create base report
        report = {
            "test_id": fixture_name,
            "question": question,
            "expected_response": expected_response,
            "actual_response": {
                "tables": [tm.table_name for tm in actual_response.table_mapping],
                "columns": [
                    f"{cm.table_name}.{cm.column_name}"
                    for cm in actual_response.column_mapping
                ],
            },
        }

        # If judge is enabled, use it for evaluation
        if USE_LLM_JUDGE and judge is not None:
            # Extract judge criteria if present
            criteria = JudgeCriteria(**test_data.get("judge_criteria", {}))

            # Format for judge evaluation
            expected_str = json.dumps(expected_response, indent=2)
            actual_str = json.dumps(report["actual_response"], indent=2)

            # Use LLM Judge
            judge_result = judge.evaluate(
                question=f"Identify tables and columns for SQL query: {question}",
                expected=expected_str,
                actual=actual_str,
                criteria=criteria,
                test_type="sql_grounding",
            )

            time.sleep(1)

            # Add results
            report["judge_result"] = judge_result.model_dump()
            report["passed"] = judge_result.passed
            judge_results["grounding"][fixture_name] = judge_result

            results["grounding"].append(report)
            with open(
                current_path / "reports" / "sql_grounding_judge_report.json", "w"
            ) as f:
                json.dump(results["grounding"], f, indent=2)

            assert judge_result.passed, (
                f"LLM Judge evaluation failed:\n"
                f"Scores: {judge_result.scores.model_dump()}\n"
                f"Reasoning: {judge_result.reasoning}\n"
                f"Assessment: {judge_result.overall_assessment}"
            )
        else:
            # Basic validation
            assert actual_response.table_mapping
            assert actual_response.column_mapping

    @pytest.mark.parametrize(
        "fixture_name, fixture",
        [
            pytest.param(fixture_name, fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
            if "sql_stages" in fixture and "filter" in fixture["sql_stages"]
        ],
    )
    def test_eval_sql_filter(self, fixture_name, fixture):
        """Test SQL filter stage - extracting WHERE conditions."""
        test_data = fixture["sql_stages"]["filter"]
        question = test_data["question"]
        expected_response = test_data["expected_response"]

        # Mock grounding result
        grounding_tables = test_data.get("grounding_tables", ["products"])
        grounding_columns = test_data.get("grounding_columns", ["is_active", "name"])

        # Create table mappings
        table_mappings = []
        for table in grounding_tables:
            table_mappings.append(
                commands.TableMapping(
                    question_term=table, table_name=table, confidence=0.9
                )
            )

        # Create column mappings
        column_mappings = []
        for col in grounding_columns:
            table_name = col.split(".")[0] if "." in col else grounding_tables[0]
            column_name = col.split(".")[1] if "." in col else col
            column_mappings.append(
                commands.ColumnMapping(
                    question_term=column_name,
                    table_name=table_name,
                    column_name=column_name,
                    confidence=0.9,
                )
            )

        grounding_result = commands.GroundingResponse(
            table_mapping=table_mappings,
            column_mapping=column_mappings,
        )

        # Create filter command
        q_id = str(uuid.uuid4())
        command = commands.SQLFilter(
            question=question,
            q_id=q_id,
            column_mapping=grounding_result.column_mapping,
        )

        # Execute filter
        actual_response = self.adapter.filter(command)

        time.sleep(1)

        # Create report
        report = {
            "test_id": fixture_name,
            "question": question,
            "expected_response": expected_response,
            "actual_response": {
                "where_conditions": [
                    f"{cond.column} {cond.operator} {cond.value}"
                    for cond in actual_response.conditions
                ]
                if actual_response.conditions
                else [],
                "having_conditions": [],
            },
        }

        # Judge evaluation
        if USE_LLM_JUDGE and judge is not None:
            criteria = JudgeCriteria(**test_data.get("judge_criteria", {}))

            expected_str = json.dumps(expected_response, indent=2)
            actual_str = json.dumps(report["actual_response"], indent=2)

            judge_result = judge.evaluate(
                question=f"Extract WHERE conditions from: {question}",
                expected=expected_str,
                actual=actual_str,
                criteria=criteria,
                test_type="sql_filter",
            )

            time.sleep(1)

            report["judge_result"] = judge_result.model_dump()
            report["passed"] = judge_result.passed
            judge_results["filter"][fixture_name] = judge_result

            results["filter"].append(report)
            with open(
                current_path / "reports" / "sql_filter_judge_report.json", "w"
            ) as f:
                json.dump(results["filter"], f, indent=2)

            assert judge_result.passed, (
                f"LLM Judge evaluation failed:\n"
                f"Scores: {judge_result.scores.model_dump()}\n"
                f"Reasoning: {judge_result.reasoning}\n"
                f"Assessment: {judge_result.overall_assessment}"
            )

    @pytest.mark.parametrize(
        "fixture_name, fixture",
        [
            pytest.param(fixture_name, fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
            if "sql_stages" in fixture and "aggregation" in fixture["sql_stages"]
        ],
    )
    def test_eval_sql_aggregation(self, fixture_name, fixture):
        """Test SQL aggregation stage - detecting GROUP BY and aggregates."""
        test_data = fixture["sql_stages"]["aggregation"]
        question = test_data["question"]
        expected_response = test_data["expected_response"]

        # Mock grounding result
        grounding_tables = test_data.get("grounding_tables", ["orders", "customers"])
        grounding_columns = test_data.get(
            "grounding_columns", ["customer_id", "grand_total_amount", "company_name"]
        )

        # Create table mappings
        table_mappings = []
        for table in grounding_tables:
            table_mappings.append(
                commands.TableMapping(
                    question_term=table, table_name=table, confidence=0.9
                )
            )

        # Create column mappings
        column_mappings = []
        for col in grounding_columns:
            table_name = col.split(".")[0] if "." in col else grounding_tables[0]
            column_name = col.split(".")[1] if "." in col else col
            column_mappings.append(
                commands.ColumnMapping(
                    question_term=column_name,
                    table_name=table_name,
                    column_name=column_name,
                    confidence=0.9,
                )
            )

        grounding_result = commands.GroundingResponse(
            table_mapping=table_mappings,
            column_mapping=column_mappings,
        )

        # Create aggregation command
        q_id = str(uuid.uuid4())
        command = commands.SQLAggregation(
            question=question,
            q_id=q_id,
            column_mapping=grounding_result.column_mapping,
        )

        # Execute aggregation
        actual_response = self.adapter.aggregation(command)

        time.sleep(1)

        # Create report
        report = {
            "test_id": fixture_name,
            "question": question,
            "expected_response": expected_response,
            "actual_response": {
                "group_by_columns": actual_response.group_by_columns,
                "aggregate_functions": [
                    agg.model_dump() for agg in actual_response.aggregations
                ]
                if actual_response.aggregations
                else [],
                "requires_aggregation": actual_response.is_aggregation_query,
            },
        }

        # Judge evaluation
        if USE_LLM_JUDGE and judge is not None:
            criteria = JudgeCriteria(**test_data.get("judge_criteria", {}))

            expected_str = json.dumps(expected_response, indent=2)
            actual_str = json.dumps(report["actual_response"], indent=2)

            judge_result = judge.evaluate(
                question=f"Identify aggregation needs for: {question}",
                expected=expected_str,
                actual=actual_str,
                criteria=criteria,
                test_type="sql_aggregation",
            )

            time.sleep(1)

            report["judge_result"] = judge_result.model_dump()
            report["passed"] = judge_result.passed
            judge_results["aggregation"][fixture_name] = judge_result

            results["aggregation"].append(report)
            with open(
                current_path / "reports" / "sql_aggregation_judge_report.json", "w"
            ) as f:
                json.dump(results["aggregation"], f, indent=2)

            assert judge_result.passed, (
                f"LLM Judge evaluation failed:\n"
                f"Scores: {judge_result.scores.model_dump()}\n"
                f"Reasoning: {judge_result.reasoning}\n"
                f"Assessment: {judge_result.overall_assessment}"
            )

    @pytest.mark.parametrize(
        "fixture_name, fixture",
        [
            pytest.param(fixture_name, fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
            if "sql_stages" in fixture and "join" in fixture["sql_stages"]
        ],
    )
    def test_eval_sql_join(self, fixture_name, fixture):
        """Test SQL join inference stage - determining required joins."""
        test_data = fixture["sql_stages"]["join"]
        question = test_data["question"]
        expected_response = test_data["expected_response"]

        # Mock grounding result
        grounding_tables = test_data.get(
            "grounding_tables", ["customers", "orders", "order_details", "products"]
        )
        grounding_columns = test_data.get(
            "grounding_columns", ["customers.company_name", "products.name"]
        )

        # Create table mappings
        table_mappings = []
        for table in grounding_tables:
            table_mappings.append(
                commands.TableMapping(
                    question_term=table, table_name=table, confidence=0.9
                )
            )

        # Create column mappings
        column_mappings = []
        for col in grounding_columns:
            table_name = col.split(".")[0] if "." in col else grounding_tables[0]
            column_name = col.split(".")[1] if "." in col else col
            column_mappings.append(
                commands.ColumnMapping(
                    question_term=column_name,
                    table_name=table_name,
                    column_name=column_name,
                    confidence=0.9,
                )
            )

        grounding_result = commands.GroundingResponse(
            table_mapping=table_mappings,
            column_mapping=column_mappings,
        )

        # Create join inference command
        q_id = str(uuid.uuid4())
        command = commands.SQLJoinInference(
            question=question,
            q_id=q_id,
            table_mapping=grounding_result.table_mapping,
            relationships=self.schema.relationships,
        )

        # Execute join inference
        actual_response = self.adapter.join_inference(command)

        time.sleep(1)

        # Create report
        report = {
            "test_id": fixture_name,
            "question": question,
            "expected_response": expected_response,
            "actual_response": {
                "joins": [
                    {
                        "type": join.join_type,
                        "left_table": join.from_table,
                        "right_table": join.to_table,
                        "on_condition": f"{join.from_table}.{join.from_column} = {join.to_table}.{join.to_column}",
                    }
                    for join in actual_response.joins
                ]
                if actual_response.joins
                else [],
                "requires_joins": bool(actual_response.joins),
            },
        }

        # Judge evaluation
        if USE_LLM_JUDGE and judge is not None:
            criteria = JudgeCriteria(**test_data.get("judge_criteria", {}))

            expected_str = json.dumps(expected_response, indent=2)
            actual_str = json.dumps(report["actual_response"], indent=2)

            judge_result = judge.evaluate(
                question=f"Determine joins needed for: {question}",
                expected=expected_str,
                actual=actual_str,
                criteria=criteria,
                test_type="sql_join",
            )

            time.sleep(1)

            report["judge_result"] = judge_result.model_dump()
            report["passed"] = judge_result.passed
            judge_results["join"][fixture_name] = judge_result

            results["join"].append(report)
            with open(
                current_path / "reports" / "sql_join_judge_report.json", "w"
            ) as f:
                json.dump(results["join"], f, indent=2)

            assert judge_result.passed, (
                f"LLM Judge evaluation failed:\n"
                f"Scores: {judge_result.scores.model_dump()}\n"
                f"Reasoning: {judge_result.reasoning}\n"
                f"Assessment: {judge_result.overall_assessment}"
            )

    @classmethod
    def teardown_class(cls):
        """Generate summary reports for all SQL stages."""

        if not USE_LLM_JUDGE:
            return

        # Generate summary for each stage
        for stage in ["grounding", "filter", "aggregation", "join"]:
            if not judge_results[stage]:
                continue

            total_tests = len(judge_results[stage])
            passed_tests = sum(1 for r in judge_results[stage].values() if r.passed)

            # Calculate average scores
            avg_accuracy = (
                sum(r.scores.accuracy for r in judge_results[stage].values())
                / total_tests
            )
            avg_relevance = (
                sum(r.scores.relevance for r in judge_results[stage].values())
                / total_tests
            )
            avg_completeness = (
                sum(r.scores.completeness for r in judge_results[stage].values())
                / total_tests
            )
            avg_hallucination = (
                sum(r.scores.hallucination for r in judge_results[stage].values())
                / total_tests
            )

            summary = {
                "test_type": f"sql_{stage}",
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
                    for test_id, result in judge_results[stage].items()
                    if not result.passed
                ],
            }

            # Write summary
            with open(
                current_path / "reports" / f"sql_{stage}_judge_summary.json", "w"
            ) as f:
                json.dump(summary, f, indent=2)

            print(f"\nSQL {stage.capitalize()} Test Summary with LLM Judge:")
            print(f"Total Tests: {total_tests}")
            print(f"Passed: {passed_tests}")
            print(f"Failed: {total_tests - passed_tests}")
            print(f"Pass Rate: {summary['pass_rate']}")
            print("\nAverage Scores:")
            for metric, score in summary["average_scores"].items():
                print(f"  {metric.capitalize()}: {score}/10")
