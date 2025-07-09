import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

from src.agent.adapters.adapter import SQLAgentAdapter
from src.agent.config import get_llm_config
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


def create_mock_database_schema():
    """Create a mock database schema for testing."""
    # Mock schema based on the actual database
    tables = [
        commands.Table(
            name="products",
            columns=[
                commands.Column(
                    name="product_id", data_type="SERIAL", is_primary_key=True
                ),
                commands.Column(name="name", data_type="VARCHAR(200)"),
                commands.Column(name="sku", data_type="VARCHAR(100)"),
                commands.Column(name="category_id", data_type="INTEGER"),
                commands.Column(name="is_active", data_type="BOOLEAN"),
            ],
        ),
        commands.Table(
            name="employees",
            columns=[
                commands.Column(
                    name="employee_id", data_type="SERIAL", is_primary_key=True
                ),
                commands.Column(name="name", data_type="VARCHAR(100)"),
                commands.Column(name="role_id", data_type="INTEGER"),
                commands.Column(name="territory_id", data_type="INTEGER"),
                commands.Column(name="manager_id", data_type="INTEGER"),
                commands.Column(name="salary", data_type="DECIMAL(12,2)"),
                commands.Column(name="salary_currency_code", data_type="VARCHAR(3)"),
            ],
        ),
        commands.Table(
            name="roles",
            columns=[
                commands.Column(
                    name="role_id", data_type="SERIAL", is_primary_key=True
                ),
                commands.Column(name="name", data_type="VARCHAR(50)"),
            ],
        ),
        commands.Table(
            name="territories",
            columns=[
                commands.Column(
                    name="territory_id", data_type="SERIAL", is_primary_key=True
                ),
                commands.Column(name="name", data_type="VARCHAR(100)"),
                commands.Column(name="country_id", data_type="INTEGER"),
            ],
        ),
        commands.Table(
            name="inventory",
            columns=[
                commands.Column(
                    name="inventory_id", data_type="SERIAL", is_primary_key=True
                ),
                commands.Column(name="product_id", data_type="INTEGER"),
                commands.Column(name="territory_id", data_type="INTEGER"),
                commands.Column(name="quantity_on_hand", data_type="INTEGER"),
                commands.Column(name="reorder_level", data_type="INTEGER"),
            ],
        ),
    ]

    relationships = [
        commands.Relationship(
            from_table="employees",
            from_column="role_id",
            to_table="roles",
            to_column="role_id",
            relationship_type="many-to-one",
        ),
        commands.Relationship(
            from_table="employees",
            from_column="territory_id",
            to_table="territories",
            to_column="territory_id",
            relationship_type="many-to-one",
        ),
        commands.Relationship(
            from_table="inventory",
            from_column="product_id",
            to_table="products",
            to_column="product_id",
            relationship_type="many-to-one",
        ),
        commands.Relationship(
            from_table="inventory",
            from_column="territory_id",
            to_table="territories",
            to_column="territory_id",
            relationship_type="many-to-one",
        ),
    ]

    return commands.DatabaseSchema(tables=tables, relationships=relationships)


class TestEvalSQLStages:
    def setup_method(self):
        """Setup mock database schema for each test."""
        self.schema = create_mock_database_schema()
        self.adapter = SQLAgentAdapter(get_llm_config())
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
            schema=self.schema,
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
                "tables": actual_response.tables,
                "columns": actual_response.columns,
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
            with open("sql_grounding_judge_report.json", "w") as f:
                json.dump(results["grounding"], f, indent=2)

            assert judge_result.passed, (
                f"LLM Judge evaluation failed:\n"
                f"Scores: {judge_result.scores.model_dump()}\n"
                f"Reasoning: {judge_result.reasoning}\n"
                f"Assessment: {judge_result.overall_assessment}"
            )
        else:
            # Basic validation
            assert actual_response.tables
            assert actual_response.columns

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
        grounding_result = commands.GroundingResponse(
            tables=test_data.get("grounding_tables", ["products"]),
            columns=test_data.get("grounding_columns", ["is_active", "name"]),
        )

        # Create filter command
        q_id = str(uuid.uuid4())
        command = commands.SQLFilter(
            question=question,
            q_id=q_id,
            schema=self.schema,
            grounding=grounding_result,
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
                "where_conditions": actual_response.where_conditions,
                "having_conditions": actual_response.having_conditions,
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
            with open("sql_filter_judge_report.json", "w") as f:
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
        grounding_result = commands.GroundingResponse(
            tables=test_data.get("grounding_tables", ["orders", "customers"]),
            columns=test_data.get(
                "grounding_columns", ["customer_id", "grand_total_amount"]
            ),
        )

        # Create aggregation command
        q_id = str(uuid.uuid4())
        command = commands.SQLAggregation(
            question=question,
            q_id=q_id,
            schema=self.schema,
            grounding=grounding_result,
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
                "aggregate_functions": actual_response.aggregate_functions,
                "requires_aggregation": actual_response.requires_aggregation,
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
            with open("sql_aggregation_judge_report.json", "w") as f:
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
        grounding_result = commands.GroundingResponse(
            tables=test_data.get(
                "grounding_tables", ["employees", "roles", "territories"]
            ),
            columns=test_data.get(
                "grounding_columns", ["name", "role_id", "territory_id"]
            ),
        )

        # Create join inference command
        q_id = str(uuid.uuid4())
        command = commands.SQLJoinInference(
            question=question,
            q_id=q_id,
            schema=self.schema,
            grounding=grounding_result,
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
                        "type": join.type,
                        "left_table": join.left_table,
                        "right_table": join.right_table,
                        "on_condition": join.on_condition,
                    }
                    for join in actual_response.joins
                ]
                if actual_response.joins
                else [],
                "requires_joins": actual_response.requires_joins,
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
            with open("sql_join_judge_report.json", "w") as f:
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
            with open(f"sql_{stage}_judge_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            print(f"\nSQL {stage.capitalize()} Test Summary with LLM Judge:")
            print(f"Total Tests: {total_tests}")
            print(f"Passed: {passed_tests}")
            print(f"Failed: {total_tests - passed_tests}")
            print(f"Pass Rate: {summary['pass_rate']}")
            print("\nAverage Scores:")
            for metric, score in summary["average_scores"].items():
                print(f"  {metric.capitalize()}: {score}/10")
