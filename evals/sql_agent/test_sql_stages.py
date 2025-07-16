import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from src.agent.adapters.adapter import SQLAgentAdapter
from src.agent.domain import commands
from evals.llm_judge import LLMJudge, JudgeCriteria

current_path = Path(__file__).parent


def load_yaml_fixtures(test_dir):
    """Load YAML test fixtures from the sql_stages directory."""
    fixtures = {}
    sql_stages_dir = test_dir / "sql_stages"

    if not sql_stages_dir.exists():
        print(f"WARNING: sql_stages directory not found at {sql_stages_dir}")
        return fixtures

    # Map of stage names to their YAML files
    stage_files = {
        "grounding": "grounding.yaml",
        "filter": "filter.yaml",
        "aggregation": "aggregation.yaml",
        "join": "join.yaml",
    }

    for stage_name, yaml_file in stage_files.items():
        yaml_path = sql_stages_dir / yaml_file
        if not yaml_path.exists():
            continue

        with open(yaml_path, "r") as f:
            suite_data = yaml.safe_load(f)

        # Extract tests from the suite
        for test in suite_data.get("tests", []):
            test_name = f"{stage_name}_{test['name']}"

            # Merge suite defaults with test-specific criteria
            judge_criteria = suite_data.get("default_judge_criteria", {}).copy()
            if "judge_criteria" in test:
                judge_criteria.update(test["judge_criteria"])

            # Create test data in expected format
            test_data = {
                "question": test["question"],
                "expected_response": test["expected_response"],
                "judge_criteria": judge_criteria,
            }

            # Add stage-specific fields
            if stage_name in ["filter", "aggregation", "join"]:
                if "grounding_tables" in test:
                    test_data["grounding_tables"] = test["grounding_tables"]
                if "grounding_columns" in test:
                    test_data["grounding_columns"] = test["grounding_columns"]

            # Convert to expected format with stage nested structure
            fixtures[test_name] = {"sql_stages": {stage_name: test_data}}

    return fixtures


# Load fixtures from YAML files
fixtures = load_yaml_fixtures(current_path)

# Load database schema
with open(current_path / "schema.json", "r") as f:
    schema_data = json.load(f)

# Create DatabaseSchema object
db_schema = commands.DatabaseSchema(**schema_data)


class TestEvalSQLStages:
    """SQL Stages evaluation tests."""

    def setup_method(self):
        """Setup mock database schema for each test."""
        self.judge = LLMJudge()
        self.current_path = current_path
        self.schema = db_schema
        self.adapter = SQLAgentAdapter()
        # Mock the database adapter
        self.adapter.db = MagicMock()
        self.adapter.db.get_schema.return_value = self.schema

    def create_grounding_result(self, tables, columns):
        """Helper to create a mock grounding result."""
        table_mapping = [
            commands.TableMapping(table_name=table, relevance_score=0.9)
            for table in tables
        ]

        column_mapping = []
        for col in columns:
            if "." in col:
                table_name, column_name = col.split(".")
            else:
                table_name = tables[0] if tables else "unknown"
                column_name = col
            column_mapping.append(
                commands.ColumnMapping(
                    table_name=table_name, column_name=column_name, relevance_score=0.9
                )
            )

        return commands.GroundingResult(
            table_mapping=table_mapping, column_mapping=column_mapping
        )

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

        # Transform response for evaluation
        actual_response_dict = {
            "tables": [tm.table_name for tm in actual_response.table_mapping],
            "columns": [
                f"{cm.table_name}.{cm.column_name}"
                for cm in actual_response.column_mapping
            ],
        }

        # Use LLM Judge for evaluation
        criteria = JudgeCriteria(**test_data.get("judge_criteria", {}))
        judge_result = self.judge.evaluate(
            question=f"Identify tables and columns for SQL query: {question}",
            expected=str(expected_response),
            actual=str(actual_response_dict),
            criteria=criteria,
            test_type="sql_grounding",
        )

        # Assert judge passed
        assert judge_result.passed, f"Judge failed: {judge_result.overall_assessment}"

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
        grounding_result = self.create_grounding_result(
            grounding_tables, grounding_columns
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

        # Transform response for evaluation
        actual_response_dict = {
            "where_conditions": [
                f"{cond.column} {cond.operator} {cond.value}"
                for cond in actual_response.conditions
            ]
            if actual_response.conditions
            else [],
            "having_conditions": [],
        }

        # Use LLM Judge for evaluation
        criteria = JudgeCriteria(**test_data.get("judge_criteria", {}))
        judge_result = self.judge.evaluate(
            question=f"Extract WHERE conditions from: {question}",
            expected=str(expected_response),
            actual=str(actual_response_dict),
            criteria=criteria,
            test_type="sql_filter",
        )

        # Assert judge passed
        assert judge_result.passed, f"Judge failed: {judge_result.overall_assessment}"

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
        grounding_result = self.create_grounding_result(
            grounding_tables, grounding_columns
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

        # Transform response for evaluation
        actual_response_dict = {
            "group_by_columns": actual_response.group_by_columns,
            "aggregate_functions": [
                agg.model_dump() for agg in actual_response.aggregations
            ]
            if actual_response.aggregations
            else [],
            "requires_aggregation": actual_response.is_aggregation_query,
        }

        # Use LLM Judge for evaluation
        criteria = JudgeCriteria(**test_data.get("judge_criteria", {}))
        judge_result = self.judge.evaluate(
            question=f"Identify aggregation needs for: {question}",
            expected=str(expected_response),
            actual=str(actual_response_dict),
            criteria=criteria,
            test_type="sql_aggregation",
        )

        # Assert judge passed
        assert judge_result.passed, f"Judge failed: {judge_result.overall_assessment}"

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
        grounding_result = self.create_grounding_result(
            grounding_tables, grounding_columns
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

        # Transform response for evaluation
        actual_response_dict = {
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
            "requires_joins": actual_response.requires_joins
            if hasattr(actual_response, "requires_joins")
            else bool(actual_response.joins),
        }

        # Use LLM Judge for evaluation
        criteria = JudgeCriteria(**test_data.get("judge_criteria", {}))
        judge_result = self.judge.evaluate(
            question=f"Determine joins needed for: {question}",
            expected=str(expected_response),
            actual=str(actual_response_dict),
            criteria=criteria,
            test_type="sql_join",
        )

        # Assert judge passed
        assert judge_result.passed, f"Judge failed: {judge_result.overall_assessment}"
