import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agent.adapters.adapter import SQLAgentAdapter
from src.agent.domain import commands
from tests.utils import get_fixtures
from tests.evals.base_sql_eval import BaseSQLEvalTest

current_path = Path(__file__).parent

# Load fixtures for different SQL stages
fixtures = get_fixtures(current_path, keys=["sql_stages"])

# Load database schema
with open(current_path / "schema.json", "r") as f:
    schema_data = json.load(f)

# Create DatabaseSchema object
db_schema = commands.DatabaseSchema(**schema_data)


class TestEvalSQLStages(BaseSQLEvalTest):
    """SQL Stages evaluation tests."""

    RUN_TYPE = "sql_stages"
    TEST_TYPE = "sql_stages"

    def setup_method(self):
        """Setup mock database schema for each test."""
        super().setup_method()
        self.current_path = current_path
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

        # Transform response for evaluation
        actual_response_dict = {
            "tables": [tm.table_name for tm in actual_response.table_mapping],
            "columns": [
                f"{cm.table_name}.{cm.column_name}"
                for cm in actual_response.column_mapping
            ],
        }

        # Evaluate with judge and record to database
        self.evaluate_with_judge(
            fixture_name=fixture_name,
            question=question,
            expected_response=expected_response,
            actual_response=actual_response_dict,
            test_data=test_data,
            judge_question=f"Identify tables and columns for SQL query: {question}",
            sql_stage="grounding",
            schema_context={"tables": [t.model_dump() for t in self.schema.tables]},
        )

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

        # Evaluate with judge and record to database
        self.evaluate_with_judge(
            fixture_name=fixture_name,
            question=question,
            expected_response=expected_response,
            actual_response=actual_response_dict,
            test_data=test_data,
            judge_question=f"Extract WHERE conditions from: {question}",
            sql_stage="filter",
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

        # Evaluate with judge and record to database
        self.evaluate_with_judge(
            fixture_name=fixture_name,
            question=question,
            expected_response=expected_response,
            actual_response=actual_response_dict,
            test_data=test_data,
            judge_question=f"Identify aggregation needs for: {question}",
            sql_stage="aggregation",
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

        # Evaluate with judge and record to database
        self.evaluate_with_judge(
            fixture_name=fixture_name,
            question=question,
            expected_response=expected_response,
            actual_response=actual_response_dict,
            test_data=test_data,
            judge_question=f"Determine joins needed for: {question}",
            sql_stage="join",
            schema_context={
                "relationships": [r.model_dump() for r in self.schema.relationships]
            },
        )

    @classmethod
    def teardown_class(cls):
        """Generate summary reports for all SQL stages."""
        # Call parent teardown which handles database completion and summary
        super().teardown_class()
