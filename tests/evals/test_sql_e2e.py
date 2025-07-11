import json
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from src.agent.adapters.adapter import SQLAgentAdapter
from src.agent.config import get_agent_config
from src.agent.domain import commands
from src.agent.domain.sql_model import SQLBaseAgent
from tests.evals.base_sql_eval import (
    SQL_COMMAND_HANDLERS,
    BaseSQLEvalTest,
)
from tests.utils import get_fixtures

current_path = Path(__file__).parent

fixtures = get_fixtures(current_path, keys=["sql_e2e"])

# Debug: Print fixture loading info
if not fixtures:
    print(f"WARNING: No fixtures loaded from {current_path}")
    # Fallback: try to load fixtures
    import os

    if os.path.exists(current_path / "sql_e2e"):
        print(f"sql_e2e directory exists at {current_path / 'sql_e2e'}")
else:
    print(f"Loaded {len(fixtures)} SQL E2E fixtures")

# Load database schema
with open(current_path / "schema.json", "r") as f:
    schema_data = json.load(f)

# Create DatabaseSchema object
db_schema = commands.DatabaseSchema(**schema_data)


class TestEvalSQLEndToEnd(BaseSQLEvalTest):
    """SQL End-to-End evaluation tests."""

    RUN_TYPE = "sql_e2e"
    TEST_TYPE = "sql_e2e"

    def setup_method(self):
        """Setup for each test."""
        super().setup_method()
        self.current_path = current_path
        self.schema = db_schema

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

            # Process through SQL pipeline using command handlers
            current_command = sql_question

            # Run through the complete SQL generation pipeline
            while not isinstance(current_command, commands.SQLConstruction):
                current_command = agent.update(current_command)

                # Find and execute the appropriate handler
                for command_type, handler_name in SQL_COMMAND_HANDLERS.items():
                    if isinstance(current_command, command_type):
                        handler = getattr(adapter, handler_name)
                        response = handler(current_command)

                        # Store result in agent
                        result_attr = f"{handler_name}_result"
                        setattr(agent, result_attr, response)
                        break

        # Get the final SQL query
        actual_sql = (
            agent.construction_result.sql_query
            if hasattr(agent, "construction_result")
            else ""
        )

        # Evaluate the result using the base class method
        self.evaluate_with_judge(
            fixture_name=fixture_name,
            question=question,
            expected_response={"sql": expected_sql},
            actual_response={"sql": actual_sql},
            test_data=test_data,
            judge_question=question,
            sql_query=actual_sql,
            schema_context=schema_data,
        )

    @classmethod
    def teardown_class(cls):
        """Generate summary report after all tests."""
        # Call parent teardown which handles database completion and summary
        super().teardown_class()
