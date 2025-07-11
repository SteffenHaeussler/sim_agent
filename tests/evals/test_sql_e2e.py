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
    USE_LLM_JUDGE,
    BaseSQLEvalTest,
)
from tests.utils import get_fixtures

current_path = Path(__file__).parent

fixtures = get_fixtures(current_path, keys=["sql_e2e"])

# Load database schema
with open(current_path / "schema.json", "r") as f:
    schema_data = json.load(f)

# Create DatabaseSchema object
db_schema = commands.DatabaseSchema(**schema_data)


class TestEvalSQLEndToEnd(BaseSQLEvalTest):
    def setup_method(self):
        """Setup for each test."""
        super().__init__()
        self.current_path = current_path

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

        # Evaluate the result
        if self.judge:
            # Use judge evaluation
            self.evaluate_with_judge(
                stage_name="e2e",
                fixture_name=fixture_name,
                question=question,
                expected_response={"sql": expected_sql},
                actual_response_dict={"sql": actual_sql},
                test_data=test_data,
                judge_question=question,
            )
        else:
            # Simple exact match
            passed = actual_sql.strip() == expected_sql.strip()
            report = {
                "test_id": fixture_name,
                "question": question,
                "expected_sql": expected_sql,
                "actual_sql": actual_sql,
                "passed": passed,
            }
            self.results["e2e"] = self.results.get("e2e", [])
            self.results["e2e"].append(report)
            self.write_report(
                self.current_path / "reports" / "sql_e2e_report.json",
                self.results["e2e"],
            )
            assert passed, (
                f"SQL mismatch:\nExpected:\n{expected_sql}\n\nActual:\n{actual_sql}"
            )

    @classmethod
    def teardown_class(cls):
        """Generate summary report after all tests."""
        if not USE_LLM_JUDGE:
            return

        # Access class-level results
        judge_results = cls._class_judge_results

        if "e2e" in judge_results:
            # Create a temporary instance for summary generation
            instance = cls()
            instance.__init__()
            instance.current_path = current_path
            instance.generate_stage_summary("e2e", judge_results["e2e"])
