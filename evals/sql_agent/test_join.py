import time
import uuid
from pathlib import Path

import pytest

from evals.llm_judge import JudgeCriteria, LLMJudge
from evals.utils import (
    get_model_info_for_test,
    load_database_schema,
    load_yaml_fixtures,
    save_test_report,
)
from src.agent.adapters.llm import LLM
from src.agent.domain import commands, sql_model

current_path = Path(__file__).parent
# Load fixtures from YAML file
fixtures = load_yaml_fixtures(current_path, "join")
schema = load_database_schema(current_path, "schema/schema.json")


class TestEvalJoin:
    """SQL Join Inference evaluation tests."""

    def setup_method(self):
        """Initialize LLM Judge for evaluation."""
        self.judge = LLMJudge()

    def setup_class(self):
        """Setup report file."""
        self.results = []

    def teardown_class(self):
        """Save results to report file."""
        model_info = get_model_info_for_test("sql_join")
        save_test_report(self.results, "sql_join", model_info)

    @pytest.mark.parametrize(
        "fixture_name, fixture",
        [
            pytest.param(fixture_name, fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_eval_join(self, fixture_name, fixture, agent_config, llm_config):
        """Run SQL join inference test with optional LLM judge evaluation."""

        # Extract test data - fixture is now the test data directly
        question_text = fixture["question"]
        table_mapping = fixture.get("table_mapping", [])
        expected_response = fixture["expected_response"]

        # Convert table_mapping to TableMapping objects
        table_mappings = [commands.TableMapping(**mapping) for mapping in table_mapping]

        q_id = f"test_join_{str(uuid.uuid4())}"
        sql_question = commands.SQLQuestion(question=question_text, q_id=q_id)

        llm = LLM(llm_config)
        agent = sql_model.SQLBaseAgent(
            question=sql_question,
            kwargs=agent_config,
        )

        # Create SQLFilter command as input to join inference
        # (In the SQL pipeline, join inference comes after filter)
        filter_command = commands.SQLFilter(
            question=question_text,
            q_id=q_id,
            column_mapping=[],  # Not needed for join test
            conditions=[],  # Not needed for join test
        )

        # Set up the construction state with table mappings and schema
        agent.construction.table_mapping = table_mappings
        agent.construction.schema_info = schema

        # Start timing
        start_time = time.time()

        # Prepare join inference command
        join_command = agent.update(filter_command)

        # Execute join inference
        response = llm.use(
            join_command.question, response_model=commands.SQLJoinInference
        )

        # Extract actual joins
        actual_joins = []
        if response.joins:
            for join in response.joins:
                actual_joins.append(
                    {
                        "from_table": join.from_table,
                        "to_table": join.to_table,
                        "from_column": join.from_column,
                        "to_column": join.to_column,
                        "join_type": join.join_type,
                    }
                )

        # Calculate execution time
        execution_time_ms = int((time.time() - start_time) * 1000)

        # Add delay to avoid rate limiting
        time.sleep(1)

        # Use LLM Judge for evaluation
        criteria = JudgeCriteria(**fixture.get("judge_criteria", {}))
        judge_result = self.judge.evaluate(
            question=question_text,
            expected=str(expected_response),
            actual=str(actual_joins),
            criteria=criteria,
            test_type="sql_join",
        )

        # Record result
        result = {
            "test_name": fixture_name,
            "question": question_text,
            "expected": str(expected_response),
            "actual": str(actual_joins),
            "passed": judge_result.passed,
            "execution_time_ms": execution_time_ms,
            "overall_score": (
                judge_result.scores.accuracy
                + judge_result.scores.relevance
                + judge_result.scores.completeness
                + judge_result.scores.hallucination
            )
            / 4,
            "accuracy": judge_result.scores.accuracy,
            "relevance": judge_result.scores.relevance,
            "completeness": judge_result.scores.completeness,
            "hallucination": judge_result.scores.hallucination,
            "judge_assessment": judge_result.overall_assessment,
        }
        self.__class__.results.append(result)

        # Assert judge passed
        assert judge_result.passed, f"Judge failed: {judge_result.overall_assessment}"
