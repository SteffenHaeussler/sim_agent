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
fixtures = load_yaml_fixtures(current_path, "aggregate")
schema = load_database_schema(current_path, "schema/schema.json")


class TestEvalAggregate:
    """SQL Aggregation evaluation tests."""

    def setup_method(self):
        """Initialize LLM Judge for evaluation."""
        self.judge = LLMJudge()

    def setup_class(self):
        """Setup report file."""
        self.results = []

    def teardown_class(self):
        """Save results to report file."""
        model_info = get_model_info_for_test("sql_aggregate")
        save_test_report(self.results, "sql_aggregate", model_info)

    @pytest.mark.parametrize(
        "fixture_name, fixture",
        [
            pytest.param(fixture_name, fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_eval_aggregate(self, fixture_name, fixture, agent_config, llm_config):
        """Run SQL aggregation test with optional LLM judge evaluation."""

        # Extract test data - fixture is now the test data directly
        question_text = fixture["question"]
        column_mapping = fixture.get("column_mapping", [])
        expected_response = fixture["expected_response"]

        # Convert column_mapping to ColumnMapping objects
        column_mappings = [
            commands.ColumnMapping(**mapping) for mapping in column_mapping
        ]

        q_id = f"test_aggregate_{str(uuid.uuid4())}"
        sql_question = commands.SQLQuestion(question=question_text, q_id=q_id)

        llm = LLM(llm_config)
        agent = sql_model.SQLBaseAgent(
            question=sql_question,
            kwargs=agent_config,
        )

        # Create SQLJoinInference command as input to aggregation
        # (In the SQL pipeline, aggregation comes after join inference)
        join_command = commands.SQLJoinInference(
            question=question_text,
            q_id=q_id,
            table_mapping=[],  # Not needed for aggregation test
            relationships=[],  # Not needed for aggregation test
            joins=[],  # Not needed for aggregation test
        )

        # Set up the construction state with column mappings
        agent.construction.column_mapping = column_mappings
        agent.construction.schema_info = schema

        # Start timing
        start_time = time.time()

        # Prepare aggregation command
        aggregation_command = agent.update(join_command)

        # Execute aggregation
        response = llm.use(
            aggregation_command.question, response_model=commands.SQLAggregation
        )

        # Extract actual aggregations and group by
        actual_response = {
            "aggregations": [],
            "group_by_columns": response.group_by_columns or [],
            "is_aggregation_query": response.is_aggregation_query,
        }

        if response.aggregations:
            for agg in response.aggregations:
                actual_response["aggregations"].append(
                    {
                        "function": agg.function,
                        "column": agg.column,
                        "alias": agg.alias,
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
            actual=str(actual_response),
            criteria=criteria,
            test_type="sql_aggregate",
        )

        # Record result
        result = {
            "test_name": fixture_name,
            "question": question_text,
            "expected": str(expected_response),
            "actual": str(actual_response),
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
