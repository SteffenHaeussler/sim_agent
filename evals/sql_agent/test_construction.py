import time
import uuid
from pathlib import Path

import pytest

from evals.llm_judge import JudgeCriteria, LLMJudge
from evals.utils import (
    get_model_info_for_test,
    load_database_schema,
    load_yaml_fixtures,
    normalize_sql,
    save_test_report,
)
from src.agent.adapters.llm import LLM
from src.agent.domain import commands, sql_model

current_path = Path(__file__).parent
# Load fixtures from YAML file
fixtures = load_yaml_fixtures(current_path, "construction")
schema = load_database_schema(current_path, "schema/schema.json")


class TestEvalConstruction:
    """SQL Construction evaluation tests."""

    def setup_method(self):
        """Initialize LLM Judge for evaluation."""
        self.judge = LLMJudge()

    def setup_class(self):
        """Setup report file."""
        self.results = []

    def teardown_class(self):
        """Save results to report file."""
        model_info = get_model_info_for_test("sql_construction")
        save_test_report(self.results, "sql_construction", model_info)

    @pytest.mark.parametrize(
        "fixture_name, fixture",
        [
            pytest.param(fixture_name, fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_eval_construction(self, fixture_name, fixture, agent_config, llm_config):
        """Run SQL construction test with optional LLM judge evaluation."""

        # Extract test data - fixture is now the test data directly
        question_text = fixture["question"]
        column_mapping = fixture.get("column_mapping", [])
        table_mapping = fixture.get("table_mapping", [])
        conditions = fixture.get("conditions", [])
        joins = fixture.get("joins", [])
        aggregations = fixture.get("aggregations", [])
        group_by_columns = fixture.get("group_by_columns", [])
        is_aggregation_query = fixture.get("is_aggregation_query", False)
        expected_sql = fixture["expected_response"]["sql"]

        q_id = f"test_construction_{str(uuid.uuid4())}"
        sql_question = commands.SQLQuestion(question=question_text, q_id=q_id)

        llm = LLM(llm_config)
        agent = sql_model.SQLBaseAgent(
            question=sql_question,
            kwargs=agent_config,
        )

        # Create SQLAggregation command as input to construction
        # (In the SQL pipeline, construction comes after aggregation)
        aggregation_command = commands.SQLAggregation(
            question=question_text,
            q_id=q_id,
            column_mapping=[commands.ColumnMapping(**cm) for cm in column_mapping],
            aggregations=[commands.AggregationFunction(**agg) for agg in aggregations],
            group_by_columns=group_by_columns,
            is_aggregation_query=is_aggregation_query,
        )

        # Set up the construction state with all necessary components
        agent.construction.schema_info = schema
        agent.construction.column_mapping = [
            commands.ColumnMapping(**cm) for cm in column_mapping
        ]
        agent.construction.table_mapping = [
            commands.TableMapping(**tm) for tm in table_mapping
        ]
        agent.construction.conditions = [
            commands.FilterCondition(**cond) for cond in conditions
        ]
        agent.construction.joins = [commands.JoinPath(**join) for join in joins]
        agent.construction.aggregations = aggregation_command.aggregations
        agent.construction.group_by_columns = aggregation_command.group_by_columns
        agent.construction.is_aggregation_query = (
            aggregation_command.is_aggregation_query
        )

        # Start timing
        start_time = time.time()

        # Prepare construction command
        construction_command = agent.update(aggregation_command)

        # Execute construction
        response = llm.use(
            construction_command.question, response_model=commands.SQLConstruction
        )

        # Extract actual SQL query
        actual_sql = response.sql_query or ""

        # Calculate execution time
        execution_time_ms = int((time.time() - start_time) * 1000)

        # Add delay to avoid rate limiting
        time.sleep(1)

        # Normalize SQL for comparison

        # Use LLM Judge for evaluation
        criteria = JudgeCriteria(**fixture.get("judge_criteria", {}))
        judge_result = self.judge.evaluate(
            question=question_text,
            expected=normalize_sql(expected_sql),
            actual=normalize_sql(actual_sql) if actual_sql else "NO SQL GENERATED",
            criteria=criteria,
            test_type="sql_construction",
        )

        # Record result
        result = {
            "test_name": fixture_name,
            "question": question_text,
            "expected": normalize_sql(expected_sql),
            "actual": normalize_sql(actual_sql) if actual_sql else "NO SQL GENERATED",
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
