import time
import uuid
from pathlib import Path

import pytest

from evals.llm_judge import JudgeCriteria, LLMJudge
from evals.utils import load_database_schema, load_yaml_fixtures, save_test_report
from src.agent.adapters.llm import LLM
from src.agent.domain import commands, sql_model

current_path = Path(__file__).parent
# Load fixtures from YAML file
fixtures = load_yaml_fixtures(current_path, "filter")
schema = load_database_schema(current_path, "schema/schema.json")


class TestEvalFilter:
    """SQL Filter evaluation tests."""

    def setup_method(self):
        """Initialize LLM Judge for evaluation."""
        self.judge = LLMJudge()

    def setup_class(self):
        """Setup report file."""
        self.results = []

    def teardown_class(self):
        """Save results to report file."""
        save_test_report(self.results, "sql_filter")

    @pytest.mark.parametrize(
        "fixture_name, fixture",
        [
            pytest.param(fixture_name, fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_eval_filter(self, fixture_name, fixture, agent_config, llm_config):
        """Run SQL filter test with optional LLM judge evaluation."""

        # Extract test data - fixture is now the test data directly
        question_text = fixture["question"]
        column_mapping = fixture.get("column_mapping", [])
        expected_response = fixture["expected_response"]

        column_mapping = [commands.ColumnMapping(**i) for i in column_mapping]
        # expected_response = [commands.FilterCondition(**i) for i in expected_response]

        q_id = f"test_filter_{str(uuid.uuid4())}"
        sql_question = commands.SQLQuestion(question=question_text, q_id=q_id)

        llm = LLM(llm_config)
        agent = sql_model.SQLBaseAgent(
            question=sql_question,
            kwargs=agent_config,
        )

        # Create SQLGrounding command as input to filter
        grounding_command = commands.SQLGrounding(
            question=question_text,
            q_id=q_id,
            column_mapping=column_mapping,
            table_mapping=[],  # Not needed for filter test
            tables=schema.tables,
        )

        # Start timing
        start_time = time.time()

        # Prepare filter command
        filter_command = agent.update(grounding_command)

        # Execute filter
        response = llm.use(filter_command.question, response_model=commands.SQLFilter)

        # Extract actual conditions
        actual_conditions = []
        if response.conditions:
            for condition in response.conditions:
                actual_conditions.append(
                    {
                        "column": condition.column,
                        "operator": condition.operator,
                        "value": condition.value,
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
            actual=str(actual_conditions),
            criteria=criteria,
            test_type="sql_filter",
        )

        # Record result
        result = {
            "test_name": fixture_name,
            "question": question_text,
            "expected": str(expected_response),
            "actual": str(actual_conditions),
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
