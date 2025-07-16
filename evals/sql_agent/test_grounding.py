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
fixtures = load_yaml_fixtures(current_path, "grounding")
schema = load_database_schema(current_path, "schema/schema.json")


class TestEvalGrounding:
    """SQL Grounding evaluation tests."""

    def setup_method(self):
        """Initialize LLM Judge for evaluation."""
        self.judge = LLMJudge()

    def setup_class(self):
        """Setup report file."""
        self.results = []

    def teardown_class(self):
        """Save results to report file."""
        save_test_report(self.results, "sql_grounding")

    @pytest.mark.parametrize(
        "fixture_name, fixture",
        [
            pytest.param(fixture_name, fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_eval_grounding(self, fixture_name, fixture, agent_config, llm_config):
        """Run SQL grounding test with optional LLM judge evaluation."""

        # Extract test data - fixture is now the test data directly
        question_text = fixture["question"]
        expected_response = fixture["expected_response"]

        q_id = f"test_grounding_{str(uuid.uuid4())}"
        sql_question = commands.SQLQuestion(question=question_text, q_id=q_id)

        llm = LLM(llm_config)
        agent = sql_model.SQLBaseAgent(
            question=sql_question,
            kwargs=agent_config,
        )

        # Create SQLCheck command as input to grounding
        # (In the SQL pipeline, grounding comes after check)
        check_command = commands.SQLCheck(
            question=question_text,
            q_id=q_id,
            approved=True,  # Assume check passed for grounding test
        )

        # Set up the construction state with schema
        agent.construction.schema_info = schema

        # Start timing
        start_time = time.time()

        # Prepare grounding command
        grounding_command = agent.update(check_command)

        # Execute grounding
        response = llm.use(
            grounding_command.question, response_model=commands.SQLGrounding
        )

        # Extract actual table and column mappings
        actual_response = {"table_mapping": [], "column_mapping": []}

        if response.table_mapping:
            for table_map in response.table_mapping:
                actual_response["table_mapping"].append(
                    {
                        "question_term": table_map.question_term,
                        "table_name": table_map.table_name,
                        "confidence": table_map.confidence,
                    }
                )

        if response.column_mapping:
            for col_map in response.column_mapping:
                actual_response["column_mapping"].append(
                    {
                        "question_term": col_map.question_term,
                        "table_name": col_map.table_name,
                        "column_name": col_map.column_name,
                        "confidence": col_map.confidence,
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
            test_type="sql_grounding",
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
