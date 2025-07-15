import time
from pathlib import Path

import pytest

from src.agent import config
from src.agent.adapters import agent_tools
from tests.utils import get_fixtures
from evals.base_eval_db import BaseEvaluationTest

current_path = Path(__file__).parent
fixtures = get_fixtures(current_path, keys=["tool_agent"])
tools = agent_tools.Tools(config.get_tools_config())


class TestEvalPlanning(BaseEvaluationTest):
    """Tool agent evaluation tests."""

    RUN_TYPE = "tool_agent"
    TEST_TYPE = "tool_agent"
    EVALUATION_CATEGORY = "tool_agent"

    @pytest.mark.parametrize(
        "fixture_name, fixture",
        [
            pytest.param(fixture_name, fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_eval_tool_agent(self, fixture_name, fixture):
        """Run tool agent test with optional LLM judge evaluation."""

        # Extract test data
        test_data = fixture["tool_agent"]
        question = test_data["question"]
        expected_response = test_data["response"]

        # Start timing
        start_time = time.time()

        # Execute tool agent
        response, _ = tools.use(question)

        # Calculate execution time
        execution_time_ms = int((time.time() - start_time) * 1000)

        # Add delay to avoid rate limiting (tool agent makes many small API calls)
        time.sleep(60)

        if isinstance(response, list):
            response = sorted(response)

        # Handle different response types for basic pass/fail
        if isinstance(expected_response, dict) and "plot" in expected_response:
            basic_passed = len(response) > 0
        elif isinstance(expected_response, dict) and "comparison" in expected_response:
            basic_passed = len(response) > 0
        else:
            basic_passed = expected_response in response

        # Extract tools used from response (this is tool-specific logic)
        tools_used = []
        if isinstance(response, list):
            # In this case, response might contain tool names
            tools_used = [str(item) for item in response[:5]]  # Limit to first 5

        # Evaluate with judge and record to database
        if not self.judge:
            # If no judge, use basic pass/fail
            self.record_test_result(
                fixture_name=fixture_name,
                question=question,
                expected_response=expected_response,
                actual_response=response,
                passed=basic_passed,
                execution_time_ms=execution_time_ms,
                tools_used=tools_used,
                tool_outputs={"response_type": type(response).__name__},
            )
        else:
            self.evaluate_with_judge(
                fixture_name=fixture_name,
                question=question,
                expected_response=expected_response,
                actual_response=response,
                test_data=test_data,
                execution_time_ms=execution_time_ms,
                tools_used=tools_used,
                tool_outputs={"response_type": type(response).__name__},
                metadata={"execution_delay_ms": 60000},  # 60 second delay
            )

    @classmethod
    def teardown_class(cls):
        """Generate summary report after all tests."""
        # Call parent teardown which handles database completion and summary
        super().teardown_class()
