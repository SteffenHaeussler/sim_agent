import time
from pathlib import Path

import pytest

from evals.llm_judge import JudgeCriteria, LLMJudge
from evals.utils import load_yaml_fixtures
from src.agent import config
from src.agent.adapters import agent_tools

current_path = Path(__file__).parent
# Load e2e fixtures since this tests the same functionality via direct tool calls
fixtures = load_yaml_fixtures(
    current_path, "", recursive=False
)  # Load YAML files from current directory
tools = agent_tools.Tools(config.get_tools_config())


class TestEvalPlanning:
    """Tool agent evaluation tests."""

    def setup_method(self):
        """Initialize LLM Judge for evaluation."""
        self.judge = LLMJudge()

    @pytest.mark.parametrize(
        "fixture_name, fixture",
        [
            pytest.param(fixture_name, fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_eval_tool_agent(self, fixture_name, fixture):
        """Run tool agent test with optional LLM judge evaluation."""

        # Extract test data - flat structure from YAML
        question = fixture["question"]
        expected_response = fixture["response"]

        # Start timing
        # start_time = time.time()

        # Execute tool agent
        response, _ = tools.use(question)

        # Calculate execution time
        # execution_time_ms = int((time.time() - start_time) * 1000)

        # Add delay to avoid rate limiting (tool agent makes many small API calls)
        time.sleep(60)

        if isinstance(response, list):
            response = sorted(response)

        # Handle different response types for basic pass/fail
        # if isinstance(expected_response, dict) and "plot" in expected_response:
        #     basic_passed = len(response) > 0
        # elif isinstance(expected_response, dict) and "comparison" in expected_response:
        #     basic_passed = len(response) > 0
        # else:
        #     basic_passed = expected_response in response

        # Use LLM Judge for evaluation
        criteria = JudgeCriteria(**fixture.get("judge_criteria", {}))
        judge_result = self.judge.evaluate(
            question=question,
            expected=str(expected_response),
            actual=str(response),
            criteria=criteria,
            test_type="tool_agent",
        )

        # Assert judge passed
        assert judge_result.passed, f"Judge failed: {judge_result.overall_assessment}"
