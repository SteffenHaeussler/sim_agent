from pathlib import Path

import pytest

from src.agent import config
from src.agent.adapters import agent_tools
from tests.utils import get_fixtures

current_path = Path(__file__).parent

fixtures = get_fixtures(current_path, keys=["tool_agent"])
results = []

tools = agent_tools.Tools(config.get_tools_config())


class TestEvalPlanning:
    @pytest.mark.parametrize(
        "fixture",
        [
            pytest.param(fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_eval_tool_agent(self, fixture):
        question, expected_response = (
            fixture["tool_agent"]["question"],
            fixture["tool_agent"]["response"],
        )

        response, _ = tools.use(question)

        if isinstance(response, list):
            response = sorted(response)

        if isinstance(expected_response, dict) and "plot" in expected_response:
            assert len(response) > 0
            # assert re.match(r"^[A-Za-z0-9+/]+={0,2}$", response)

        elif isinstance(expected_response, dict) and "comparison" in expected_response:
            assert len(response) > 0
            # assert "Mean" in response
            # assert "Standard Deviation" in response
            # assert "Minimum" in response
            # assert "Maximum" in response
            # assert "25th Percentile" in response
            # assert "50th Percentile" in response
            # assert "75th Percentile" in response

        else:
            assert expected_response in response
