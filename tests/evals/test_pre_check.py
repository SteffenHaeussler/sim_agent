import uuid
from pathlib import Path

import pytest

from src.agent.adapters.llm import LLM
from src.agent.config import get_agent_config, get_llm_config
from src.agent.domain import commands, model
from tests.utils import get_fixtures

current_path = Path(__file__).parent

fixtures = get_fixtures(current_path, keys=["pre_check"])
results = []


class TestEvalGuardrails:
    @pytest.mark.parametrize(
        "fixture",
        [
            pytest.param(fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_eval_guardrails(self, fixture):
        question, expected_response = (
            fixture["pre_check"]["question"],
            fixture["pre_check"]["approved"],
        )
        q_id = uuid.uuid4()
        question = commands.Question(question=question, q_id=q_id)

        llm = LLM(get_llm_config())
        agent = model.BaseAgent(
            question=question,
            kwargs=get_agent_config(),
        )

        command = agent.update(question)

        response = llm.use(
            command.question, response_model=commands.GuardrailPreCheckModel
        )

        assert response.approved == expected_response
