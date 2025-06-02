import uuid
from pathlib import Path

import pytest

from src.agent.adapters.llm import LLM
from src.agent.config import get_agent_config, get_llm_config
from src.agent.domain import commands, model
from tests.utils import get_fixtures

current_path = Path(__file__).parent

fixtures = get_fixtures(current_path, keys=["enhance"])
results = []


class TestEvalEnhance:
    @pytest.mark.parametrize(
        "fixture",
        [
            pytest.param(fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_eval_tool_agent(self, fixture):
        question, candidates, expected_response = (
            fixture["enhance"]["question"],
            fixture["enhance"]["candidates"],
            fixture["enhance"]["response"],
        )
        q_id = uuid.uuid4()
        question = commands.Question(question=question, q_id=q_id)

        llm = LLM(get_llm_config())
        agent = model.BaseAgent(
            question=question,
            kwargs=get_agent_config(),
        )

        candidates = [
            commands.RerankResponse(
                question=question.question,
                **candidate,
            )
            for candidate in candidates
        ]
        agent.question = question.question

        command = commands.Rerank(
            question=question,
            candidates=candidates,
            q_id=q_id,
        )

        command = agent.prepare_enhancement(command)

        response = llm.use(command.question, response_model=commands.LLMResponseModel)

        assert response.response == expected_response
