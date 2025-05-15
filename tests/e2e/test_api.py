import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from agentic_ai.src.agent.domain.commands import LLMResponseModel
from agentic_ai.src.agent.entrypoints.app import app

client = TestClient(app)


class TestAPI(unittest.TestCase):
    @patch("src.agent.adapters.llm.LLM.use")
    @patch("src.agent.adapters.agent_tools.Tools.use")
    def test_happy_path_returns_200_and_answers(
        self,
        mock_CodeAgent,
        mock_LLM,
    ):
        mock_CodeAgent.return_value = "agent test"

        mock_LLM.return_value = LLMResponseModel(
            response="test answer", chain_of_thought="chain_of_thought"
        )

        params = {"question": "test"}

        response = client.get("/answer", params=params)

        assert response.status_code == 200
        assert (
            response.json()["response"] == "\nQuestion:\ntest\nResponse:\ntest answer"
        )

    def test_unhappy_path_returns_400_and_answers(self):
        params = {"question": None}
        response = client.get("/answer", params=params)

        assert response.status_code == 400
        assert response.json()["detail"] == "No question asked"
