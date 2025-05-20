import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.agent.domain.commands import LLMResponseModel
from src.agent.entrypoints.app import app

client = TestClient(app)


class TestAPI(unittest.TestCase):
    @patch("src.agent.adapters.rag.BaseRAG.retrieve")
    @patch("src.agent.adapters.rag.BaseRAG.rerank")
    @patch("src.agent.adapters.rag.BaseRAG.embed")
    @patch("src.agent.adapters.llm.LLM.use")
    @patch("src.agent.adapters.agent_tools.Tools.use")
    def test_happy_path_returns_200_and_answers(
        self,
        mock_CodeAgent,
        mock_LLM,
        mock_embed,
        mock_rerank,
        mock_retrieve,
    ):
        mock_CodeAgent.return_value = "agent test"

        mock_LLM.return_value = LLMResponseModel(
            response="test answer", chain_of_thought="chain_of_thought"
        )

        mock_embed.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_rerank.return_value = {
            "question": "test_question",
            "text": "test_text",
            "score": 0.5,
        }

        mock_retrieve.return_value = {
            "results": [
                {
                    "question": "test_question",
                    "description": "test_text",
                    "score": 0.5,
                    "id": "test_id",
                    "tag": "test_tag",
                    "name": "test_name",
                }
            ]
        }

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
