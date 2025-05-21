import unittest
from unittest.mock import patch

from src.agent.domain.commands import (
    GuardrailPostCheckModel,
    GuardrailPreCheckModel,
    LLMResponseModel,
)
from src.agent.entrypoints.main import answer


class TestCLI(unittest.TestCase):
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
        mock_CodeAgent.return_value = ("agent test", "agent memory")

        mock_LLM.side_effect = [
            GuardrailPreCheckModel(
                approved=True,
                chain_of_thought="chain_of_thought",
                response="test answer",
            ),
            LLMResponseModel(
                response="test answer", chain_of_thought="chain_of_thought"
            ),
            LLMResponseModel(
                response="test answer", chain_of_thought="chain_of_thought"
            ),
            GuardrailPostCheckModel(
                chain_of_thought="chain_of_thought",
                approved=True,
                summary="summary",
                issues=[],
                plausibility="plausibility",
                factual_consistency="factual_consistency",
                clarity="clarity",
                completeness="completeness",
            ),
        ]

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

        question = "test"

        response = answer(question, "test_session_id")

        assert response == "done"

    def test_unhappy_path_returns_400_and_answers(
        self,
    ):
        question = None
        try:
            response = answer(question, "test_session_id")
        except Exception as e:
            assert str(e) == "No question asked"
