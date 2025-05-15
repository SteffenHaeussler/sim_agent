import unittest
from unittest.mock import patch

from src.agent.domain.commands import LLMResponseModel
from src.agent.entrypoints.main import answer


class TestCLI(unittest.TestCase):
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
