from unittest.mock import patch

from src.agent.adapters import agent_tools, db, llm, rag
from src.agent.adapters.adapter import AgentAdapter
from src.agent.domain import commands


class TestAdapter:
    def test_agent_init(self):
        adapter = AgentAdapter()

        assert type(adapter.tools) is agent_tools.Tools
        assert type(adapter.db) is db.AbstractDB
        assert type(adapter.llm) is llm.LLM
        assert type(adapter.rerank) is rag.AbstractModel
        assert type(adapter.retrieve) is db.AbstractDB

    @patch("src.agent.adapters.agent_tools.Tools.use")
    def test_agent_use_tools(self, mock_CodeAgent):
        question = "test"
        adapter = AgentAdapter()

        mock_CodeAgent.return_value = "agent test"
        question = commands.UseTools("test", None)

        response = adapter.answer(question)

        assert response.response == "agent test"

    @patch("src.agent.adapters.llm.LLM.use")
    def test_agent_llm(self, mock_LLM):
        mock_LLM.return_value = commands.LLMResponseModel(
            response="test answer", chain_of_thought="chain_of_thought"
        )

        question = commands.LLMResponse(question="test", q_id="test_session_id")
        adapter = AgentAdapter()

        response = adapter.answer(question)

        assert response.response == "test answer"
        assert response.chain_of_thought == "chain_of_thought"
