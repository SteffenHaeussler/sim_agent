from unittest.mock import patch

from src.agent.adapters import agent_tools, llm, rag
from src.agent.adapters.adapter import AgentAdapter
from src.agent.domain import commands


class TestAdapter:
    def test_agent_init(self):
        adapter = AgentAdapter()

        assert type(adapter.tools) is agent_tools.Tools
        assert type(adapter.llm) is llm.LLM
        assert type(adapter.rag) is rag.BaseRAG

    @patch("src.agent.adapters.agent_tools.Tools.use")
    def test_agent_use_tools(self, mock_CodeAgent):
        question = "test"
        adapter = AgentAdapter()

        mock_CodeAgent.return_value = ("agent test", "memory")
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

    @patch("src.agent.adapters.rag.BaseRAG.rerank")
    def test_agent_rerank(self, mock_rerank):
        question = "test"
        candidates = [
            commands.KBResponse(
                description="1234", id="2", tag="tag", name="name", score=0.0
            )
        ]
        adapter = AgentAdapter()

        mock_rerank.return_value = {
            "question": "test",
            "text": "1234",
            "score": -10.0,
        }
        question = commands.Rerank(question="test", q_id="1", candidates=candidates)

        response = adapter.answer(question)

        assert response.question == "test"
        assert response.q_id == "1"
        assert response.candidates[0].question == "test"
        assert response.candidates[0].text == "1234"
        assert response.candidates[0].score == -10.0
        assert response.candidates[0].id == "2"
        assert response.candidates[0].tag == "tag"
        assert response.candidates[0].name == "name"

    @patch("src.agent.adapters.rag.BaseRAG.retrieve")
    @patch("src.agent.adapters.rag.BaseRAG.embed")
    def test_agent_retrieve(self, mock_embed, mock_retrieve):
        question = "test"

        adapter = AgentAdapter()

        mock_embed.return_value = {
            "embedding": [0.1, 0.2, 0.3],
        }
        mock_retrieve.return_value = {
            "results": [
                {
                    "id": "1",
                    "text": "1234",
                    "score": -10.0,
                    "description": "1234",
                    "tag": "tag",
                    "name": "name",
                    "location": "location",
                    "area": "area",
                },
                {
                    "id": "2",
                    "text": "1234",
                    "score": -11.0,
                    "description": "1234",
                    "tag": "tag",
                    "name": "name",
                    "location": "location",
                    "area": "area",
                },
            ]
        }
        question = commands.Retrieve(question="test", q_id="1")

        response = adapter.answer(question)

        assert response.question == "test"
        assert response.q_id == "1"
        assert response.candidates[0].description == "1234"
        assert response.candidates[0].score == -10.0
        assert response.candidates[0].id == "1"
        assert response.candidates[0].tag == "tag"
        assert response.candidates[0].name == "name"
        assert response.candidates[1].description == "1234"
        assert response.candidates[1].score == -11.0
        assert response.candidates[1].id == "2"
        assert response.candidates[1].tag == "tag"
        assert response.candidates[1].name == "name"

    @patch("src.agent.adapters.rag.BaseRAG.embed")
    def test_agent_no_retrieve(self, mock_embed):
        question = "test"

        adapter = AgentAdapter()

        mock_embed.return_value = None
        question = commands.Retrieve(question="test", q_id="1")

        response = adapter.answer(question)

        assert response.question == "test"
        assert response.q_id == "1"
        assert response.candidates == []

    @patch("src.agent.adapters.llm.LLM.use")
    def test_agent_enhance(self, mock_LLM):
        mock_LLM.return_value = commands.LLMResponseModel(
            response="test answer", chain_of_thought="chain_of_thought"
        )

        question = commands.Enhance(question="test", q_id="test_session_id")
        adapter = AgentAdapter()

        response = adapter.answer(question)

        assert response.response == "test answer"
        assert response.chain_of_thought == "chain_of_thought"
