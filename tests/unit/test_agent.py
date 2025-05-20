from unittest.mock import patch

import pytest

from src.agent.domain import commands
from src.agent.domain.model import BaseAgent


class TestAgent:
    def test_agent_initialization(self):
        question = commands.Question(question="test query", q_id="test session id")
        agent = BaseAgent(question)

        assert agent.question == "test query"
        assert agent.q_id == "test session id"
        assert agent.enhancement is None
        assert agent.tool_answer is None
        assert agent.response is None
        assert agent.is_answered is False
        assert agent.previous_command is None
        assert agent.kwargs is None
        assert agent.events == []

    def test_agent_change_llm_response(self):
        question = commands.Question(question="test query", q_id="test session id")
        agent = BaseAgent(question)

        tool_answer = commands.UseTools(
            question="test query",
            q_id="test session id",
            response="test response",
        )

        command = commands.LLMResponse(
            question="test query",
            q_id="test session id",
            response="test response",
            chain_of_thought="test chain of thought",
        )
        agent.tool_answer = tool_answer
        agent.update(command)

        assert agent.tool_answer == tool_answer
        assert agent.is_answered is True
        assert agent.response.response == "test response"
        assert agent.previous_command is type(command)

    def test_agent_change_question(self):
        question = commands.Enhance(
            question="test query", q_id="test session id", response="test response"
        )
        agent = BaseAgent(question)

        response = agent.update(question)
        assert response == commands.UseTools(
            question="test response",
            q_id="test session id",
        )
        assert agent.is_answered is False
        assert agent.previous_command is type(question)

    def test_agent_none_change_question(self):
        question = commands.Enhance(
            question="test query", q_id="test session id", response=None
        )
        agent = BaseAgent(question)

        response = agent.update(question)
        assert response == commands.UseTools(
            question="test query",
            q_id="test session id",
        )
        assert agent.is_answered is False
        assert agent.previous_command is type(question)

    def test_agent_check_question(self):
        question = commands.Question(question="test query", q_id="test session id")
        agent = BaseAgent(question)

        response = agent.check_question(question)
        assert response == commands.Retrieve(
            question="test query", q_id="test session id"
        )

    @patch("src.agent.domain.model.BaseAgent.create_prompt")
    def test_agent_change_use_tools(self, mock_create_prompt):
        question = commands.UseTools(
            question="test query", q_id="test session id", response="test response"
        )

        mock_create_prompt.return_value = "test prompt"

        agent = BaseAgent(question)

        response = agent.update(question)
        assert response == commands.LLMResponse(
            question="test prompt",
            q_id="test session id",
        )

    @patch("src.agent.domain.model.BaseAgent.create_prompt")
    def test_agent_change_rerank(self, mock_create_prompt):
        question = commands.Rerank(
            question="test query", q_id="test session id", candidates=[]
        )

        mock_create_prompt.return_value = "test prompt"

        agent = BaseAgent(question)

        response = agent.update(question)
        assert response == commands.Enhance(
            question="test prompt", q_id="test session id"
        )

    def test_agent_change_retrieve(self):
        question = commands.Retrieve(
            question="test query", q_id="test session id", candidates=[]
        )
        agent = BaseAgent(question)

        response = agent.update(question)
        assert response == commands.Rerank(
            question="test query", q_id="test session id", candidates=[]
        )

    def test_change_llm_response_without_tools(self):
        question = commands.Question(question="test query", q_id="test session id")
        agent = BaseAgent(question)

        command = commands.LLMResponse(
            question="test query",
            q_id="test session id",
            response="test response",
            chain_of_thought="test chain of thought",
        )
        with pytest.raises(
            ValueError, match="Tool answer is required for LLM response"
        ):
            agent.update(command)
