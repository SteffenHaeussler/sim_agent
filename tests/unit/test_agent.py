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
        question = commands.Question(question="test query", q_id="test session id")
        agent = BaseAgent(question)

        response = agent.update(question)
        assert response == commands.UseTools(
            question="test query",
            q_id="test session id",
        )
        assert agent.is_answered is False
        assert agent.previous_command is type(question)

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
