from unittest.mock import patch

import pytest

from src.agent.config import get_agent_config
from src.agent.domain import commands, events
from src.agent.domain.model import BaseAgent


class TestAgent:
    def test_agent_initialization(self):
        question = commands.Question(question="test query", q_id="test session id")
        agent = BaseAgent(question, get_agent_config())

        assert agent.question == "test query"
        assert agent.q_id == "test session id"
        assert agent.enhancement is None
        assert agent.tool_answer is None
        assert agent.response is None
        assert agent.is_answered is False
        assert agent.previous_command is None
        assert agent.kwargs is not None
        assert agent.events == []
        assert agent.base_prompts is not None

    def test_agent_change_llm_response(self):
        question = commands.Question(question="test query", q_id="test session id")
        agent = BaseAgent(question, get_agent_config())

        tool_answer = commands.UseTools(
            question="test query",
            q_id="test session id",
            response="test response",
            memory=["test memory"],
        )

        command = commands.LLMResponse(
            question="test query",
            q_id="test session id",
            response="test response",
            chain_of_thought="test chain of thought",
        )
        agent.tool_answer = tool_answer
        agent.agent_memory = ["test memory"]

        new_command = agent.update(command)

        assert agent.tool_answer == tool_answer
        assert agent.is_answered is False
        assert agent.response.response == "test response"
        assert agent.previous_command is type(command)

        assert new_command == commands.FinalCheck(
            question="test guardrails post check", q_id="test session id"
        )

    def test_agent_change_question(self):
        question = commands.Enhance(
            question="test query", q_id="test session id", response="test response"
        )
        agent = BaseAgent(question, get_agent_config())

        response = agent.update(question)
        assert response == commands.UseTools(
            question="test response",
            q_id="test session id",
        )
        assert agent.is_answered is False
        assert agent.previous_command is type(question)

    def test_agent_change_check_approved(self):
        question = commands.Check(
            question="test query", q_id="test session id", approved=True
        )
        agent = BaseAgent(question, get_agent_config())

        response = agent.update(question)
        assert response == commands.Retrieve(
            question="test query",
            q_id="test session id",
        )

    def test_agent_change_check_rejected(self):
        question = commands.Check(
            question="test query",
            q_id="test session id",
            approved=False,
            response="test response",
        )
        agent = BaseAgent(question, get_agent_config())

        response = agent.update(question)
        assert agent.is_answered is True
        assert response == events.RejectedRequest(
            question="test query",
            response="test response",
            q_id="test session id",
        )

    def test_agent_none_change_question(self):
        question = commands.Enhance(
            question="test query", q_id="test session id", response=None
        )
        agent = BaseAgent(question, get_agent_config())

        response = agent.update(question)
        assert response == commands.UseTools(
            question="test query",
            q_id="test session id",
        )
        assert agent.is_answered is False
        assert agent.previous_command is type(question)

    def test_agent_check_question(self):
        question = commands.Question(question="test query", q_id="test session id")
        agent = BaseAgent(question, get_agent_config())

        response = agent.update(question)
        assert response == commands.Check(
            question="test guardrails pre check", q_id="test session id"
        )

    @patch("src.agent.domain.model.BaseAgent.create_prompt")
    def test_agent_change_use_tools(self, mock_create_prompt):
        question = commands.UseTools(
            question="test query", q_id="test session id", response="test response"
        )

        mock_create_prompt.return_value = "test prompt"

        agent = BaseAgent(question, get_agent_config())

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

        agent = BaseAgent(question, get_agent_config())

        response = agent.update(question)
        assert response == commands.Enhance(
            question="test prompt", q_id="test session id"
        )

    def test_agent_change_retrieve(self):
        question = commands.Retrieve(
            question="test query", q_id="test session id", candidates=[]
        )
        agent = BaseAgent(question, get_agent_config())

        response = agent.update(question)
        assert response == commands.Rerank(
            question="test query", q_id="test session id", candidates=[]
        )

    def test_agent_final_check(self):
        question = commands.FinalCheck(
            question="test query",
            q_id="test session id",
            approved=True,
            summary="test summary",
            issues=[],
        )
        agent = BaseAgent(question, get_agent_config())

        agent.response = commands.LLMResponse(
            question="test query",
            q_id="test session id",
            response="test response",
            chain_of_thought="test chain of thought",
        )

        response = agent.update(question)

        assert agent.is_answered is True
        assert response is None
        assert agent.evaluation is not None
        assert agent.evaluation.response == "test response"
        assert type(agent.evaluation) is events.Evaluation

    def test_update_state(self):
        question = commands.Question(question="test query", q_id="test session id")

        agent = BaseAgent(question, get_agent_config())

        agent._update_state(question)

        assert agent.is_answered is False
        assert agent.response is None
        assert agent.previous_command is type(question)

    def test_update_state_duplicates(self):
        question = commands.Question(question="test query", q_id="test session id")

        agent = BaseAgent(question, get_agent_config())
        agent.previous_command = type(question)

        agent._update_state(question)

        assert agent.is_answered is True
        assert agent.response == events.FailedRequest(
            question=agent.question,
            exception="Internal error: Duplicate command",
            q_id=agent.q_id,
        )

    def test_is_answered_is_true(self):
        question = commands.Question(question="test query", q_id="test session id")
        agent = BaseAgent(question, get_agent_config())

        agent.is_answered = True

        response = agent.update(question)
        assert response is None

    def test_change_llm_response_without_tools(self):
        question = commands.Question(question="test query", q_id="test session id")
        agent = BaseAgent(question, get_agent_config())

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

    def test_update_not_implemented(self):
        question = commands.Question(question="test query", q_id="test session id")
        agent = BaseAgent(question, get_agent_config())

        command = commands.Command()

        with pytest.raises(
            NotImplementedError,
            match=f"Not implemented yet for BaseAgent: {type(command)}",
        ):
            agent.update(command)
