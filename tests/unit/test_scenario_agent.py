import json

import pytest

from src.agent.config import get_agent_config
from src.agent.domain import commands, events
from src.agent.domain.scenario_model import ScenarioBaseAgent


class TestScenarioAgent:
    def test_agent_initialization(self):
        question = commands.Scenario(question="test query", q_id="test session id")
        agent = ScenarioBaseAgent(question, get_agent_config())

        assert agent.question == "test query"
        assert agent.q_id == "test session id"
        assert agent.evaluation is None
        assert agent.response is None
        assert agent.send_response is None
        assert agent.is_answered is False
        assert agent.previous_command is None
        assert agent.kwargs is not None
        assert agent.events == []
        assert agent.base_prompts is not None
        assert agent.sql_query is None

    def test_agent_prepare_check(self):
        question = commands.Scenario(question="test query", q_id="test session id")
        agent = ScenarioBaseAgent(question, get_agent_config())

        schema = commands.DatabaseSchema(
            tables=[
                commands.Table(
                    name="test_table",
                    columns=[
                        commands.Column(name="id", type="int"),
                        commands.Column(name="name", type="str"),
                    ],
                ),
            ],
            relationships=[
                commands.Relationship(
                    table_name="test_table",
                    column_name="id",
                    foreign_table_name="test_table",
                    foreign_column_name="id",
                )
            ],
        )

        question.schema_info = schema

        new_command = agent.update(question)

        assert agent.is_answered is False
        assert new_command == commands.Check(
            question="test check prompt",
            q_id="test session id",
            approved=None,
        )

    def test_agent_change_check(self):
        schema = commands.DatabaseSchema(
            tables=[
                commands.Table(
                    name="test_table", columns=[commands.Column(name="id", type="int")]
                ),
            ],
            relationships=[],
        )

        question = commands.Check(
            question="test query",
            q_id="test session id",
            approved=False,
            response="test response",
        )
        agent = ScenarioBaseAgent(question, get_agent_config())

        agent.scenario.schema_info = schema

        response = agent.update(question)
        assert response == commands.ScenarioLLMResponse(
            question="test response prompt",
            q_id="test session id",
            tables=schema.tables,
            tools=[
                "id_to_name: Converts asset ids to the asset names.",
                "name_to_id: Converts asset names to ids.",
                "compare_data: Compare data from two assets.",
                "final_answer: Provides a final answer to the given problem.",
                "get_data: Get data from an asset.",
                "asset_information: Get information about an asset.",
                "get_neighbors: Get neighbors of an asset.",
                "plot_data: Plot data from data.",
                "export_data: Stores data in external cloud storage.",
            ],
            candidates=None,
        )

    def test_agent_final_candidates_response(self):
        question = commands.Scenario(question="test query", q_id="test session id")

        agent = ScenarioBaseAgent(question, get_agent_config())

        candidate = commands.ScenarioCandidate(
            question="test question",
            endpoint="test endpoint",
        )

        command = commands.ScenarioLLMResponse(
            question="test query",
            q_id="test session id",
            candidates=[candidate],
        )
        new_command = agent.update(command)
        assert new_command == commands.ScenarioFinalCheck(
            question="test final check prompt",
            q_id="test session id",
            candidates=[candidate],
        )

        assert agent.response == events.Response(
            question="test query",
            response=json.dumps(
                [
                    {
                        "sub_id": "sub-1",
                        "question": "test question",
                        "endpoint": "test endpoint",
                    }
                ]
            ),
            q_id="test session id",
        )

    def test_agent_final_no_candidates_response(self):
        question = commands.Scenario(question="test query", q_id="test session id")

        agent = ScenarioBaseAgent(question, get_agent_config())

        command = commands.ScenarioLLMResponse(
            question="test query",
            q_id="test session id",
            candidates=None,
        )
        new_command = agent.update(command)
        assert new_command == commands.ScenarioFinalCheck(
            question="test final check prompt",
            q_id="test session id",
            candidates=None,
        )

        assert agent.response == events.Response(
            question="test query",
            response="No candidates found",
            q_id="test session id",
        )

    def test_agent_final_check(self):
        question = commands.ScenarioFinalCheck(
            question="test query",
            q_id="test session id",
            approved=True,
            summary="test summary",
            issues=[],
            plausibility="test plausibility",
            usefulness="test usefulness",
            clarity="test clarity",
        )
        agent = ScenarioBaseAgent(question, get_agent_config())
        agent.sql_query = "SELECT * FROM test_table"

        agent.response = events.Evaluation(
            question="test query",
            response="test response",
            q_id="test session id",
            approved=True,
            summary="test summary",
            issues=[],
        )

        response = agent.update(question)

        assert agent.is_answered is True
        assert response is None
        assert agent.response.response == "test response"
        assert type(agent.response) is events.Evaluation

    def test_update_state(self):
        question = commands.Scenario(question="test query", q_id="test session id")

        agent = ScenarioBaseAgent(question, get_agent_config())

        agent._update_state(question)

        assert agent.is_answered is False
        assert agent.response is None
        assert agent.previous_command is type(question)

    def test_update_state_duplicates(self):
        question = commands.Scenario(question="test query", q_id="test session id")

        agent = ScenarioBaseAgent(question, get_agent_config())
        agent.previous_command = type(question)

        agent._update_state(question)

        assert agent.is_answered is True
        assert agent.response == events.FailedRequest(
            question=agent.question,
            exception="Internal error: Duplicate command",
            q_id=agent.q_id,
        )

    def test_is_answered_is_true(self):
        question = commands.Scenario(question="test query", q_id="test session id")
        agent = ScenarioBaseAgent(question, get_agent_config())

        agent.is_answered = True

        response = agent.update(question)
        assert response is None

    def test_update_not_implemented(self):
        question = commands.Scenario(question="test query", q_id="test session id")
        agent = ScenarioBaseAgent(question, get_agent_config())

        command = commands.Command()

        with pytest.raises(
            NotImplementedError,
            match=f"Not implemented yet for BaseAgent: {type(command)}",
        ):
            agent.update(command)
