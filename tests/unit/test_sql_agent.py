import pandas as pd
import pytest

from src.agent.config import get_agent_config
from src.agent.domain import commands, events
from src.agent.domain.sql_model import SQLBaseAgent


class TestSQLAgent:
    def test_agent_initialization(self):
        question = commands.SQLQuestion(question="test query", q_id="test session id")
        agent = SQLBaseAgent(question, get_agent_config())

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

    def test_agent_change_execution(self):
        question = commands.SQLQuestion(question="test query", q_id="test session id")
        agent = SQLBaseAgent(question, get_agent_config())

        sql_query = "SELECT * FROM test_table"
        agent.sql_query = sql_query

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

        construction = commands.SQLConstruction(
            question="test query",
            q_id="test session id",
            schema_info=schema,
        )
        agent.construction = construction

        command = commands.SQLExecution(
            question="test query",
            q_id="test session id",
            sql_query=sql_query,
            data={"data": pd.DataFrame({"test": [1, 2, 3]})},
        )
        new_command = agent.update(command)

        assert agent.is_answered is False
        assert (
            agent.response.response
            == "|   test |\n|-------:|\n|      1 |\n|      2 |\n|      3 |"
        )
        assert agent.previous_command is type(command)

        assert new_command == commands.SQLValidation(
            question="test validate prompt",
            q_id="test session id",
            sql_query=sql_query,
            tables=schema.tables,
            relationships=schema.relationships,
        )

    def test_agent_change_execution_no_data(self):
        question = commands.SQLQuestion(question="test query", q_id="test session id")
        agent = SQLBaseAgent(question, get_agent_config())

        sql_query = "SELECT * FROM test_table"
        agent.sql_query = sql_query

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

        construction = commands.SQLConstruction(
            question="test query",
            q_id="test session id",
            schema_info=schema,
        )
        agent.construction = construction

        command = commands.SQLExecution(
            question="test query",
            q_id="test session id",
            sql_query=sql_query,
            data={"data": None},
        )
        new_command = agent.update(command)

        assert agent.is_answered is False
        assert agent.response.response == ""
        assert agent.previous_command is type(command)

        assert new_command == commands.SQLValidation(
            question="test validate prompt",
            q_id="test session id",
            sql_query=sql_query,
            tables=schema.tables,
            relationships=schema.relationships,
        )

    def test_agent_change_execution_bad_data(self):
        question = commands.SQLQuestion(question="test query", q_id="test session id")
        agent = SQLBaseAgent(question, get_agent_config())

        sql_query = "SELECT * FROM test_table"
        agent.sql_query = sql_query

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

        construction = commands.SQLConstruction(
            question="test query",
            q_id="test session id",
            schema_info=schema,
        )
        agent.construction = construction

        command = commands.SQLExecution(
            question="test query",
            q_id="test session id",
            sql_query=sql_query,
            data={"data": "test data"},
        )
        new_command = agent.update(command)

        assert agent.is_answered is False
        assert agent.response.response == "No data available"
        assert agent.previous_command is type(command)

        assert new_command == commands.SQLValidation(
            question="test validate prompt",
            q_id="test session id",
            sql_query=sql_query,
            tables=schema.tables,
            relationships=schema.relationships,
        )

    def test_agent_prepare_check(self):
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

        command = commands.SQLQuestion(
            question="test query",
            q_id="test session id",
        )

        agent = SQLBaseAgent(command, get_agent_config())

        command.schema_info = schema

        new_command = agent.update(command)

        assert new_command == commands.SQLCheck(
            question="test check prompt",
            q_id="test session id",
        )

    def test_agent_change_check_approved(self):
        question = commands.SQLCheck(
            question="test query", q_id="test session id", approved=True
        )

        agent = SQLBaseAgent(question, get_agent_config())

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

        construction = commands.SQLConstruction(
            question="test query",
            q_id="test session id",
            schema_info=schema,
        )
        agent.construction = construction

        response = agent.update(question)
        assert response == commands.SQLGrounding(
            question="test grounding prompt",
            q_id="test session id",
            tables=schema.tables,
            table_mapping=None,
            column_mapping=None,
        )

    def test_agent_change_check_rejected(self):
        question = commands.SQLCheck(
            question="test query",
            q_id="test session id",
            approved=False,
            response="test response",
        )
        agent = SQLBaseAgent(question, get_agent_config())

        response = agent.update(question)
        assert agent.is_answered is True
        assert response == events.RejectedRequest(
            question="test query",
            response="test response",
            q_id="test session id",
        )

    def test_agent_change_grounding(self):
        table_mapping = commands.TableMapping(
            question_term="test_table",
            table_name="test_table",
            confidence=1.0,
        )

        column_mapping = commands.ColumnMapping(
            question_term="id",
            table_name="test_table",
            column_name="id",
            confidence=1.0,
        )

        table = commands.Table(
            name="test_table",
            columns=[
                commands.Column(name="id", type="int"),
                commands.Column(name="name", type="str"),
            ],
        )

        command = commands.SQLGrounding(
            question="test query",
            q_id="test session id",
            tables=[table],
            table_mapping=[table_mapping],
            column_mapping=[column_mapping],
        )

        question = commands.SQLQuestion(question="test query", q_id="test session id")

        agent = SQLBaseAgent(question, get_agent_config())

        response = agent.update(command)

        assert response == commands.SQLFilter(
            question="test filter prompt",
            q_id="test session id",
            column_mapping=[column_mapping],
        )

    def test_agent_change_filter(self):
        table_mapping = commands.TableMapping(
            question_term="test_table",
            table_name="test_table",
            confidence=1.0,
        )

        column_mapping = commands.ColumnMapping(
            question_term="id",
            table_name="test_table",
            column_name="id",
            confidence=1.0,
        )

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

        condition = commands.FilterCondition(
            column="id",
            operator="=",
            value="1",
        )

        command = commands.SQLFilter(
            question="test query",
            q_id="test session id",
            column_mapping=[column_mapping],
            conditions=[condition],
        )

        question = commands.SQLQuestion(question="test query", q_id="test session id")

        agent = SQLBaseAgent(question, get_agent_config())

        agent.construction.table_mapping = [table_mapping]
        agent.construction.schema_info = schema

        response = agent.update(command)

        assert response == commands.SQLJoinInference(
            question="test join prompt",
            q_id="test session id",
            table_mapping=[table_mapping],
            relationships=schema.relationships,
            joins=None,
        )

    def test_agent_change_join_inference(self):
        table_mapping = commands.TableMapping(
            question_term="test_table",
            table_name="test_table",
            confidence=1.0,
        )

        column_mapping = commands.ColumnMapping(
            question_term="id",
            table_name="test_table",
            column_name="id",
            confidence=1.0,
        )

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

        join = commands.JoinPath(
            from_table="test_table",
            to_table="test_table",
            from_column="id",
            to_column="id",
            join_type="INNER",
        )

        command = commands.SQLJoinInference(
            question="test query",
            q_id="test session id",
            table_mapping=[table_mapping],
            relationships=schema.relationships,
            joins=[join],
        )

        question = commands.SQLQuestion(question="test query", q_id="test session id")

        agent = SQLBaseAgent(question, get_agent_config())

        agent.construction.column_mapping = [column_mapping]

        response = agent.update(command)

        assert response == commands.SQLAggregation(
            question="test aggregate prompt",
            q_id="test session id",
            column_mapping=[column_mapping],
            aggregations=None,
            group_by_columns=None,
            is_aggregation_query=None,
        )

    def test_agent_change_aggregation(self):
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

        aggregation = commands.AggregationFunction(
            function="SUM",
            column="id",
            alias="sum_id",
        )

        column_mapping = commands.ColumnMapping(
            question_term="id",
            table_name="test_table",
            column_name="id",
            confidence=1.0,
        )

        question = commands.SQLAggregation(
            question="test query",
            q_id="test session id",
            column_mapping=[column_mapping],
            aggregations=[aggregation],
            group_by_columns=["test"],
            is_aggregation_query=False,
        )
        agent = SQLBaseAgent(question, get_agent_config())

        agent.construction.schema_info = schema

        response = agent.update(question)

        assert response == commands.SQLConstruction(
            question="test construction prompt",
            q_id="test session id",
            schema_info=schema,
            column_mapping=None,
            table_mapping=None,
            aggregations=[aggregation],
            group_by_columns=["test"],
            is_aggregation_query=False,
            sql_query=None,
        )

    def test_agent_change_construction(self):
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

        question = commands.SQLConstruction(
            question="test query",
            q_id="test session id",
            schema_info=schema,
            sql_query="SELECT * FROM test_table",
        )
        agent = SQLBaseAgent(question, get_agent_config())

        response = agent.update(question)
        assert response == commands.SQLExecution(
            question="test query",
            q_id="test session id",
            sql_query="SELECT * FROM test_table",
        )

    def test_agent_final_check(self):
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

        question = commands.SQLValidation(
            question="test query",
            q_id="test session id",
            approved=True,
            summary="test summary",
            issues=[],
            tables=schema.tables,
            relationships=schema.relationships,
            sql_query="SELECT * FROM test_table",
        )
        agent = SQLBaseAgent(question, get_agent_config())
        agent.sql_query = "SELECT * FROM test_table"

        agent.response = events.Response(
            question="test query",
            response="test response",
            q_id="test session id",
        )

        response = agent.update(question)

        assert agent.is_answered is True
        assert response is None
        assert agent.response.response == "test response"
        assert type(agent.response) is events.Response

    def test_update_state(self):
        question = commands.SQLQuestion(question="test query", q_id="test session id")

        agent = SQLBaseAgent(question, get_agent_config())

        agent._update_state(question)

        assert agent.is_answered is False
        assert agent.response is None
        assert agent.previous_command is type(question)

    def test_update_state_duplicates(self):
        question = commands.SQLQuestion(question="test query", q_id="test session id")

        agent = SQLBaseAgent(question, get_agent_config())
        agent.previous_command = type(question)

        agent._update_state(question)

        assert agent.is_answered is True
        assert agent.response == events.FailedRequest(
            question=agent.question,
            exception="Internal error: Duplicate command",
            q_id=agent.q_id,
        )

    def test_is_answered_is_true(self):
        question = commands.SQLQuestion(question="test query", q_id="test session id")
        agent = SQLBaseAgent(question, get_agent_config())

        agent.is_answered = True

        response = agent.update(question)
        assert response is None

    def test_update_not_implemented(self):
        question = commands.SQLQuestion(question="test query", q_id="test session id")
        agent = SQLBaseAgent(question, get_agent_config())

        command = commands.Command()

        with pytest.raises(
            NotImplementedError,
            match=f"Not implemented yet for BaseAgent: {type(command)}",
        ):
            agent.update(command)
