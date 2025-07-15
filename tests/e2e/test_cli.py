import unittest
from unittest.mock import patch

from sqlalchemy import MetaData

from src.agent.domain import commands
from src.agent.entrypoints.main import answer


class TestCLI(unittest.TestCase):
    @patch("src.agent.adapters.notifications.SlackNotifications.send")
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
        mock_slack,
    ):
        mock_CodeAgent.return_value = ("agent test", "agent memory")

        mock_LLM.side_effect = [
            commands.GuardrailPreCheckModel(
                approved=True,
                chain_of_thought="chain_of_thought",
                response="test answer",
            ),
            commands.LLMResponseModel(
                response="test answer", chain_of_thought="chain_of_thought"
            ),
            commands.LLMResponseModel(
                response="test answer", chain_of_thought="chain_of_thought"
            ),
            commands.GuardrailPostCheckModel(
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

        mock_slack.return_value = None

        question = "test"

        response = answer(question, "test_session_id")

        assert response == "done"

    def test_unhappy_path_returns_400_and_answers(
        self,
    ):
        question = ""
        try:
            answer(question, "test_session_id")
        except Exception as e:
            assert str(e) == "No question asked"

    @patch("src.agent.adapters.notifications.SlackNotifications.send")
    @patch("src.agent.adapters.database.BaseDatabaseAdapter.execute_query")
    @patch("src.agent.adapters.llm.LLM.use")
    @patch("src.agent.adapters.database.BaseDatabaseAdapter.get_schema")
    @patch("src.agent.adapters.rag.BaseRAG.retrieve")
    @patch("src.agent.adapters.rag.BaseRAG.rerank")
    @patch("src.agent.adapters.rag.BaseRAG.embed")
    def test_happy_path_returns_200_and_query(
        self,
        mock_embed,
        mock_rerank,
        mock_retrieve,
        mock_get_schema,
        mock_LLM,
        mock_execute,
        mock_slack,
    ):
        # Create a proper SQLAlchemy MetaData object
        metadata = MetaData()
        mock_get_schema.return_value = metadata

        mock_LLM.side_effect = [
            commands.GuardrailPreCheckModel(
                approved=True,
                chain_of_thought="chain_of_thought",
                response="test answer",
            ),
            commands.GroundingResponse(
                table_mapping=[
                    commands.TableMapping(
                        question_term="test_question",
                        table_name="test_table",
                        confidence=0.5,
                    )
                ],
                column_mapping=[
                    commands.ColumnMapping(
                        question_term="test_question",
                        table_name="test_table",
                        column_name="id",
                        confidence=0.5,
                    )
                ],
                chain_of_thought="chain_of_thought",
            ),
            commands.FilterResponse(
                conditions=[
                    commands.FilterCondition(
                        column="id",
                        operator="=",
                        value="1",
                        chain_of_thought="chain_of_thought",
                    )
                ],
                chain_of_thought="chain_of_thought",
            ),
            commands.JoinInferenceResponse(
                joins=[
                    commands.JoinPath(
                        from_table="test_table",
                        to_table="test_table",
                        from_column="id",
                        to_column="id",
                        join_type="INNER",
                    )
                ],
                chain_of_thought="chain_of_thought",
            ),
            commands.AggregationResponse(
                aggregations=[
                    commands.AggregationFunction(
                        function="COUNT",
                        column="id",
                        alias="count_id",
                    )
                ],
                group_by_columns=["id"],
                is_aggregation_query=True,
                chain_of_thought="chain_of_thought",
            ),
            commands.ConstructionResponse(
                sql_query="SELECT COUNT(*) FROM test_table",
                chain_of_thought="chain_of_thought",
            ),
            commands.ValidationResponse(
                approved=True,
                issues=[],
                summary="summary",
                confidence=0.5,
                chain_of_thought="chain_of_thought",
            ),
        ]

        mock_execute.return_value = {"results": [{"id": 1, "name": "test_name"}]}

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

        mock_slack.return_value = None

        question = "test"
        response = answer(question, "test_session_id", "sql")

        assert response == "done"

    def test_unhappy_path_returns_400_and_query(
        self,
    ):
        question = ""
        try:
            answer(question, "test_session_id", "sql")
        except Exception as e:
            assert str(e) == "No question asked"

    @patch("src.agent.adapters.notifications.SlackNotifications.send")
    @patch("src.agent.adapters.llm.LLM.use")
    @patch("src.agent.adapters.database.BaseDatabaseAdapter.get_schema")
    def test_happy_path_returns_200_and_scenario(
        self,
        mock_get_schema,
        mock_LLM,
        mock_slack,
    ):
        # Create a proper SQLAlchemy MetaData object
        metadata = MetaData()
        mock_get_schema.return_value = metadata

        mock_LLM.side_effect = [
            commands.GuardrailPreCheckModel(
                approved=True,
                chain_of_thought="chain_of_thought",
                response="test answer",
            ),
            commands.ScenarioResponse(
                candidates=[
                    commands.ScenarioCandidate(
                        question="test_question",
                        endpoint="test_endpoint",
                    )
                ],
                chain_of_thought="chain_of_thought",
            ),
            commands.ScenarioValidationResponse(
                approved=True,
                issues=[],
                summary="summary",
                confidence=0.5,
                chain_of_thought="chain_of_thought",
            ),
        ]

        mock_slack.return_value = None

        question = "test"
        response = answer(question, "test_session_id", "scenario")

        assert response == "done"

    def test_unhappy_path_returns_400_and_scenario(
        self,
    ):
        question = ""
        try:
            answer(question, "test_session_id", "scenario")
        except Exception as e:
            assert str(e) == "No question asked"
