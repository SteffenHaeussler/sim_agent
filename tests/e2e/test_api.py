import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import MetaData

from src.agent.domain import commands
from src.agent.entrypoints.app import app

client = TestClient(app)


class TestAPI(unittest.TestCase):
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

        params = {"question": "test"}
        headers = {"X-Session-ID": "test-session-123"}
        response = client.get("/answer", params=params, headers=headers)

        assert response.status_code == 200

        assert response.json()["status"] == "processing"

    def test_unhappy_path_returns_400_and_answers(self):
        params = {"question": ""}
        headers = {"X-Session-ID": "test-session-123"}
        response = client.get("/answer", params=params, headers=headers)

        assert response.status_code == 400
        assert response.json()["detail"] == "No question asked"

    def test_missing_session_id_header_returns_400(self):
        params = {"question": "test question"}
        # Intentionally not providing X-Session-ID header
        response = client.get("/answer", params=params)

        assert response.status_code == 422
        assert response.json()["detail"] == [
            {
                "type": "missing",
                "loc": ["header", "X-Session-ID"],
                "msg": "Field required",
                "input": None,
            }
        ]

    @patch("src.agent.adapters.database.BaseDatabaseAdapter.execute_query")
    @patch("src.agent.adapters.llm.LLM.use")
    @patch("src.agent.adapters.database.BaseDatabaseAdapter.get_schema")
    def test_happy_path_returns_200_and_query(
        self,
        mock_get_schema,
        mock_LLM,
        mock_execute,
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

        params = {"question": "test"}
        headers = {"X-Session-ID": "test-session-123"}

        response = client.get("/query", params=params, headers=headers)

        assert response.status_code == 200

        assert response.json()["status"] == "processing"

    def test_unhappy_path_returns_400_and_query(self):
        params = {"question": ""}
        headers = {"X-Session-ID": "test-session-123"}
        response = client.get("/query", params=params, headers=headers)

        assert response.status_code == 400
        assert response.json()["detail"] == "No question asked"

    def test_missing_session_id_header_returns_400_query(self):
        params = {"question": "test question"}
        # Intentionally not providing X-Session-ID header
        response = client.get("/query", params=params)

        assert response.status_code == 422
        assert response.json()["detail"] == [
            {
                "type": "missing",
                "loc": ["header", "X-Session-ID"],
                "msg": "Field required",
                "input": None,
            }
        ]

    @patch("src.agent.adapters.llm.LLM.use")
    @patch("src.agent.adapters.database.BaseDatabaseAdapter.get_schema")
    def test_happy_path_returns_200_and_scenario(
        self,
        mock_get_schema,
        mock_LLM,
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

        params = {"question": "test"}
        headers = {"X-Session-ID": "test-session-123"}

        response = client.get("/scenario", params=params, headers=headers)

        assert response.status_code == 200

        assert response.json()["status"] == "processing"

    def test_unhappy_path_returns_400_and_scenario(self):
        params = {"question": ""}
        headers = {"X-Session-ID": "test-session-123"}
        response = client.get("/scenario", params=params, headers=headers)

        assert response.status_code == 400
        assert response.json()["detail"] == "No question asked"

    def test_missing_session_id_header_returns_400_scenario(self):
        params = {"question": "test question"}
        # Intentionally not providing X-Session-ID header
        response = client.get("/scenario", params=params)

        assert response.status_code == 422
        assert response.json()["detail"] == [
            {
                "type": "missing",
                "loc": ["header", "X-Session-ID"],
                "msg": "Field required",
                "input": None,
            }
        ]
