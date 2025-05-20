from unittest.mock import Mock, patch

import httpx
import pytest

from src.agent.adapters.rag import BaseRAG


@pytest.fixture
def rag_instance():
    kwargs = {
        "embedding_url": "http://test-embedding-url",
        "ranking_url": "http://test-ranking-url",
        "retrieval_url": "http://test-retrieval-url",
        "n_retrieval_candidates": 5,
        "n_ranking_candidates": 3,
        "retrieval_table": "test_table",
    }
    return BaseRAG(kwargs)


class TestRAG:
    def test_rag_init(self, rag_instance):
        assert rag_instance.embedding_url == "http://test-embedding-url"
        assert rag_instance.ranking_url == "http://test-ranking-url"
        assert rag_instance.retrieval_url == "http://test-retrieval-url"
        assert rag_instance.n_retrieval_candidates == 5
        assert rag_instance.n_ranking_candidates == 3
        assert rag_instance.retrieval_table == "test_table"

    @patch("src.agent.adapters.rag.httpx.get")
    def test_embed(self, mock_get, rag_instance):
        mock_get.return_value = Mock(
            status_code=200, json=lambda: {"embedding": [0.1, 0.2, 0.3]}
        )
        response = rag_instance.embed("test_text")

        assert response == {"embedding": [0.1, 0.2, 0.3]}

    @patch("src.agent.adapters.rag.httpx.get")
    def test_rerank(self, mock_get, rag_instance):
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: {
                "question": "test_question",
                "text": "test_text",
                "score": 0.5,
                "id": "test_id",
                "tag": "test_tag",
                "name": "test_name",
            },
        )
        response = rag_instance.rerank("test_question", "test_text")

        assert response == {
            "question": "test_question",
            "text": "test_text",
            "score": 0.5,
            "id": "test_id",
            "tag": "test_tag",
            "name": "test_name",
        }

    @patch("src.agent.adapters.rag.httpx.post")
    def test_retrieve(self, mock_post, rag_instance):
        mock_post.return_value = Mock(
            status_code=200,
            json=lambda: {
                "items": [
                    {
                        "id": "test_id",
                        "score": 0.5,
                        "description": "test_text",
                        "tag": "test_tag",
                        "name": "test_name",
                    },
                ]
            },
        )
        response = rag_instance.retrieve([0.1, 0.2, 0.3])

        assert response == {
            "items": [
                {
                    "id": "test_id",
                    "score": 0.5,
                    "description": "test_text",
                    "tag": "test_tag",
                    "name": "test_name",
                },
            ]
        }

    def test_http_status_error(self, rag_instance):
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Error",
            request=Mock(),
            response=Mock(status_code=500, text="Internal Server Error"),
        )

        with patch("httpx.get", return_value=mock_response):
            result = rag_instance.embed("some text")
            assert result is None

    def test_request_error(self, rag_instance):
        with patch(
            "httpx.get", side_effect=httpx.RequestError("Network Error", request=Mock())
        ):
            result = rag_instance.embed("some text")
            assert result is None
