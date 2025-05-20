import httpx
import pytest
import responses

from .rag import BaseRAG


@pytest.fixture
def rag_instance():
    kwargs = {
        "embedding_url": "http://test-embedding-url",
        "ranking_url": "http://test-ranking-url",
        "retrieval_url": "http://test-retrieval-url",
        "n_retrieval_candidates": 5,
        "n_ranking_candidates": 3,
        "retrieval_table": "test_table"
    }
    return BaseRAG(kwargs)

@responses.activate
def test_call_api_successful_get(rag_instance):
    # Mock successful GET request
    test_url = "http://test-url"
    test_params = {"param1": "value1"}
    expected_response = {"status": "success"}

    responses.add(
        responses.GET,
        test_url,
        json=expected_response,
        status=200
    )

    response = rag_instance.call_api(test_url, test_params, "get")
    assert response is not None
    assert response.json() == expected_response

@responses.activate
def test_call_api_successful_post(rag_instance):
    # Mock successful POST request
    test_url = "http://test-url"
    test_body = {"data": "test"}
    expected_response = {"status": "success"}

    responses.add(
        responses.POST,
        test_url,
        json=expected_response,
        status=200
    )

    response = rag_instance.call_api(test_url, test_body, "post")
    assert response is not None
    assert response.json() == expected_response

@responses.activate
def test_call_api_http_error(rag_instance):
    # Mock HTTP error
    test_url = "http://test-url"
    responses.add(
        responses.GET,
        test_url,
        json={"error": "Not Found"},
        status=404
    )

    response = rag_instance.call_api(test_url)
    assert response is None

@responses.activate
def test_call_api_request_error(rag_instance):
    # Mock connection error
    test_url = "http://test-url"
    responses.add(
        responses.GET,
        test_url,
        body=httpx.RequestError("Connection error")
    )

    response = rag_instance.call_api(test_url)
    assert response is None

@responses.activate
def test_call_api_invalid_method(rag_instance):
    # Test invalid HTTP method
    test_url = "http://test-url"

    with pytest.raises(ValueError, match="Invalid method"):
        rag_instance.call_api(test_url, method="invalid")

@responses.activate
def test_call_api_json_decode_error(rag_instance):
    # Mock response with invalid JSON
    test_url = "http://test-url"
    responses.add(
        responses.GET,
        test_url,
        body="invalid json",
        status=200
    )

    response = rag_instance.call_api(test_url)
    assert response is None    response = rag_instance.call_api(test_url)
    assert response is None
