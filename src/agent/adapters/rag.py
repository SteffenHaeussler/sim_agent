import json
from abc import ABC

import httpx
from loguru import logger


class AbstractModel(ABC):
    def __init__(self):
        pass

    def rerank(self):
        pass

    def retrieve(self):
        pass


class BaseRAG(AbstractModel):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs

        self.embedding_url = kwargs["embedding_url"]
        self.ranking_url = kwargs["ranking_url"]
        self.retrieval_url = kwargs["retrieval_url"]
        self.n_retrieval_candidates = int(kwargs["n_retrieval_candidates"])
        self.n_ranking_candidates = int(kwargs["n_ranking_candidates"])
        self.retrieval_table = kwargs["retrieval_table"]

    def embed(self, text: str):
        response = self.call_api(self.embedding_url, {"text": text})

        return response.json() if response else None

    def rerank(self, question: str, text: str):
        response = self.call_api(
            self.ranking_url,
            {"text": text, "question": question, "table": self.retrieval_table},
        )

        return response.json() if response else None

    def retrieve(self, embedding: list[float]):
        response = self.call_api(
            self.retrieval_url,
            {
                "embedding": embedding,
                "n_items": self.n_retrieval_candidates,
                "table": self.retrieval_table,
            },
            method="post",
        )

        return response.json() if response else None

    def call_api(self, api_url, body={}, method="get"):
        try:
            if method == "get":
                response = httpx.get(api_url, params=body, timeout=30.0)
            elif method == "post":
                response = httpx.post(api_url, json=body, timeout=30.0)
            else:
                raise ValueError("Invalid method")

            response.raise_for_status()
            return response

        except httpx.HTTPStatusError as e:
            logger.debug(
                f"HTTP error fetching name for {api_url}: {e.response.status_code} - {e.response.text}"
            )
        except httpx.RequestError as e:
            logger.debug(f"Request error fetching name for {api_url}: {e}")
        except json.JSONDecodeError as e:
            logger.debug(
                f"JSON decode error for {api_url}. Response text: {response.text}. Error: {e}"
            )
        except Exception as e:  # Catch any other unexpected errors
            logger.debug(f"An unexpected error occurred for {api_url}: {e}")

        return None
