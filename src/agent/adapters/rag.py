from abc import ABC
from typing import Dict, List, Optional

import httpx
from loguru import logger


class AbstractModel(ABC):
    """
    AbstractModel is an abstract base class for all RAG models.

    Methods:
        - rerank(self) -> None: Rerank the text.
        - retrieve(self) -> None: Retrieve the text.
    """

    def __init__(self):
        pass

    def embed(self, text: str) -> Dict[str, List[float]]:
        pass

    def rerank(self, question: str, text: str) -> Dict[str, str]:
        pass

    def retrieve(self, embedding: list[float]) -> Dict[str, List[str]]:
        pass


class BaseRAG(AbstractModel):
    """
    BaseRAG is a class that implements the AbstractModel interface.

    Methods:
        - embed(self, text: str) -> Dict[str, List[float]]: Embed the text.
        - rerank(self, question: str, text: str) -> Dict[str, str]: Rerank the text.
        - retrieve(self, embedding: list[float]) -> Dict[str, List[str]]: Retrieve the text.
        - call_api(self, api_url, body={}, method="get") -> None: Call the API.
    """

    def __init__(self, kwargs):
        """
        Initialize the BaseRAG model.

        Args:
            kwargs: Dict: The kwargs.
        """
        super().__init__()
        self.kwargs = kwargs

        self.embedding_url = kwargs["embedding_url"]
        self.ranking_url = kwargs["ranking_url"]
        self.retrieval_url = kwargs["retrieval_url"]
        self.n_retrieval_candidates = int(kwargs["n_retrieval_candidates"])
        self.n_ranking_candidates = int(kwargs["n_ranking_candidates"])
        self.retrieval_table = kwargs["retrieval_table"]

    def embed(self, text: str) -> Dict[str, List[float]]:
        """
        Embed the text.

        Args:
            text: str: The text to embed.

        Returns:
            response: Dict[str, List[float]]: The response from the embedding API.
        """

        response = self.call_api(self.embedding_url, {"text": text})

        return response.json() if response else None

    def rerank(self, question: str, text: str) -> Dict[str, str]:
        """
        Rerank the text.

        Args:
            question: str: The question to rerank the text.
            text: str: The text to rerank.
        """
        response = self.call_api(
            self.ranking_url,
            {"text": text, "question": question, "table": self.retrieval_table},
        )

        return response.json() if response else None

    def retrieve(self, embedding: list[float]) -> Dict[str, List[str]]:
        """
        Retrieve the text.

        Args:
            embedding: list[float]: The embedding to retrieve the text.
        """
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

    def call_api(
        self, api_url: str, body: Dict = {}, method: str = "get"
    ) -> Optional[httpx.Response]:
        """
        Calls the RAG API. Errors are ignored and returned as None.

        Args:
            api_url: str: The API URL.
            body: Dict: The body of the request.
            method: str: The method of the request.

        Returns:
            response: httpx.Response: The response from the API.
        """
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
        except Exception as e:  # Catch any other unexpected errors
            logger.debug(f"An unexpected error occurred for {api_url}: {e}")

        return None
