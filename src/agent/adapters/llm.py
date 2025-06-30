from abc import ABC

import instructor
from langfuse import get_client, observe
from litellm import completion
from pydantic import BaseModel

from src.agent.observability.context import ctx_query_id


class AbstractLLM(ABC):
    """
    AbstractLLM is an abstract base class for all LLM models.

    Methods:
        - use(self, question: str, response_model: BaseModel) -> BaseModel: Use the LLM model.
    """

    def __init__(self):
        """
        Initialize the LLM model.

        Args:
            kwargs: Dict: The kwargs.

        Returns:
            None
        """
        pass

    def use(self, question: str, response_model: BaseModel) -> BaseModel:
        """
        Calls the LLM model with the given question and response model.

        Args:
            question: str: The question to use the LLM model.
            response_model: BaseModel: The response model.

        Returns:
            response: BaseModel: The response from the LLM model.
        """
        pass


class LLM(AbstractLLM):
    def __init__(self, kwargs):
        """
        Initialize the LLM model.

        Args:
            kwargs: Dict: The kwargs.

        Returns:
            None
        """
        self.model_id = kwargs["model_id"]
        self.temperature = float(kwargs["temperature"])
        self.client = self.init_llm()

    def init_llm(self):
        """
        Initialize the LLM model.

        Returns:
            client: instructor.from_litellm: The LLM model.
        """
        client = instructor.from_litellm(completion)
        return client

    @observe(as_type="generation")
    def use(self, question: str, response_model: BaseModel) -> BaseModel:
        """
        Calls the LLM model with the given question and response model.

        Args:
            question: str: The question to use the LLM model.
            response_model: BaseModel: The response model.

        Returns:
            response: BaseModel: The response from the LLM model.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ]

        langfuse = get_client()

        langfuse.update_current_trace(
            name="llm_call",
            input=messages.copy(),
            metadata={"temperature": self.temperature, "model": self.model_id},
            session_id=ctx_query_id.get(),
        )

        response = self.client.chat.completions.create(
            messages=messages,
            response_model=response_model,
            model=self.model_id,
            temperature=self.temperature,
        )

        return response
