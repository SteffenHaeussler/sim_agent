from abc import ABC

import instructor
from langfuse.decorators import langfuse_context, observe
from litellm import completion
from pydantic import BaseModel

from src.agent.observability.context import ctx_query_id


class AbstractLLM(ABC):
    def __init__(self):
        pass

    def use(self, question: str, response_model: BaseModel) -> BaseModel:
        pass


class LLM(AbstractLLM):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs

        self.model_id = kwargs["model_id"]
        self.temperature = float(kwargs["temperature"])
        self.client = self.init_llm()

    def init_llm(self):
        client = instructor.from_litellm(completion)
        return client

    @observe(as_type="generation")
    def use(self, question: str, response_model: BaseModel) -> BaseModel:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ]

        langfuse_context.update_current_observation(
            name="llm_call",
            input=messages.copy(),
            model=self.model_id,
            metadata={"temperature": self.temperature},
            session_id=ctx_query_id.get(),
        )

        response = self.client.chat.completions.create(
            messages=messages,
            response_model=response_model,
            model=self.model_id,
            generation_config={
                "temperature": self.temperature,
            },
        )

        return response
