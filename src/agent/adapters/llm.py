from abc import ABC

import instructor
from litellm import completion
from pydantic import BaseModel


class AbstractLLM(ABC):
    def __init__(self):
        pass

    def use(self, question: str, response_model: BaseModel) -> BaseModel:
        pass


class LLM(AbstractLLM):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs

        self.model_id = kwargs.get("model_id")
        self.temperature = float(kwargs.get("temperature", 0.0))
        self.client = self.init_llm()

    def init_llm(self):
        client = instructor.from_litellm(completion)
        return client

    def use(self, question: str, response_model: BaseModel) -> BaseModel:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ]

        response = self.client.chat.completions.create(
            messages=messages,
            response_model=response_model,
            model=self.model_id,
            generation_config={
                "temperature": self.temperature,
            },
        )

        return response
