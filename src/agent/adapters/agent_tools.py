from abc import ABC
from typing import Dict


class AbstractTools(ABC):
    def __init__(self):
        pass

    def use(self):
        pass


class Tools(AbstractTools):
    def __init__(
        self,
        args: str,
        kwargs: Dict,
    ):
        self.args = args
        self.kwargs = kwargs
        self.code_agent = self.init_agent()

    def init_agent(self):
        raise NotImplementedError("Tools for init_agent method must be implemented.")

    def use(self, question):
        response = self.code_agent.run(question)
        return response
