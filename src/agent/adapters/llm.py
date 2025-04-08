from abc import ABC


class AbstractLLM(ABC):
    def __init__(self):
        pass

    def __call__(self, question):
        pass
