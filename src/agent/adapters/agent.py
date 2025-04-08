from abc import ABC

from src.agent.adapters import db, llm, model, tools
from src.agent.domain import commands, model


class AbstractAgent(ABC):
    def __init__(self):
        self.seen = set()
        return None

    def add(self, agent: model.Agent):
        self.seen.add(agent)

    def collect_new_events(self):
        for agent in self.seen:
            while agent.events:
                yield agent.events.pop(0)


class AgentAdapter(AbstractAgent):
    def __init__(self):
        super().__init__()
        self.db = db.AbstractDB()
        self.llm = llm.AbstractLLM()
        self.tools = tools.AbstractTools()
        self.rerank = model.AbstractRerank()
        self.retrieve = db.AbstractDB()

    def answer(self, command: commands.Command) -> str:
        if type(command) is commands.Check:
            response = self.check(command.question)
        elif type(command) is commands.Question:
            response = self.use(command.question)
        elif type(command) is commands.Retrieve:
            response = self.retrieve(command.question)
        elif type(command) is commands.Rerank:
            response = self.rerank(command.question)
        else:
            raise NotImplementedError("Not implemented yet")
        return response

    def check(self, question: str) -> str:
        response = self.llm(question)
        return response

    def read(self, question):
        response = self.db.read(question)
        return response

    def use(self, question):
        response = self.tools.use(question)
        return response

    def retrieve(self, question):
        response = self.retrieve.retrieve(question)
        return response

    def rerank(self, question):
        response = self.rerank.rerank(question)
        return response
