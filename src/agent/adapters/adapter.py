from abc import ABC

from langfuse.decorators import langfuse_context, observe

from src.agent import config
from src.agent.adapters import agent_tools, db, llm, transformer
from src.agent.domain import commands, model


class AbstractAdapter(ABC):
    def __init__(self):
        self.seen = set()
        self.db = db.AbstractDB()
        self.llm = llm.AbstractLLM()
        self.tools = agent_tools.AbstractTools()
        self.rerank = transformer.AbstractModel()
        self.retrieve = db.AbstractDB()

    def add(self, agent: model.BaseAgent):
        self.seen.add(agent)

    def collect_new_events(self):
        for agent in self.seen:
            while agent.events:
                yield agent.events.pop(0)

    def answer(self, command: commands.Command) -> str:
        raise NotImplementedError("Not implemented yet")


class AgentAdapter(AbstractAdapter):
    def __init__(self):
        super().__init__()

        self.tools = agent_tools.Tools(
            kwargs=config.get_tools_config(),
        )
        self.llm = llm.LLM(
            kwargs=config.get_llm_config(),
        )

    def answer(self, command: commands.Command) -> commands.Command:
        if type(command) is commands.Question:
            response = self.question(command)
        # if type(command) is commands.Check:
        #     response = self.check(command.question)
        elif type(command) is commands.UseTools:
            response = self.use(command)
        # elif type(command) is commands.Retrieve:
        #     response = self.retrieve(command.question)
        # elif type(command) is commands.Rerank:
        #     response = self.rerank(command.question)
        elif type(command) is commands.LLMResponse:
            response = self.finalize(command)
        else:
            raise NotImplementedError(
                f"Not implemented in AgentAdapter: {type(command)}"
            )
        return response

    # def check(self, question: str) -> str:
    #     response = self.llm(question)
    #     return response

    @observe()
    def finalize(self, command: commands.LLMResponse) -> commands.LLMResponse:
        langfuse_context.update_current_trace(
            name="finalize",
            session_id=command.q_id,
        )

        response = self.llm.use(command.question, commands.LLMResponseModel)

        command.response = response.response
        command.chain_of_thought = response.chain_of_thought

        return command

    @observe()
    def question(self, command: commands.Question) -> commands.Question:
        langfuse_context.update_current_trace(
            name="question",
            session_id=command.q_id,
        )

        return command

    # def read(self, question):
    #     response = self.db.read(question)
    #     return response

    @observe()
    def use(self, command: commands.UseTools) -> commands.UseTools:
        langfuse_context.update_current_trace(
            name="use",
            session_id=command.q_id,
        )

        response = self.tools.use(command.question)

        command.response = response
        return command

    # def retrieve(self, question):
    #     response = self.retrieve.retrieve(question)
    #     return response

    # def rerank(self, question):
    #     response = self.rerank.rerank(question)
    #     return response
