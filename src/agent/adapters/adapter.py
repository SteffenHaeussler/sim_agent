from abc import ABC

from langfuse.decorators import langfuse_context, observe

from src.agent import config
from src.agent.adapters import agent_tools, db, llm, rag
from src.agent.domain import commands, model


class AbstractAdapter(ABC):
    def __init__(self):
        self.seen = set()
        self.db = db.AbstractDB()
        self.llm = llm.AbstractLLM()
        self.tools = agent_tools.AbstractTools()
        self.rag = rag.AbstractModel()
        self.guardrails = llm.AbstractLLM()

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
        self.rag = rag.BaseRAG(config.get_rag_config())
        self.guardrails = llm.LLM(
            kwargs=config.get_guardrails_config(),
        )

    def answer(self, command: commands.Command) -> commands.Command:
        if type(command) is commands.Question:
            response = self.question(command)
        elif type(command) is commands.Check:
            response = self.check(command)
        elif type(command) is commands.UseTools:
            response = self.use(command)
        elif type(command) is commands.Retrieve:
            response = self.retrieve(command)
        elif type(command) is commands.Rerank:
            response = self.rerank(command)
        elif type(command) is commands.Enhance:
            response = self.enhance(command)
        elif type(command) is commands.LLMResponse:
            response = self.finalize(command)
        elif type(command) is commands.FinalCheck:
            response = self.evaluation(command)
        else:
            raise NotImplementedError(
                f"Not implemented in AgentAdapter: {type(command)}"
            )
        return response

    @observe()
    def check(self, command: commands.Check) -> commands.Check:
        langfuse_context.update_current_trace(
            name="check",
            session_id=command.q_id,
        )
        response = self.guardrails.use(
            command.question, commands.GuardrailPreCheckModel
        )

        command.response = response.response
        command.chain_of_thought = response.chain_of_thought
        command.approved = response.approved

        return command

    @observe()
    def evaluation(self, command: commands.FinalCheck) -> commands.FinalCheck:
        langfuse_context.update_current_trace(
            name="evaluation",
            session_id=command.q_id,
        )
        response = self.guardrails.use(
            command.question, commands.GuardrailPostCheckModel
        )

        command.chain_of_thought = response.chain_of_thought
        command.approved = response.approved
        command.summary = response.summary
        command.issues = response.issues
        command.plausibility = response.plausibility
        command.factual_consistency = response.factual_consistency
        command.clarity = response.clarity
        command.completeness = response.completeness

        return command

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

    @observe()
    def use(self, command: commands.UseTools) -> commands.UseTools:
        langfuse_context.update_current_trace(
            name="use",
            session_id=command.q_id,
        )

        response, memory = self.tools.use(command.question)

        command.response = response
        command.memory = memory

        return command

    @observe()
    def retrieve(self, command: commands.Retrieve):
        langfuse_context.update_current_trace(
            name="retrieve",
            session_id=command.q_id,
        )
        candidates = []
        response = self.rag.embed(command.question)

        if response is not None:
            response = self.rag.retrieve(response["embedding"])

            for candidate in response["results"]:
                candidates.append(commands.KBResponse(**candidate))

        command.candidates = candidates
        return command

    @observe()
    def rerank(self, command: commands.Rerank):
        langfuse_context.update_current_trace(
            name="rerank",
            session_id=command.q_id,
        )
        candidates = []

        for candidate in command.candidates:
            response = self.rag.rerank(command.question, candidate.description)

            temp = candidate.model_dump()
            temp.pop("score", None)
            candidates.append(commands.RerankResponse(**response, **temp))

        candidates = sorted(candidates, key=lambda x: -x.score)

        command.candidates = candidates[: self.rag.n_ranking_candidates]
        return command

    @observe()
    def enhance(self, command: commands.Enhance):
        langfuse_context.update_current_trace(
            name="enhance",
            session_id=command.q_id,
        )

        response = self.llm.use(command.question, commands.LLMResponseModel)

        command.response = response.response
        command.chain_of_thought = response.chain_of_thought

        return command
