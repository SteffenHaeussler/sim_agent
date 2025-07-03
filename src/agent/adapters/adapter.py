from abc import ABC

from langfuse import get_client, observe

from src.agent import config
from src.agent.adapters import agent_tools, database, llm, rag
from src.agent.domain import commands, model


class AbstractAdapter(ABC):
    """
    AbstractAdapter is an abstract base class for all adapters.
    It defines the interface for external services.

    It defines the flow of commands from the model agent to the external service.

    Question -> Check -> UseTools -> Retrieve -> Rerank -> Enhance -> LLMResponse -> FinalCheck

    Methods:
        - add(agent: model.BaseAgent): Add an agent to the adapter.
        - collect_new_events(): Collect new events from the model agent.
        - answer(command: commands.Command) -> str: General entrypoint for a command.
    """

    def __init__(self):
        self.agent = None
        self.database = database.AbstractDatabase()
        self.llm = llm.AbstractLLM()
        self.tools = agent_tools.AbstractTools()
        self.rag = rag.AbstractModel()
        self.guardrails = llm.AbstractLLM()

    def add(self, agent: model.BaseAgent):
        """
        Add an agent to the adapter.

        Args:
            agent: model.BaseAgent: The agent to add.

        Returns:
            None
        """
        self.agent = agent

    def answer(self, command: commands.Command) -> str:
        """
        Answer a command.

        Args:
            command: commands.Command: The command to answer.

        Returns:
            str: The answer to the command.
        """
        raise NotImplementedError("Not implemented yet")

    def collect_new_events(self):
        """
        Collect new events from the model agent.

        Returns:
            An iterator of events.
        """
        while self.agent.events:
            event = self.agent.events.pop(0)
            yield event


class AgentAdapter(AbstractAdapter):
    """
    AgentAdapter is an adapter for the model agent.
    It defines the flow of commands from the model agent to the external service.

    Question -> Check -> UseTools -> Retrieve -> Rerank -> Enhance -> LLMResponse -> FinalCheck

    Methods:
        - answer(command: commands.Command) -> commands.Command: General entrypoint for a command.
        - check(command: commands.Check) -> commands.Check: Check the incoming question via guardrails.
        - evaluation(command: commands.FinalCheck) -> commands.FinalCheck: Evaluate the response via guardrails.
        - finalize(command: commands.LLMResponse) -> commands.LLMResponse: Finalize the response via LLM.
        - question(command: commands.Question) -> commands.Question: only for tracing.
        - use(command: commands.UseTools) -> commands.UseTools: Use the agent tools to process the question.
        - retrieve(command: commands.Retrieve) -> commands.Retrieve: Retrieve the most relevant documents from the knowledge base.
        - rerank(command: commands.Rerank) -> commands.Rerank: Rerank the documents from the knowledge base.
        - enhance(command: commands.Enhance) -> commands.Enhance: Enhance the question via LLM based on the reranked document.

    Adapters:
        - database: Database adapter.
        - guardrails: Performs checks via guardrails.
        - llm: Calls a LLM.
        - rag: RAG model to enhance questions and retrieve documents.
        - tools: Use the agent tools to process the question.
    """

    def __init__(self):
        super().__init__()

        self.database = database.BaseDatabaseAdapter(
            kwargs=config.get_database_config(),
        )

        self.guardrails = llm.LLM(
            kwargs=config.get_guardrails_config(),
        )
        self.llm = llm.LLM(
            kwargs=config.get_llm_config(),
        )
        self.rag = rag.BaseRAG(config.get_rag_config())
        self.tools = agent_tools.Tools(
            kwargs=config.get_tools_config(),
        )

    def answer(self, command: commands.Command) -> commands.Command:
        """
        Answer a command. Processes each request by the command type

        Args:
            command: commands.Command: The command to answer.

        Returns:
            commands.Command: The command to answer.
        """
        match command:
            case commands.Question():
                response = self.question(command)
            case commands.Check():
                response = self.check(command)
            case commands.Retrieve():
                response = self.retrieve(command)
            case commands.Rerank():
                response = self.rerank(command)
            case commands.Enhance():
                response = self.enhance(command)
            case commands.UseTools():
                response = self.use(command)
            case commands.LLMResponse():
                response = self.finalize(command)
            case commands.FinalCheck():
                response = self.evaluate(command)
            case _:
                raise NotImplementedError(
                    f"Not implemented in AgentAdapter: {type(command)}"
                )
        return response

    @observe()
    def check(self, command: commands.Check) -> commands.Check:
        """
        Check the incoming question via guardrails.

        Args:
            command: commands.Check: The command to check.

        Returns:
            commands.Check: The command to check.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
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
    def enhance(self, command: commands.Enhance):
        """
        Enhance the question via LLM based on the reranked document.

        Args:
            command: commands.Enhance: The command to enhance the question.

        Returns:
            commands.Enhance: The command to enhance the question.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name="enhance",
            session_id=command.q_id,
        )

        response = self.llm.use(command.question, commands.LLMResponseModel)

        command.response = response.response
        command.chain_of_thought = response.chain_of_thought

        return command

    @observe()
    def evaluate(self, command: commands.FinalCheck) -> commands.FinalCheck:
        """
        Evaluate the response via guardrails.

        Args:
            command: commands.FinalCheck: The command to evaluate.

        Returns:
            commands.FinalCheck: The command to evaluate.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
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
        """
        Finalize the response via LLM.

        Args:
            command: commands.LLMResponse: The command to finalize the response.

        Returns:
            commands.LLMResponse: The command to finalize the response.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name="finalize",
            session_id=command.q_id,
        )

        response = self.llm.use(command.question, commands.LLMResponseModel)

        command.response = response.response
        command.chain_of_thought = response.chain_of_thought

        return command

    @observe()
    def question(self, command: commands.Question) -> commands.Question:
        """
        Only for tracing.

        Args:
            command: commands.Question: The command to handle a question.

        Returns:
            commands.Question: The command to handle a question.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name="question",
            session_id=command.q_id,
        )

        return command

    @observe()
    def rerank(self, command: commands.Rerank):
        """
        Rerank the documents from the knowledge base.

        Args:
            command: commands.Rerank: The command to rerank the documents.

        Returns:
            commands.Rerank: The command to rerank the documents.
        """
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
    def retrieve(self, command: commands.Retrieve):
        """
        Retrieve the most relevant documents from the knowledge base.

        Args:
            command: commands.Retrieve: The command to retrieve the most relevant documents.

        Returns:
            commands.Retrieve: The command to retrieve the most relevant documents.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
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
    def use(self, command: commands.UseTools) -> commands.UseTools:
        """
        Use the agent tools to process the question.

        Args:
            command: commands.UseTools: The command to use the agent tools.

        Returns:
            commands.UseTools: The command to use the agent tools.
        """
        langfuse = get_client()

        langfuse.update_current_trace(
            name="use",
            session_id=command.q_id,
        )
        response, memory = self.tools.use(command.question)

        command.memory = memory

        if isinstance(response, dict) and "data" in response:
            command.data = response
            command.response = (
                "Response is a data extraction. FileStorage is not implemented yet."
            )

        elif isinstance(response, dict) and "plot" in response:
            command.response = "Response is a plot."
            command.data = response
        else:
            command.response = response

        return command
