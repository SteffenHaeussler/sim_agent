import os
from abc import ABC
from datetime import datetime
from typing import Dict, List, Tuple

import yaml
from langfuse.decorators import langfuse_context, observe
from opentelemetry import trace
from smolagents import (
    ActionStep,
    CodeAgent,
    LiteLLMModel,
    PlanningStep,
    PromptTemplates,
    TaskStep,
)

import src.agent.adapters.tools as tools
from src.agent.observability.context import ctx_query_id


class AbstractTools(ABC):
    """
    AbstractTools is an abstract base class for all tools.
    It defines the interface for all tools.

    Methods:
        - use(self): Use the tools.
    """

    def __init__(self):
        pass

    def use(self):
        pass


class Tools(AbstractTools):
    """
    Tools is a class that uses the tools.

    Methods:
        - use(self): Use the tools.
        - init_model(self): Initialize the llm model.
        - init_prompt_templates(self): Initialize the prompt templates for tool calls.
        - init_agent(self): Initialize the agent.
        - get_memory(self): Get the agent's memory.
    """

    def __init__(
        self,
        kwargs: Dict,
    ):
        """
        Initialize the tools.

        Args:
            kwargs: Dict: The kwargs.

        Returns:
            None
        """
        self.kwargs = kwargs
        self.llm_model_id = kwargs["llm_model_id"]
        self.max_steps = int(kwargs["max_steps"])

        # in this order
        self.model = self.init_model(self.kwargs)
        self.prompt_templates = self.init_prompt_templates(self.kwargs)
        self.agent = self.init_agent(self.kwargs)

    def get_memory(self) -> List[str]:
        """
        Get the agent's memory.

        Returns:
            memory: List[str]: The agent's memory for each step.
        """
        memory = []

        for step in self.agent.memory.steps:
            if type(step) is TaskStep:
                memory.append(step.task)
            elif type(step) is ActionStep:
                if step.model_output is not None:
                    memory.append(step.model_output)
            elif type(step) is PlanningStep:
                memory.append(step.plan)

        return memory

    def init_agent(self, kwargs: Dict) -> CodeAgent:
        """
        Initialize the agent.

        Args:
            kwargs: Dict: The kwargs.

        Returns:
            agent: CodeAgent: The agent.
        """
        agent = CodeAgent(
            tools=[
                tools.CompareData(**kwargs),
                tools.ConvertIdToName(**kwargs),
                tools.ConvertNameToId(**kwargs),
                tools.GetData(**kwargs),
                tools.GetInformation(**kwargs),
                tools.GetNeighbors(**kwargs),
                tools.PlotData(**kwargs),
                tools.FinalAnswerTool(**kwargs),
            ],
            model=self.model,
            stream_outputs=True,
            additional_authorized_imports=["pandas", "numpy"],
            prompt_templates=self.prompt_templates,
            max_steps=self.max_steps,
        )
        return agent

    def init_model(self, kwargs: Dict) -> LiteLLMModel:
        """
        Initialize the llm model.

        Args:
            kwargs: Dict: The kwargs.

        Returns:
            model: LiteLLMModel: The llm model.
        """
        api_base = kwargs["llm_api_base"]
        model = LiteLLMModel(model_id=self.llm_model_id, api_base=api_base)

        return model

    def init_prompt_templates(self, kwargs: Dict) -> PromptTemplates:
        """
        Initialize the prompt templates for tool calls.

        Args:
            kwargs: Dict: The kwargs.

        Returns:
            prompt_templates: PromptTemplates: The prompt templates.
        """
        prompt_path = kwargs["prompt_path"]

        with open(prompt_path, "r") as file:
            base_prompts = yaml.safe_load(file)

        base_prompts["system_prompt"] = base_prompts["system_prompt"].replace(
            "{{current_date}}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        prompt_templates = PromptTemplates(**base_prompts)

        return prompt_templates

    def use(self, question: str) -> Tuple[str, List[str]]:
        """
        Use the agent's tools.

        Args:
            question: str: The question to use the agent's tools.

        Returns:
            response: str: The response from the agent's tools.
            memory: List[str]: The agent's memory for each step.
        """
        if os.getenv("TELEMETRY_ENABLED", None) == "true":
            response = self._use_with_telemetry(question)
        else:
            response = self._use(question)

        memory = self.get_memory()
        return response, memory

    def _use(self, question: str) -> str:
        """
        Use the agent's tools without telemetry.

        Args:
            question: str: The question to use the agent's tools.

        Returns:
            response: str: The response from the agent's tools.
        """
        response = self.agent.run(question)
        return response

    @observe()
    def _use_with_telemetry(self, question: str) -> str:
        """
        Use the agent's tools with telemetry.

        Args:
            question: str: The question to use the agent's tools.

        Returns:
            response: str: The response from the agent's tools.
        """
        langfuse_context.update_current_observation(
            name="use_tools",
            session_id=ctx_query_id.get(),
        )

        tracer = trace.get_tracer("smolagents")

        with tracer.start_as_current_span("Smolagent-Trace") as span:
            span.set_attribute("session.id", ctx_query_id.get())
            span.set_attribute("langfuse.session.id", ctx_query_id.get())
            span.set_attribute("langfuse.session_id", ctx_query_id.get())
            span.set_attribute("session_id", ctx_query_id.get())

            response = self.agent.run(question)

        return response
