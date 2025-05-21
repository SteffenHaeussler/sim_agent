import os
from abc import ABC
from datetime import datetime
from typing import Dict, List

import yaml
from langfuse.decorators import langfuse_context, observe
from opentelemetry import trace
from smolagents import CodeAgent, LiteLLMModel, PromptTemplates
from smolagents.memory import ActionStep, PlanningStep, TaskStep

import src.agent.adapters.tools as tools
from src.agent.observability.context import ctx_query_id


class AbstractTools(ABC):
    def __init__(self):
        pass

    def use(self):
        pass


class Tools(AbstractTools):
    def __init__(
        self,
        kwargs: Dict,
    ):
        self.kwargs = kwargs
        self.max_steps = int(kwargs["max_steps"])
        self.llm_model_id = kwargs["llm_model_id"]

        self.model = self.init_model(self.kwargs)
        self.prompt_templates = self.init_prompt_templates(self.kwargs)

        self.agent = self.init_agent(self.kwargs)

    def init_model(self, kwargs: Dict):
        api_base = kwargs["llm_api_base"]

        model = LiteLLMModel(model_id=self.llm_model_id, api_base=api_base)

        return model

    def init_prompt_templates(self, kwargs: Dict):
        prompt_path = kwargs["prompt_path"]

        with open(prompt_path, "r") as file:
            base_prompts = yaml.safe_load(file)

        base_prompts["system_prompt"] = base_prompts["system_prompt"].replace(
            "{{current_date}}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        prompt_templates = PromptTemplates(**base_prompts)

        return prompt_templates

    def init_agent(self, kwargs: Dict):
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

    def get_memory(self) -> List[str]:
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

    def use(self, question):
        if os.environ["TELEMETRY_ENABLED"] == "true":
            response = self._use_with_telemetry(question)
        else:
            response = self._use(question)

        memory = self.get_memory()
        return response, memory

    def _use(self, question):
        response = self.agent.run(question)
        return response

    @observe()
    def _use_with_telemetry(self, question):
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
