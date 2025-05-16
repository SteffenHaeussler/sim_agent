from abc import ABC
from datetime import datetime
from typing import Dict

import yaml
from smolagents import CodeAgent, LiteLLMModel, PromptTemplates

import src.agent.adapters.tools as tools


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

    def use(self, question):
        response = self.agent.run(question)
        return response
