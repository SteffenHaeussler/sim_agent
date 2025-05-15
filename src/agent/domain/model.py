from typing import Dict, Optional

import yaml

from src.agent.domain import commands, events
from src.agent.utils import populate_template


class BaseAgent:
    def __init__(self, question: commands.Question, kwargs: Dict = None):
        if not question or not question.question:
            raise ValueError("Question is required to enhance")

        self.question = question.question
        self.q_id = question.q_id
        self.enhancement = None
        self.tool_answer = None
        self.response = None

        self.is_answered = False
        self.previous_command = None

        self.kwargs = kwargs

        # self.cls_guard = guardrails.Guardrails(self)
        # self.cls_rag = rag.RAG(self)
        self.events = []

    # def check(self, question):
    #     if not question:
    #         raise ValueError("Question is required to enhance")

    #     check = self.cls_guard.check(question)
    #     self.cls_guard.check = check

    #     return None

    def change_llm_response(self, command: commands.LLMResponse) -> None:
        if self.tool_answer is None:
            raise ValueError("Tool answer is required for LLM response")

        self.is_answered = True

        response = events.Response(
            question=self.question,
            response=command.response,
            q_id=self.q_id,
        )

        self.response = response
        return None

    def change_question(self, command: commands.Question) -> commands.UseTools:
        # if self.enhancement is None:
        #     raise ValueError("Enhancement is required to use tools

        new_command = commands.UseTools(
            question=command.question,
            q_id=command.q_id,
        )

        return new_command

    def change_use_tools(self, command: commands.UseTools) -> commands.LLMResponse:
        self.tool_answer = command

        prompt = self.create_final_answer_prompt(command)
        new_command = commands.LLMResponse(
            question=prompt,
            q_id=command.q_id,
        )

        return new_command

    def create_final_answer_prompt(self, command: commands.UseTools) -> str:
        prompt_path = self.kwargs["prompt_path"]

        with open(prompt_path, "r") as file:
            base_prompts = yaml.safe_load(file)

        prompt = base_prompts.get("final_answer", {}).get("final_answer", None)

        if prompt is None:
            raise ValueError("final_answer prompt not found")

        prompt = populate_template(
            prompt,
            {
                "question": command.question,
                "response": command.response,
            },
        )

        return prompt

    # def rerank(self, question):
    #     if not question:
    #         raise ValueError("Question is required to enhance")

    #     rerank = self.cls_rag.rerank(question)
    #     self.cls_rag.rerank = rerank

    #     return rerank

    # def retrieve(self, question):
    #     if not question:
    #         raise ValueError("Question is required to enhance")

    #     retrieve = self.cls_rag.retrieve(question)
    #     self.cls_rag.retrieve = retrieve

    #     return retrieve

    def _update_state(self, response: commands.Command) -> None:
        if self.previous_command is type(response):
            self.is_answered = True
            self.response = events.FailedRequest(
                question=self.question,
                exception=response,
            )

        else:
            self.previous_command = type(response)

        return None

    def update(self, command: commands.Command) -> Optional[commands.Command]:
        self._update_state(command)

        if self.is_answered:
            return None

        # if type(command) is commands.Check:
        #     new_command = self.check(response)
        if type(command) is commands.Question:
            new_command = self.change_question(command)
        # elif type(command) is commands.Retrieve:
        #     new_command = self.retrieve(command)
        # elif type(command) is commands.Rerank:
        #     new_command = self.rerank(command)
        elif type(command) is commands.UseTools:
            new_command = self.change_use_tools(command)
        elif type(command) is commands.LLMResponse:
            new_command = self.change_llm_response(command)
        else:
            raise NotImplementedError(
                f"Not implemented yet for BaseAgent: {type(command)}"
            )

        return new_command
