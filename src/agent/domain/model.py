from typing import Dict, Optional, Union

import yaml

from src.agent.domain import commands, events, rag
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
        self.cls_rag = rag.RAGLogic(self)
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

    def check_question(self, command: commands.Question) -> commands.UseTools:
        new_command = commands.Retrieve(
            question=command.question,
            q_id=command.q_id,
        )

        return new_command

    def change_question(self, command: commands.Enhance) -> commands.UseTools:
        self.enhancement = command.enhancement
        breakpoint()

        new_command = commands.UseTools(
            question=command.enhancement,
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

    def create_prompt(self, command: Union[commands.UseTools, commands.Enhance]) -> str:
        prompt_path = self.kwargs["prompt_path"]

        with open(prompt_path, "r") as file:
            base_prompts = yaml.safe_load(file)

        if type(command) is commands.UseTools:
            prompt = base_prompts.get("finalize", None)

        elif type(command) is commands.Enhance:
            prompt = base_prompts.get("enhance", None)
        else:
            raise ValueError("Invalid command type")

        if prompt is None:
            raise ValueError("final_answer prompt not found")
        breakpoint()
        if type(command) is commands.UseTools:
            prompt = populate_template(
                prompt,
                {
                    "question": command.question,
                    "response": command.response,
                },
            )
        elif type(command) is commands.Enhance:
            prompt = populate_template(
                prompt,
                {
                    "question": command.question,
                },
            )
        else:
            raise ValueError("Invalid command type")

        return prompt

    def change_rerank(self, command: commands.Rerank) -> commands.Enhance:
        # if not command.question:
        #     raise ValueError("Question is required to enhance")

        # rerank = self.cls_rag.rerank(question)
        # self.cls_rag.rerank = rerank
        breakpoint()

        new_command = commands.Enhance(
            question=command.question,
            q_id=command.q_id,
        )

        return new_command

    def change_retrieve(self, command: commands.Retrieve) -> commands.Rerank:
        # if not command.question:
        #     raise ValueError("Question is required to enhance")

        # retrieve = self.cls_rag.retrieve(question)
        # # self.cls_rag.retrieve = retrieve
        new_command = commands.Rerank(
            question=command.question,
            q_id=command.q_id,
            candidates=command.candidates,
        )

        return new_command

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
            new_command = self.check_question(command)
        elif type(command) is commands.Retrieve:
            new_command = self.change_retrieve(command)
        elif type(command) is commands.Rerank:
            new_command = self.change_rerank(command)
        elif type(command) is commands.Enhance:
            new_command = self.change_question(command)
        elif type(command) is commands.UseTools:
            new_command = self.change_use_tools(command)
        elif type(command) is commands.LLMResponse:
            new_command = self.change_llm_response(command)
        else:
            raise NotImplementedError(
                f"Not implemented yet for BaseAgent: {type(command)}"
            )

        return new_command
