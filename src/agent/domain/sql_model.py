import json
from typing import Dict, List, Optional

import yaml

from src.agent.domain import commands, events
from src.agent.utils import populate_template


class SQLBaseAgent:
    """
    BaseAgent is the model logic for the agent. It's uses a state machine to process and propagate different commands.
    The update method is the main method that decides the next command based on the current state and the incoming command.

    The events list is used to store outgoing events and will be picked up for notifications.
    Is_answered is used to check if the agent has answered the question and stops the state machine.
    Previous_command is used to check if the command is a duplicate and stops the state machine.

    Following commands are supported:
    - Question: The initial command to start the agent.
    - Check: Check the incoming question via guardrails.
    - Retrieve: Retrieve the most relevant documents from the knowledge base.
    - Rerank: Rerank the documents from the knowledge base.
    - Enhance: Enhance the question via LLM based on the reranked document.
    - UseTools: Use the agent tools to process the question.
    - LLMResponse: Use the LLM to process the question.
    - FinalCheck: Check the final answer via guardrails.

    Methods:
    - init_prompts: Initialize the prompts for the agent.
    - change_llm_response: Change the LLM response.
    - final_check: Check the final answer.
    - check_question: Check the question.
    - change_check: Change the check.
    - change_retrieve: Change the retrieve.
    - change_rerank: Change the rerank.
    - change_question: Change the question.
    - change_use_tools: Change the use tools.
    - create_prompt: Create the prompt for the command.
    - update: Update the state of the agent.
    """

    def __init__(self, question: commands.Question, kwargs: Dict = None):
        if not question or not question.question:
            raise ValueError("Question is required to enhance")

        self.kwargs = kwargs
        self.events = []
        self.is_answered = False

        self.agent_memory = None
        self.enhancement = None
        self.evaluation = None
        self.q_id = question.q_id
        self.question = question.question
        self.previous_command = None
        self.response = None
        self.send_response = None
        self.tool_answer = None

        self.base_prompts = self.init_prompts()

    def create_prompt(
        self,
        command: commands.Command,
        memory: List[str] = None,
    ) -> str:
        """
        Gets and preprocesses the prompt by the incoming command.

        Args:
            command: commands.Command: The command to create the prompt for.

        Returns:
            prompt: str: The prepared prompt for the command.
        """
        if type(command) is commands.UseTools:
            prompt = self.base_prompts.get("finalize", None)
        elif type(command) is commands.Rerank:
            prompt = self.base_prompts.get("enhance", None)
        elif type(command) is commands.Question:
            prompt = self.base_prompts.get("guardrails", {}).get("pre_check", None)
        elif type(command) is commands.LLMResponse:
            prompt = self.base_prompts.get("guardrails", {}).get("post_check", None)
        else:
            raise ValueError("Invalid command type")

        if prompt is None:
            raise ValueError("Prompt not found")

        if type(command) is commands.UseTools:
            prompt = populate_template(
                prompt,
                {
                    "question": command.question,
                    "response": command.response,
                },
            )
        elif type(command) is commands.Rerank:
            candidates = [i.model_dump() for i in command.candidates]
            candidates = json.dumps(candidates)

            prompt = populate_template(
                prompt,
                {
                    "question": command.question,
                    "information": candidates,
                },
            )
        elif type(command) is commands.Question:
            prompt = populate_template(
                prompt,
                {
                    "question": command.question,
                },
            )
        elif type(command) is commands.LLMResponse:
            prompt = populate_template(
                prompt,
                {
                    "question": command.question,
                    "response": command.response,
                    "memory": "\n".join(memory),
                },
            )
        else:
            raise ValueError("Invalid command type")

        return prompt

    def init_prompts(self) -> Dict:
        """
        Initialize the prompts for the agent.

        Returns:
            base_prompts: Dict: The base prompts for the agent.
        """
        try:
            with open(self.kwargs["prompt_path"], "r") as file:
                base_prompts = yaml.safe_load(file)
        except FileNotFoundError:
            raise ValueError("Prompt path not found")

        return base_prompts

    def prepare_aggregation(
        self, command: commands.SQLJoinInference
    ) -> commands.SQLAggregation:
        """
        Prepares the guardrails check for the question.

        Args:
            command: commands.Question: The command to change the question.

        Returns:
            new_command: commands.Check: The new command.
        """
        prompt = self.create_prompt(command)

        new_command = commands.SQLAggregation(
            question=prompt,
            q_id=command.q_id,
        )

        return new_command

    def prepare_construction(
        self, command: commands.SQLAggregation
    ) -> commands.SQLConstruction:
        """
        Prepares the guardrails check for the question.

        Args:
            command: commands.Question: The command to change the question.

        Returns:
            new_command: commands.Check: The new command.
        """
        prompt = self.create_prompt(command)

        new_command = commands.SQLValidation(
            question=prompt,
            q_id=command.q_id,
        )

        return new_command

    def prepare_execution(self, command: commands.SQLValidation) -> None:
        """
        Prepares the evaluation event to be sent.

        Args:
            command: commands.SQLValidation: The command to final check the answer.

        Returns:
            None
        """
        self.is_answered = True

        self.query = events.Query(
            response=self.response.response,
            question=self.question,
            q_id=self.q_id,
            approved=command.approved,
            summary=command.summary,
        )

        return None

    def prepare_filter(self, command: commands.SQLGrounding) -> commands.SQLFilter:
        prompt = self.create_prompt(command)

        new_command = commands.SQLFilter(
            question=prompt,
            q_id=command.q_id,
        )

        return new_command

    def prepare_grounding(self, command: commands.SQLCheck) -> commands.SQLGrounding:
        """
        Prepares the guardrails check for the question.

        Args:
            command: commands.Question: The command to change the question.

        Returns:
            new_command: commands.Check: The new command.
        """
        prompt = self.create_prompt(command)

        new_command = commands.SQLGrounding(
            question=prompt,
            q_id=command.q_id,
        )

        return new_command

    def prepare_guardrails_check(self, command: commands.Question) -> commands.Check:
        """
        Prepares the guardrails check for the question.

        Args:
            command: commands.Question: The command to change the question.

        Returns:
            new_command: commands.Check: The new command.
        """
        prompt = self.create_prompt(command)

        new_command = commands.Check(
            question=prompt,
            q_id=command.q_id,
        )

        return new_command

    def prepare_join_inference(
        self, command: commands.SQLFilter
    ) -> commands.SQLJoinInference:
        """
        Prepares the guardrails check for the question.

        Args:
            command: commands.Question: The command to change the question.

        Returns:
            new_command: commands.Check: The new command.
        """
        prompt = self.create_prompt(command)

        new_command = commands.SQLJoinInference(
            question=prompt,
            q_id=command.q_id,
        )

        return new_command

    def prepare_validation(
        self, command: commands.SQLConstruction
    ) -> commands.SQLValidation:
        """
        Prepares the guardrails check for the question.

        Args:
            command: commands.Question: The command to change the question.

        Returns:
            new_command: commands.Check: The new command.
        """
        prompt = self.create_prompt(command)

        new_command = commands.SQLValidation(
            question=prompt,
            q_id=command.q_id,
        )

        return new_command

    def _update_state(self, response: commands.Command) -> None:
        """
        Update the internal state of the agent and check for repetition.

        Args:
            response: commands.Command: The command to update the state.

        Returns:
            None
        """
        if self.previous_command is type(response):
            self.is_answered = True
            self.response = events.FailedRequest(
                question=self.question,
                exception="Internal error: Duplicate command",
                q_id=self.q_id,
            )

        else:
            self.previous_command = type(response)

        return None

    def update(self, command: commands.Command) -> Optional[commands.Command]:
        """
        Update the state of the agent.

        Args:
            command: commands.Command: The command to update the state.

        Returns:
            Optional[commands.Command]: The next command.
        """
        self._update_state(command)

        if self.is_answered:
            return None

        # following the command chain
        match command:
            case commands.SQLQuestion():
                new_command = self.prepare_guardrails_check(command)
            case commands.SQLCheck():
                new_command = self.prepare_grounding(command)
            case commands.SQLGrounding():
                new_command = self.prepare_filter(command)
            case commands.SQLFilter():
                new_command = self.prepare_join_inference(command)
            case commands.SQLJoinInference():
                new_command = self.prepare_aggregation(command)
            case commands.SQLAggregation():
                new_command = self.prepare_construction(command)
            case commands.SQLConstruction():
                new_command = self.prepare_validation(command)
            case commands.SQLValidation():
                new_command = self.prepare_execution(command)
            case _:
                raise NotImplementedError(
                    f"Not implemented yet for BaseAgent: {type(command)}"
                )

        return new_command
