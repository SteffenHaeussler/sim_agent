import json
from copy import deepcopy
from typing import Dict, List, Optional

import yaml
from src.agent.adapters import tools
from src.agent.domain import commands, events
from src.agent.utils import populate_template

tool_names = tools.__all__


class ScenarioBaseAgent:
    """
    BaseAgent is the model logic for the agent. It's uses a state machine to process and propagate different commands.
    The update method is the main method that decides the next command based on the current state and the incoming command.

    The events list is used to store outgoing events and will be picked up for notifications.
    Is_answered is used to check if the agent has answered the question and stops the state machine.
    Previous_command is used to check if the command is a duplicate and stops the state machine.

    Following commands are supported:
    - Scenario: The initial command to start the agent.
    - ScenarioCheck: Check the incoming question via guardrails.
    - ScenarioLLMResponse: Finalize the response via LLM.
    - ScenarioFinalCheck: Validate the question to schema elements.

    Methods:
    - init_prompts: Initialize the prompts for the agent.
    - prepare_guardrails_check: Prepare the guardrails check command.
    - prepare_llm_response: Prepare the llm response command.
    - prepare_validation: Prepare the validation command.
    - update: Update the state of the agent.
    - _update_state: Update the internal state of the agent.
    """

    def __init__(self, question: commands.Scenario, kwargs: Dict = None):
        if not question or not question.question:
            raise ValueError("Question is required to start the agent")

        self.kwargs = kwargs
        self.events = []
        self.is_answered = False
        self.evaluation = None
        self.q_id = question.q_id
        self.question = question.question
        self.previous_command = None
        self.response = None
        self.send_response = None
        self.sql_query = None

        self.base_prompts = self.init_prompts()
        self.scenario = commands.Scenario(
            question=self.question,
            q_id=self.q_id,
        )

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
        if type(command) is commands.Check:
            prompt = self.base_prompts.get("check", None)
        elif type(command) is commands.ScenarioLLMResponse:
            prompt = self.base_prompts.get("response", None)
        elif type(command) is commands.ScenarioFinalCheck:
            prompt = self.base_prompts.get("final_check", None)
        else:
            raise ValueError("Invalid command type")

        if prompt is None:
            raise ValueError("Prompt not found")

        if type(command) is commands.Check:
            prompt = populate_template(
                prompt,
                {
                    "question": command.question,
                },
            )
        elif type(command) is commands.ScenarioLLMResponse:
            prompt = populate_template(
                prompt,
                {
                    "question": command.question,
                    "tables": command.tables,
                    # "tools": "\n".join(command.tools),
                },
            )
        elif type(command) is commands.ScenarioFinalCheck:
            prompt = populate_template(
                prompt,
                {
                    "question": command.question,
                    "candidates": command.candidates,
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
            with open(self.kwargs["scenario_prompt_path"], "r") as file:
                base_prompts = yaml.safe_load(file)
        except FileNotFoundError:
            raise ValueError("Prompt path not found")

        return base_prompts

    def prepare_guardrails_check(self, command: commands.Question) -> commands.Check:
        """
        Prepares the guardrails check for the question.

        Args:
            command: commands.Question: The command to change the question.

        Returns:
            new_command: commands.Check: The new command.
        """
        # save the schema info
        self.scenario.schema_info = deepcopy(command.schema_info)
        # create the new command
        new_command = commands.Check(
            question=command.question,
            q_id=command.q_id,
        )

        new_command.question = self.create_prompt(new_command)

        return new_command

    def prepare_finalization(
        self, command: commands.ScenarioLLMResponse
    ) -> commands.ScenarioLLMResponse:
        """
        Prepares the finalization for the question.

        Args:
            command: commands.ScenarioLLMResponse: The command to change the question.

        Returns:
            new_command: commands.ScenarioLLMResponse: The new command.
        """
        tools = self.get_tool_info()

        new_command = commands.ScenarioLLMResponse(
            question=command.question,
            q_id=command.q_id,
            tables=deepcopy(self.scenario.schema_info.tables),
            tools=tools,
        )

        new_command.question = self.create_prompt(new_command)

        return new_command

    def prepare_response(
        self, command: commands.ScenarioLLMResponse
    ) -> commands.ScenarioLLMResponse:
        """
        Prepares the final guardrailscheck and agent response after the final LLM call.

        Args:
            command: commands.LLMResponse: The command to change the LLM response.

        Returns:
            new_command: commands.FinalCheck: The new command.
        """
        if command.candidates is None:
            response = "No candidates found"
        else:
            recommendations = [
                {
                    "sub_id": f"sub-{i + 1}",
                    "question": candidate.question,
                    "endpoint": candidate.endpoint,
                }
                for i, candidate in enumerate(command.candidates)
            ]

            response = json.dumps(recommendations)

        response = events.Response(
            question=self.question,
            response=response,
            q_id=self.q_id,
        )

        # duplication as a fix - i want to keep self.response for testing adn validation
        self.send_response = response
        self.response = response

        new_command = commands.ScenarioFinalCheck(
            question=command.question,
            q_id=command.q_id,
            candidates=command.candidates,
        )

        new_command.question = self.create_prompt(new_command)

        return new_command

    def prepare_validation(self, command: commands.SQLValidation) -> None:
        """
        Prepares the guardrails check for the question.

        Args:
            command: commands.Question: The command to change the question.

        Returns:
            new_command: commands.Check: The new command.
        """
        self.is_answered = True

        summary = command.summary

        self.evaluation = events.Evaluation(
            response=self.response.response,
            question=self.question,
            q_id=self.q_id,
            approved=command.approved,
            summary=summary,
            issues=command.issues,
        )

        return None

    def get_tool_info(self) -> List[str]:
        """
        Get the tool info for the agent.

        Returns:
            tool_info: List[str]: The tool info for the agent.
        """
        tool_info = []
        for name in tools.__all__:
            tool = getattr(tools, name)
            tool_info.append(f"{tool.name}: {tool.description}")

        return tool_info

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
            case commands.Scenario():
                new_command = self.prepare_guardrails_check(command)
            case commands.Check():
                new_command = self.prepare_finalization(command)
            case commands.ScenarioLLMResponse():
                new_command = self.prepare_response(command)
            case commands.ScenarioFinalCheck():
                new_command = self.prepare_validation(command)
            case _:
                raise NotImplementedError(
                    f"Not implemented yet for BaseAgent: {type(command)}"
                )

        return new_command
