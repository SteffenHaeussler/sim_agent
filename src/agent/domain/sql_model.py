from copy import deepcopy
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
    - SQLQuestion: The initial command to start the agent.
    - SQLCheck: Check the incoming question via guardrails.
    - SQLGrounding: Ground the incoming question.
    - SQLFilter: Filter the incoming question.
    - SQLJoinInference: Join the incoming question.
    - SQLAggregation: Aggregate the incoming question.
    - SQLConstruction: Construct the incoming question.
    - SQLValidation: Validate the incoming question.
    - SQLExecution: Execute the incoming question.

    Methods:
    - init_prompts: Initialize the prompts for the agent.
    - prepare_aggregation: Prepare the aggregation command.
    - prepare_construction: Prepare the construction command.
    - prepare_execution: Prepare the execution command.
    - prepare_filter: Prepare the filter command.
    - prepare_grounding: Prepare the grounding command.
    - prepare_guardrails_check: Prepare the guardrails check command.
    - prepare_join_inference: Prepare the join inference command.
    - prepare_validation: Prepare the validation command.
    - update: Update the state of the agent.
    - _update_state: Update the internal state of the agent.
    - create_prompt: Create the prompt for the command.
    """

    def __init__(self, question: commands.Question, kwargs: Dict = None):
        if not question or not question.question:
            raise ValueError("Question is required to enhance")

        self.kwargs = kwargs
        self.events = []
        self.is_answered = False
        self.evaluation = None
        self.q_id = question.q_id
        self.question = question.question
        self.previous_command = None
        self.response = None
        self.send_response = None
        self.query = None

        self.base_prompts = self.init_prompts()
        self.constructions = commands.SQLConstruction(
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
        if type(command) is commands.SQLCheck:
            prompt = self.base_prompts.get("check", None)
        elif type(command) is commands.SQLGrounding:
            prompt = self.base_prompts.get("ground", None)
        elif type(command) is commands.SQLFilter:
            prompt = self.base_prompts.get("filter", None)
        elif type(command) is commands.SQLJoinInference:
            prompt = self.base_prompts.get("join", None)
        elif type(command) is commands.SQLAggregation:
            prompt = self.base_prompts.get("aggregate", None)
        elif type(command) is commands.SQLConstruction:
            prompt = self.base_prompts.get("construct", None)
        elif type(command) is commands.SQLValidation:
            prompt = self.base_prompts.get("validate", None)
        else:
            raise ValueError("Invalid command type")

        if prompt is None:
            raise ValueError("Prompt not found")

        if type(command) is commands.SQLCheck:
            prompt = populate_template(
                prompt,
                {
                    "question": command.question,
                },
            )
        elif type(command) is commands.SQLGrounding:
            prompt = populate_template(
                prompt,
                {
                    "question": command.question,
                    "schema_info": command.schema_info,
                },
            )
        elif type(command) is commands.SQLFilter:
            prompt = populate_template(
                prompt,
                {
                    "question": command.question,
                    "column_mappings": command.column_mappings,
                    "table_mappings": command.table_mappings,
                },
            )
        elif type(command) is commands.SQLJoinInference:
            prompt = populate_template(
                prompt,
                {
                    "question": command.question,
                    "table_mappings": command.table_mappings,
                    "schema_info": command.schema_info,
                },
            )
        elif type(command) is commands.SQLAggregation:
            prompt = populate_template(
                prompt,
                {
                    "question": command.question,
                    "column_mappings": command.column_mappings,
                },
            )
        elif type(command) is commands.SQLConstruction:
            prompt = populate_template(
                prompt,
                {
                    "question": command.question,
                    "grounding_results": command.grounding_results,
                    "filter_results": command.filter_results,
                    "join_results": command.join_results,
                    "aggregation_results": command.aggregation_results,
                },
            )
        elif type(command) is commands.SQLValidation:
            prompt = populate_template(
                prompt,
                {
                    "question": command.question,
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
            with open(self.kwargs["sql_prompt_path"], "r") as file:
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
        self.constructions.joins = deepcopy(command.joins)

        new_command = commands.SQLAggregation(
            question=command.question,
            q_id=command.q_id,
        )

        new_command.question = self.create_prompt(new_command)

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
        self.constructions.aggregations = deepcopy(command.aggregations)
        self.constructions.group_by_columns = deepcopy(command.group_by_columns)
        self.constructions.is_aggregation_query = deepcopy(command.is_aggregation_query)

        new_command = deepcopy(self.constructions)

        new_command.question = self.create_prompt(new_command)

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
        """
        Prepares the filter for the question.

        Args:
            command: commands.SQLGrounding: The command to change the question.

        Returns:
            new_command: commands.SQLFilter: The new command.
        """

        self.constructions.column_mappings = deepcopy(command.column_mappings)
        self.constructions.table_mappings = deepcopy(command.table_mappings)

        new_command = commands.SQLFilter(
            question=command.question,
            q_id=command.q_id,
        )

        new_command.question = self.create_prompt(new_command)

        return new_command

    def prepare_grounding(self, command: commands.SQLCheck) -> commands.SQLGrounding:
        """
        Prepares the guardrails check for the question.

        Args:
            command: commands.Question: The command to change the question.

        Returns:
            new_command: commands.Check: The new command.
        """

        if command.approved:
            command.schema_info = deepcopy(self.constructions.schema_info)

            new_command = commands.SQLGrounding(
                question=command.question,
                q_id=command.q_id,
            )

            new_command.question = self.create_prompt(new_command)

        else:
            self.is_answered = True
            new_command = events.RejectedRequest(
                question=self.question,
                response=command.response,
                q_id=command.q_id,
            )
            self.send_response = new_command
            self.response = new_command

        return new_command

    def prepare_guardrails_check(self, command: commands.Question) -> commands.Check:
        """
        Prepares the guardrails check for the question.

        Args:
            command: commands.Question: The command to change the question.

        Returns:
            new_command: commands.Check: The new command.
        """
        # save the schema info
        self.constructions.schema_info = deepcopy(command.schema_info)

        # create the new command
        new_command = commands.SQLCheck(
            question=command.question,
            q_id=command.q_id,
        )

        new_command.question = self.create_prompt(new_command)

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
        # save the filter results
        self.constructions.filter_results = deepcopy(command.conditions)

        new_command = commands.SQLJoinInference(
            question=command.question,
            q_id=command.q_id,
            table_mappings=deepcopy(self.constructions.table_mappings),
            schema_info=deepcopy(self.constructions.schema_info),
        )

        new_command.question = self.create_prompt(new_command)

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
        new_command = commands.SQLValidation(
            question=command.question,
            q_id=command.q_id,
        )

        new_command.question = self.create_prompt(new_command)

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
