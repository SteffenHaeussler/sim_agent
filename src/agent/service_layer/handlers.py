from typing import Union

from langfuse.decorators import langfuse_context, observe
from loguru import logger

from src.agent import config
from src.agent.adapters.adapter import AbstractAdapter
from src.agent.adapters.notifications import AbstractNotifications
from src.agent.domain import commands, events, model


class InvalidQuestion(Exception):
    pass


@observe()
def answer(command: commands.Question, adapter: AbstractAdapter) -> None:
    """
    Handles incoming questions.

    Args:
        command: commands.Question: The question to answer.
        adapter: AbstractAdapter: The adapter to use.

    Returns:
        None
    """
    langfuse_context.update_current_trace(
        name="answer handler",
        session_id=command.q_id,
    )

    if not command or not command.question:
        raise InvalidQuestion("No question asked")

    agent = model.BaseAgent(command, config.get_agent_config())
    adapter.add(agent)

    # adapter for execution and agent for internal logic
    while not agent.is_answered and command is not None:
        logger.info(f"Calling Adapter with command: {type(command)}")
        updated_command = adapter.answer(command)
        command = agent.update(updated_command)

    event = agent.response
    agent.events.append(event)

    if agent.evaluation:
        event = agent.evaluation
        agent.events.append(event)

    return None


@observe()
def send_response(
    event: Union[events.Response, events.Evaluation],
    notifications: AbstractNotifications,
) -> None:
    """
    Sends the response to the notifications.

    Args:
        event: Union[events.Response, events.Evaluation]: The event to send.
        notifications: AbstractNotifications: The notifications to use.

    Returns:
        None
    """
    langfuse_context.update_current_trace(
        name="send_response handler",
        session_id=event.q_id,
    )
    if type(event) is events.Evaluation:
        message = f"\nQuestion:\n{event.question}\nResponse:\n{event.response}\nSummary:\n{event.summary}\nIssues:\n{event.issues}\nPlausibility:\n{event.plausibility}\nFactual Consistency:\n{event.factual_consistency}\nClarity:\n{event.clarity}\nCompleteness:\n{event.completeness}"
    elif type(event) is events.Response:
        message = f"\nQuestion:\n{event.question}\nResponse:\n{event.response}"
    notifications.send(event.q_id, message)
    return None


@observe()
def send_failure(
    event: Union[events.RejectedAnswer, events.RejectedRequest, events.FailedRequest],
    notifications: AbstractNotifications,
) -> None:
    """
    Sends the failure to the notifications.

    Args:
        event: Union[events.RejectedAnswer, events.RejectedRequest, events.FailedRequest]: The event to send.
        notifications: AbstractNotifications: The notifications to use.

    Returns:
        None
    """
    langfuse_context.update_current_trace(
        name="send_rejected handler",
        session_id=event.q_id,
    )

    if type(event) is events.FailedRequest:
        message = f"\nQuestion:\n{event.question}\nException:\n{event.exception}"
    elif type(event) is events.RejectedRequest:
        message = (
            f"\nQuestion:\n{event.question}\n was rejected. Response:\n{event.response}"
        )
    elif type(event) is events.RejectedAnswer:
        message = f"\nAnswer to question:\n{event.question}\n was rejected. Reason:\n{event.rejection}\n Wrong answer:\n{event.response}"
    else:
        raise ValueError("Invalid event type")

    notifications.send(event.q_id, message)

    return None


EVENT_HANDLERS = {
    events.FailedRequest: [send_failure],
    events.Response: [send_response],
    events.RejectedRequest: [send_failure],
    events.RejectedAnswer: [send_failure],
    events.Evaluation: [send_response],
}

COMMAND_HANDLERS = {
    commands.Question: answer,
}
