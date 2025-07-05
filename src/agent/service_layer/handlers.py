from typing import Union

from langfuse import get_client, observe
from loguru import logger

from src.agent import config
from src.agent.adapters.adapter import AbstractAdapter
from src.agent.adapters.notifications import AbstractNotifications
from src.agent.domain import commands, events, model, sql_model


class InvalidQuestion(Exception):
    pass


@observe()
def answer(
    command: commands.Question,
    adapter: AbstractAdapter,
    notifications: AbstractNotifications = None,
) -> None:
    """
    Handles incoming questions.

    Args:
        command: commands.Question: The question to answer.
        adapter: AbstractAdapter: The adapter to use.
        notifications: AbstractNotifications: The notifications to use for real-time updates.

    Returns:
        None
    """
    langfuse = get_client()

    langfuse.update_current_trace(
        name="answer handler",
        session_id=command.q_id,
    )

    if not command or not command.question:
        raise InvalidQuestion("No question asked")

    agent = model.BaseAgent(command, config.get_agent_config())
    adapter.add(agent)

    # adapter for execution and agent for internal logic
    while not agent.is_answered and command is not None:
        # Send real-time status update
        if type(command) in STEP_NAMES and notifications:
            status_event = events.StatusUpdate(
                step_name=STEP_NAMES[type(command)], q_id=agent.q_id
            )
            # Send immediately to WebSocket clients
            for notification in notifications:
                notification.send(agent.q_id, status_event)

        logger.info(f"Calling Adapter with command: {type(command)}")
        updated_command = adapter.answer(command)
        command = agent.update(updated_command)

        if agent.send_response:
            event = agent.send_response

            for notification in notifications:
                notification.send(event.q_id, event)

            agent.send_response = None

        if agent.evaluation:
            event = agent.evaluation

            for notification in notifications:
                notification.send(event.q_id, event)

    end_event = events.EndOfEvent(q_id=agent.q_id)

    for notification in notifications:
        notification.send(end_event.q_id, end_event)

    return None


@observe()
def query(
    command: commands.SQLQuestion,
    adapter: AbstractAdapter,
    notifications: AbstractNotifications = None,
) -> None:
    """
    Handles incoming questions.

    Args:
        command: commands.Question: The question to answer.
        adapter: AbstractAdapter: The adapter to use.
        notifications: AbstractNotifications: The notifications to use for real-time updates.

    Returns:
        None
    """
    langfuse = get_client()

    langfuse.update_current_trace(
        name="query handler",
        session_id=command.q_id,
    )
    if not command or not command.question:
        raise InvalidQuestion("No question asked")

    agent = sql_model.SQLBaseAgent(command, config.get_agent_config())
    adapter.add(agent)

    # adapter for execution and agent for internal logic
    while not agent.is_answered and command is not None:
        # Send real-time status update
        if type(command) in STEP_NAMES and notifications:
            status_event = events.StatusUpdate(
                step_name=STEP_NAMES[type(command)], q_id=agent.q_id
            )
            # Send immediately to WebSocket clients
            for notification in notifications:
                notification.send(agent.q_id, status_event)

        logger.info(f"Calling Adapter with command: {type(command)}")
        updated_command = adapter.answer(command)
        command = agent.update(updated_command)

        if agent.send_response:
            event = agent.send_response

            for notification in notifications:
                notification.send(event.q_id, event)

            agent.send_response = None

        if agent.evaluation:
            event = agent.evaluation

            for notification in notifications:
                notification.send(event.q_id, event)

    end_event = events.EndOfEvent(q_id=agent.q_id)

    for notification in notifications:
        notification.send(end_event.q_id, end_event)

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
    langfuse = get_client()

    langfuse.update_current_trace(
        name="send_response handler",
        session_id=event.q_id,
    )

    for notification in notifications:
        notification.send(event.q_id, event)
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
    langfuse = get_client()

    langfuse.update_current_trace(
        name="send_rejected handler",
        session_id=event.q_id,
    )

    for notification in notifications:
        notification.send(event.q_id, event)

    return None


@observe()
def send_status_update(
    event: events.StatusUpdate,
    notifications: AbstractNotifications,
) -> None:
    """
    Sends the status update to the notifications.

    Args:
        event: events.StatusUpdate: The status update event to send.
        notifications: AbstractNotifications: The notifications to use.

    Returns:
        None
    """
    langfuse = get_client()

    langfuse.update_current_trace(
        name="send_status_update handler",
        session_id=event.q_id,
    )

    for notification in notifications:
        notification.send(event.q_id, event)

    return None


EVENT_HANDLERS = {
    events.FailedRequest: [send_failure],
    events.Response: [send_response],
    events.RejectedRequest: [send_failure],
    events.RejectedAnswer: [send_failure],
    events.Evaluation: [send_response],
    events.EndOfEvent: [send_response],
    events.StatusUpdate: [send_status_update],
}

COMMAND_HANDLERS = {
    commands.Question: answer,
    commands.SQLQuestion: query,
}

# Step name mapping for status updates
STEP_NAMES = {
    commands.Question: "Processing...",
    commands.Check: "Checking...",
    commands.Retrieve: "Retrieving...",
    commands.Rerank: "Enhancing...",
    commands.Enhance: "Finetuning...",
    commands.UseTools: "Answering...",
    commands.LLMResponse: "Finalizing...",
    commands.FinalCheck: "Evaluating...",
    commands.SQLQuestion: "Processing...",
    commands.SQLCheck: "Checking...",
    commands.SQLGrounding: "Grounding...",
    commands.SQLFilter: "Filtering...",
    commands.SQLJoinInference: "Joining...",
    commands.SQLAggregation: "Aggregating...",
    commands.SQLConstruction: "Constructing...",
    commands.SQLValidation: "Validating...",
    commands.SQLExecution: "Executing...",
}
