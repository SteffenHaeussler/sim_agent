from langfuse.decorators import observe
from loguru import logger

from src.agent import config
from src.agent.adapters.adapter import AbstractAdapter
from src.agent.adapters.notifications import AbstractNotifications
from src.agent.domain import commands, events, model


class InvalidQuestion(Exception):
    pass


@observe()
def answer(command: commands.Question, adapter: AbstractAdapter) -> None:
    """service layer has only one abstraction: uow"""

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

    return None


@observe()
def send_response(
    event: events.Response,
    notifications: AbstractNotifications,
):
    message = f"\nQuestion:\n{event.question}\nResponse:\n{event.response}"
    notifications.send(event.q_id, message)
    return None


@observe()
def send_failure(
    event: events.FailedRequest,
    notifications: AbstractNotifications,
):
    message = f"\nQuestion:\n{event.question}\nException:\n{event.exception}"
    notifications.send(event.q_id, message)

    return None


EVENT_HANDLERS = {
    events.FailedRequest: [send_failure],
    events.Response: [send_response],
}

COMMAND_HANDLERS = {
    commands.Question: answer,
}
