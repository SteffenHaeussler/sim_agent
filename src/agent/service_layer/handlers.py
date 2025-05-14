from src.agent.adapters.adapter import AbstractAdapter
from src.agent.adapters.notifications import AbstractNotifications
from src.agent.domain import commands, events, model


class InvalidQuestion(Exception):
    pass


def answer(
    command: commands.Question, adapter: AbstractAdapter
) -> events.Response | events.FailedRequest:
    """service layer has only one abstraction: uow"""

    if not command or not command.question:
        raise InvalidQuestion("No question asked")

    agent = model.BaseAgent(command)
    adapter.add(agent)

    while not agent.is_answered or command is None:
        breakpoint()
        response = adapter.answer(command)
        command = agent.update(command, response)

    event = adapter.response
    return event


def send_response(
    event: events.Response,
    notifications: AbstractNotifications,
):
    message = f"\nQuestion:\n{event.question}\nResponse:\n{event.response}"
    notifications.send(None, message)
    return None


def send_failure(
    event: events.FailedRequest,
    notifications: AbstractNotifications,
):
    message = f"\nQuestion:\n{event.question}\nException:\n{event.exception}"
    notifications.send(None, message)

    return None


EVENT_HANDLERS = {
    events.FailedRequest: [send_failure],
    events.Response: [send_response],
}

COMMAND_HANDLERS = {
    commands.Question: answer,
}
