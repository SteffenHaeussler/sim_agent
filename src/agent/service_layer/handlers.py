from src.agent.adapters.agent import AbstractAgent
from src.agent.adapters.notifications import AbstractNotifications
from src.agent.domain import commands, events, model


class InvalidQuestion(Exception):
    pass


def answer(command: commands.Question, agent: AbstractAgent) -> str:
    """service layer has only one abstraction: uow"""

    if not command or not command.question:
        raise InvalidQuestion("No question asked")

    agent_model = model.Agent(command)
    agent.add(agent_model)

    response = agent.answer(agent.enhancement)
    agent.use_tools(response)

    response = agent.answer(agent.tool_answer)

    return agent.response


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
