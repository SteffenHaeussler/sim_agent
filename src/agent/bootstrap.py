import inspect

from src.agent.adapters import agent
from src.agent.adapters.notifications import AbstractNotifications, CliNotifications
from src.agent.service_layer import handlers, messagebus


def bootstrap(
    agent: agent.AbstractAgent = agent.AbstractAgent(),
    notifications: AbstractNotifications = None,
) -> messagebus.MessageBus:
    if notifications is None:
        notifications = CliNotifications()

    dependencies = {"agent": agent, "notifications": notifications}

    injected_event_handlers = {
        event_type: [
            inject_dependencies(handler, dependencies) for handler in event_handlers
        ]
        for event_type, event_handlers in handlers.EVENT_HANDLERS.items()
    }
    injected_command_handlers = {
        command_type: inject_dependencies(handler, dependencies)
        for command_type, handler in handlers.COMMAND_HANDLERS.items()
    }

    return messagebus.MessageBus(
        agent=agent,
        event_handlers=injected_event_handlers,
        command_handlers=injected_command_handlers,
    )


def inject_dependencies(handler, dependencies):
    params = inspect.signature(handler).parameters
    deps = {
        name: dependency for name, dependency in dependencies.items() if name in params
    }
    return lambda message: handler(message, **deps)
