import inspect
from typing import Dict

from fastapi.websockets import WebSocket
from langfuse.decorators import langfuse_context, observe

from src.agent.adapters import adapter
from src.agent.adapters.notifications import AbstractNotifications, CliNotifications
from src.agent.observability.context import ctx_query_id
from src.agent.service_layer import handlers, messagebus

connected_clients: Dict[str, WebSocket] = {}


@observe()
def bootstrap(
    adapter: adapter.AbstractAdapter = adapter.AbstractAdapter(),
    notifications: AbstractNotifications = None,
) -> messagebus.MessageBus:
    """
    Bootstraps the agent.

    Args:
        adapter: adapter.AbstractAdapter: The adapter to use.
        notifications: AbstractNotifications: The notifications to use.

    Returns:
        messagebus.MessageBus: The message bus.
    """

    langfuse_context.update_current_trace(
        name="bootstrap",
        session_id=ctx_query_id.get(),
    )

    if notifications is None:
        notifications = CliNotifications()

    dependencies = {"adapter": adapter, "notifications": notifications}

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
        adapter=adapter,
        event_handlers=injected_event_handlers,
        command_handlers=injected_command_handlers,
        notifications=notifications,
    )


def inject_dependencies(handler, dependencies):
    """
    Injects the dependencies into the handler.

    Args:
        handler: The handler to inject the dependencies into.
        dependencies: The dependencies to inject.
    """

    params = inspect.signature(handler).parameters
    deps = {
        name: dependency for name, dependency in dependencies.items() if name in params
    }
    return lambda message: handler(message, **deps)
