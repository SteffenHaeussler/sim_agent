import inspect

from langfuse.decorators import langfuse_context, observe

from src.agent.adapters import adapter
from src.agent.adapters.notifications import AbstractNotifications, CliNotifications
from src.agent.observability.context import ctx_query_id
from src.agent.service_layer import handlers, messagebus


@observe()
def bootstrap(
    adapter: adapter.AbstractAdapter = adapter.AbstractAdapter(),
    notifications: AbstractNotifications = None,
) -> messagebus.MessageBus:
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
    )


def inject_dependencies(handler, dependencies):
    params = inspect.signature(handler).parameters
    deps = {
        name: dependency for name, dependency in dependencies.items() if name in params
    }
    return lambda message: handler(message, **deps)
