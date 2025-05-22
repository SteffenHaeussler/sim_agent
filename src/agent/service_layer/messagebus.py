from typing import Callable, Dict, List, Type, Union

from loguru import logger

from src.agent.adapters import adapter
from src.agent.domain import commands, events

Message = Union[commands.Command, events.Event]


class MessageBus:
    """
    MessageBus is a class that handles command and events.

    Commands are send to the agent and events are sent to the notifications.

    Args:
        adapter: adapter.AbstractAdapter: The adapter to use.
        event_handlers: Dict[Type[events.Event], List[Callable]]: The event handlers.
        command_handlers: Dict[Type[commands.Command], Callable]: The command handlers.

    Returns:
        None
    """

    def __init__(
        self,
        adapter: adapter.AbstractAdapter,
        event_handlers: Dict[Type[events.Event], List[Callable]],
        command_handlers: Dict[Type[commands.Command], Callable],
    ) -> None:
        self.adapter = adapter
        self.event_handlers = event_handlers
        self.command_handlers = command_handlers

    def handle(
        self,
        message: Message,
    ) -> None:
        """
        Handles incoming messages or gets them from the internal queue.

        Args:
            message: Message: The message to handle.

        Returns:
            None
        """
        self.queue = [message]
        while self.queue:
            message = self.queue.pop(0)
            if isinstance(message, events.Event):
                self.handle_event(message)
            elif isinstance(message, commands.Command):
                self.handle_command(message)
            else:
                raise Exception(f"{message} was not an Event or Command")

    def handle_event(
        self,
        event: events.Event,
    ) -> None:
        """
        Handles incoming events and collects new commands/events from the agent.events list.

        Args:
            event: events.Event: The event to handle.

        Returns:
            None
        """
        for handler in self.event_handlers[type(event)]:
            try:
                logger.debug(f"handling event {str(event)} with handler {str(handler)}")
                handler(event)
                self.queue.extend(self.adapter.collect_new_events())
            except Exception:
                logger.exception(f"Exception handling event {event}")
                continue

    def handle_command(
        self,
        command: commands.Command,
    ) -> None:
        """
        Handles incoming commands and collects new commands/events from the agent.events list.

        Args:
            command: commands.Command: The command to handle.
        """
        logger.debug("handling command %s", command)
        try:
            handler = self.command_handlers[type(command)]
            handler(command)
            self.queue.extend(self.adapter.collect_new_events())
        except Exception:
            logger.exception("Exception handling command %s", command)
            raise
