from typing import Callable, Dict, List, Type, Union

from loguru import logger

from src.agent.adapters import agent
from src.agent.domain import commands, events

Message = Union[commands.Command, events.Event]


class MessageBus:
    def __init__(
        self,
        agent: agent.AbstractAgent,
        event_handlers: Dict[Type[events.Event], List[Callable]],
        command_handlers: Dict[Type[commands.Command], Callable],
    ):
        self.agent = agent
        self.event_handlers = event_handlers
        self.command_handlers = command_handlers

    def handle(
        self,
        message: Message,
    ):
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
    ):
        for handler in self.event_handlers[type(event)]:
            try:
                logger.debug(f"handling event {str(event)} with handler {str(handler)}")
                handler(event)
                self.queue.extend(self.agent.collect_new_events())
            except Exception:
                logger.exception(f"Exception handling event {event}")
                continue

    def handle_command(
        self,
        command: commands.Command,
    ):
        logger.debug("handling command %s", command)
        try:
            handler = self.command_handlers[type(command)]
            handler(command)
            self.queue.extend(self.agent.collect_new_events())
        except Exception:
            logger.exception("Exception handling command %s", command)
            raise
