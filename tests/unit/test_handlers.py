from collections import defaultdict

import pytest

from src.agent.adapters.adapter import AbstractAdapter
from src.agent.adapters.notifications import AbstractNotifications
from src.agent.bootstrap import bootstrap
from src.agent.domain import commands
from src.agent.service_layer.handlers import InvalidQuestion


class FakeAdapter(AbstractAdapter):
    def __init__(self):
        super().__init__()

    def answer(self, command):
        if type(command) is commands.Question:
            response = command
        elif isinstance(command, commands.UseTools):
            response = command
        elif isinstance(command, commands.LLMResponse):
            response = command
            response.response = "test response"

        return response


class FakeNotifications(AbstractNotifications):
    def __init__(self):
        self.sent = defaultdict(list)

    def send(self, destination, message):
        self.sent[destination].append(message)


def bootstrap_test_app():
    return bootstrap(adapter=FakeAdapter(), notifications=FakeNotifications())


class TestAnswer:
    def test_answers(self):
        bus = bootstrap_test_app()
        bus.handle(commands.Question("test query", "test_session_id"))

        # get the agent from the adapter
        agent = next(iter(bus.adapter.seen))

        assert agent.q_id == "test_session_id"
        assert agent.question == "test query"

    def test_answer_invalid_question(self):
        bus = bootstrap_test_app()

        with pytest.raises(InvalidQuestion, match="No question asked"):
            bus.handle(commands.Question(None, None))

    def test_for_new_agent(self):
        bus = bootstrap_test_app()

        assert bus.adapter.seen == set()

        bus.handle(commands.Question("test query", "test_session_id"))
        assert bus.adapter.seen != set()

    def test_return_response(self):
        bus = bootstrap_test_app()
        bus.handle(commands.Question("test query", "test_session_id"))

        agent = next(iter(bus.adapter.seen))
        assert agent.response.response == "test response"

    def test_sends_notification(self):
        fake_notifs = FakeNotifications()
        bus = bootstrap(adapter=FakeAdapter(), notifications=fake_notifs)
        bus.handle(commands.Question("test query", "test_session_id"))

        assert fake_notifs.sent["test_session_id"] == [
            "\nQuestion:\ntest query\nResponse:\ntest response",
        ]
