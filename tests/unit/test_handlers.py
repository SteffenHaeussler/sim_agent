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
            response.response = "test tools response"
            response.memory = ["test memory"]
        elif isinstance(command, commands.LLMResponse):
            response = command
            response.response = "test second llm response"
            response.chain_of_thought = "test chain of thought"
        elif isinstance(command, commands.Enhance):
            response = command
            response.response = "test first llm response"
            response.chain_of_thought = "test chain of thought"
        elif isinstance(command, commands.Rerank):
            response = command
            response.candidates = [
                commands.RerankResponse(
                    question="test question",
                    text="test text",
                    score=0.5,
                    id="test id",
                    tag="test tag",
                    name="test name",
                )
            ]
        elif isinstance(command, commands.Retrieve):
            response = command
            response.response = "test response"
        elif isinstance(command, commands.Check):
            if command.q_id == "test_session_id":
                response = command
                response.response = "test first llm response"
                response.approved = True
            else:
                response = command
                response.response = "test second llm response"
        elif isinstance(command, commands.FinalCheck):
            response = command
            response.response = "test response"
            response.chain_of_thought = "test chain of thought"
            response.approved = True
            response.summary = "test summary"
            response.issues = []
            response.plausibility = "test plausibility"
            response.factual_consistency = "test factual consistency"
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

        assert agent.response.response == "test second llm response"

    def test_sends_notification(self):
        fake_notifs = FakeNotifications()
        bus = bootstrap(adapter=FakeAdapter(), notifications=fake_notifs)
        bus.handle(commands.Question("test query", "test_session_id"))

        assert fake_notifs.sent["test_session_id"] == [
            "\nQuestion:\ntest query\nResponse:\ntest second llm response",
            "\nQuestion:\ntest query\nResponse:\ntest second llm response\nSummary:\ntest summary\nIssues:\n[]\nPlausibility:\ntest plausibility\nFactual Consistency:\ntest factual consistency\nClarity:\nNone\nCompleteness:\nNone",
        ]

    def test_sends_rejected_notification(self):
        fake_notifs = FakeNotifications()
        bus = bootstrap(adapter=FakeAdapter(), notifications=fake_notifs)
        bus.handle(commands.Question("test query", "test_rejected_id"))

        assert fake_notifs.sent["test_rejected_id"] == [
            "\nQuestion:\ntest query\n was rejected. Response:\ntest second llm response",
        ]
