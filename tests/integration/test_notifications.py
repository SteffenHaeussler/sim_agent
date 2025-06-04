from unittest.mock import patch

from src.agent.adapters.adapter import AgentAdapter
from src.agent.adapters.notifications import (
    CliNotifications,
    EmailNotifications,
    SlackNotifications,
    SSENotifications,
    WSNotifications,
)
from src.agent.bootstrap import bootstrap
from src.agent.domain import events


class TestNotification:
    def test_send_cli_notification_called(self):
        bus = bootstrap(
            adapter=AgentAdapter(),
            notifications=[CliNotifications()],
        )
        event = events.Response(
            question="test_query",
            response="test_response",
            q_id="test_session_id",
        )
        with patch.object(CliNotifications, "send", return_value=None) as mock_send:
            bus.handle(event)
            mock_send.assert_called_once_with("test_session_id", event)

    def test_send_email_notification_called(self):
        bus = bootstrap(
            adapter=AgentAdapter(),
            notifications=[EmailNotifications()],
        )
        event = events.Response(
            question="test_query",
            response="test_response",
            q_id="test_session_id",
        )

        with patch.object(EmailNotifications, "send", return_value=None) as mock_send:
            bus.handle(event)
            mock_send.assert_called_once_with("test_session_id", event)

    def test_send_slack_notification_called(self):
        event = events.Response(
            question="test_query",
            response="test_response",
            q_id="test_session_id",
        )

        bus = bootstrap(
            adapter=AgentAdapter(),
            notifications=[SlackNotifications()],
        )
        with patch.object(SlackNotifications, "send", return_value=None) as mock_send:
            bus.handle(event)
            mock_send.assert_called_once_with("test_session_id", event)

    def test_send_ws_notification_called(self):
        event = events.Response(
            question="test_query",
            response="test_response",
            q_id="test_session_id",
        )

        bus = bootstrap(
            adapter=AgentAdapter(),
            notifications=[WSNotifications()],
        )
        with patch.object(WSNotifications, "send", return_value=None) as mock_send:
            bus.handle(event)
            mock_send.assert_called_once_with("test_session_id", event)

    def test_send_sse_notification_called(self):
        event = events.Response(
            question="test_query",
            response="test_response",
            q_id="test_session_id",
        )
        bus = bootstrap(
            adapter=AgentAdapter(),
            notifications=[SSENotifications()],
        )
        with patch.object(SSENotifications, "send", return_value=None) as mock_send:
            bus.handle(event)
            mock_send.assert_called_once_with("test_session_id", event)
