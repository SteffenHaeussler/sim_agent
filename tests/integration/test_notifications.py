from unittest.mock import patch

from src.agent.adapters.adapter import AgentAdapter
from src.agent.adapters.notifications import ApiNotifications, CliNotifications
from src.agent.bootstrap import bootstrap
from src.agent.domain import events


class TestNotification:
    def test_send_cli_notification_called(self):
        bus = bootstrap(
            adapter=AgentAdapter(),
            notifications=CliNotifications(),
        )
        with patch.object(CliNotifications, "send", return_value=None) as mock_send:
            bus.handle(
                events.Response(
                    question="test_query",
                    response="test_response",
                    q_id="test_session_id",
                )
            )
            mock_send.assert_called_once_with(
                "test_session_id", "\nQuestion:\ntest_query\nResponse:\ntest_response"
            )

    def test_send_api_notification_called(self):
        bus = bootstrap(
            adapter=AgentAdapter(),
            notifications=ApiNotifications(),
        )
        with patch.object(ApiNotifications, "send", return_value=None) as mock_send:
            bus.handle(
                events.Response(
                    question="test_query",
                    response="test_response",
                    q_id="test_session_id",
                )
            )
            mock_send.assert_called_once_with(
                "test_session_id", "\nQuestion:\ntest_query\nResponse:\ntest_response"
            )
