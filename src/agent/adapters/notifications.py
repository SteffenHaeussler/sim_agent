import smtplib
from abc import ABC, abstractmethod
from email.message import EmailMessage

import httpx

from src.agent.config import get_email_config, get_slack_config


class AbstractNotifications(ABC):
    """
    AbstractNotifications is an abstract base class for all notifications.

    Methods:
        - send(self, destination: str, message: str) -> None: Send a notification.
    """

    @abstractmethod
    def send(self, destination: str, message: str) -> None:
        raise NotImplementedError


class ApiNotifications(AbstractNotifications):
    """
    ApiNotifications is a class that sends notifications to the API.

    Methods:
        - send(self, destination: str, message: str) -> None: Send a notification.
    """

    temp = {}

    def send(self, destination: str, message: str) -> None:
        self.temp[destination] = message


class CliNotifications(AbstractNotifications):
    """
    CliNotifications is a class that sends notifications to the CLI.

    Methods:
        - send(self, destination: str, message: str) -> None: Send a notification.
    """

    def send(self, destination: str, message: str) -> None:
        """
        Send a notification.

        Args:
            destination: str: The destination to send the notification.
            message: str: The message to send.
        """
        print("send notification:", message)


class SlackNotifications(AbstractNotifications):
    """
    SlackNotifications is a class that sends notifications to Slack.

    Methods:
        - send(self, destination: str, message: str) -> None: Send a notification.
    """

    def __init__(self):
        self.config = get_slack_config()

    def send(self, destination: str, message: str) -> None:
        """
        Send a notification.

        Args:
            destination: str: The destination to send the notification.
            message: str: The message to send.
        """
        httpx.post(
            self.config["slack_webhook_url"],
            json={"text": message},
            headers={"Content-Type": "application/json"},
        )


class EmailNotifications(AbstractNotifications):
    """
    EmailNotifications is a class that sends notifications to an email.

    Methods:
        - send(self, destination: str, message: str) -> None: Send a notification.
    """

    def __init__(self):
        self.config = get_email_config()

    def send(self, destination, message):
        msg = EmailMessage()
        msg["Subject"] = "Subject: apropos service notification"
        msg["From"] = self.config["sender_email"]
        msg["To"] = self.config["receiver_email"]
        msg.set_content(message)

        with smtplib.SMTP(
            self.config["smtp_host"], port=self.config["smtp_port"]
        ) as server:
            server.starttls()  # Secure the connection
            server.login(self.config["sender_email"], self.config["app_password"])
            server.send_message(msg)
