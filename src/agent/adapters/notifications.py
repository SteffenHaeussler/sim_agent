import asyncio
import smtplib
import time
from abc import ABC, abstractmethod
from email.message import EmailMessage

import httpx
from fastapi import WebSocket
from loguru import logger
from starlette.websockets import WebSocketState

from src.agent.config import get_email_config, get_slack_config
from src.agent.observability.context import connected_clients


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


# class WSNotifications(AbstractNotifications):
# def send(self, destination: str, message: str) -> None:
#     """
#     Send a notification to the destination (session_id) if connected.
#     """

#     breakpoint()


#     websocket = connected_clients.get(destination)
#     if websocket:
#         try:
#             # Schedule sending the message (FastAPI uses asyncio)
#             asyncio.create_task(websocket.send_text(message))
#         except Exception as e:
#             print(f"Failed to send to {destination}: {e}")
class WSNotifications(AbstractNotifications):
    def send(self, destination: str, message: str) -> None:
        client_info = connected_clients.get(destination)
        if client_info:
            websocket: WebSocket = client_info["ws"]
            target_loop: asyncio.AbstractEventLoop = client_info["loop"]

            # Ensure the websocket is still connected before trying to send
            if websocket.client_state == WebSocketState.CONNECTED:
                # Prepare the coroutine
                coro = websocket.send_text(message)

                future = asyncio.run_coroutine_threadsafe(coro, target_loop)

                try:
                    future.result(
                        timeout=5
                    )  # Wait up to 5 seconds for the send to complete
                    logger.info(f"Message successfully sent to {destination}")
                    # Update last_event_time after successful send
                    client_info["last_event_time"] = time.time()
                except asyncio.TimeoutError:
                    logger.error(f"Timeout sending message to {destination}")
                    # Optionally, you might want to clean up or mark this client as problematic
                except Exception as e:
                    logger.error(
                        f"Error sending message to {destination}: {e}", exc_info=True
                    )
            else:
                logger.warning(
                    f"Attempted to send to disconnected WebSocket: {destination}"
                )
                connected_clients.pop(destination, None)  # Clean up disconnected client
        else:
            logger.warning(f"WebSocket client not found for destination: {destination}")
            return None
