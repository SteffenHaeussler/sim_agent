from abc import ABC, abstractmethod


class AbstractNotifications(ABC):
    """
    AbstractNotifications is an abstract base class for all notifications.

    Methods:
        - send(self, destination: str, message: str) -> None: Send a notification.
    """

    @abstractmethod
    def send(self, destination: str, message: str) -> None:
        raise NotImplementedError


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


class ApiNotifications(AbstractNotifications):
    """
    ApiNotifications is a class that sends notifications to the API.

    Methods:
        - send(self, destination: str, message: str) -> None: Send a notification.
    """

    temp = {}

    def send(self, destination: str, message: str) -> None:
        self.temp[destination] = message
