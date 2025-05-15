from abc import ABC, abstractmethod


class AbstractNotifications(ABC):
    @abstractmethod
    def send(self, destination, message):
        raise NotImplementedError


class CliNotifications(AbstractNotifications):
    def send(self, destination, message):
        print("send notification:", message)


class ApiNotifications(AbstractNotifications):
    temp = {}

    def send(self, destination, message):
        self.temp[destination] = message
