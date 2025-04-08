from dataclasses import dataclass


class Event:
    pass


@dataclass
class Response(Event):
    question: str
    response: str
    q_id: str


@dataclass
class FailedRequest(Event):
    question: str
    exception: Exception
    q_id: str
