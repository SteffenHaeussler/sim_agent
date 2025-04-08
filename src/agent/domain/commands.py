from dataclasses import dataclass
from typing import Dict, List


class Command:
    pass


@dataclass
class Check(Command):
    question: str
    q_id: str
    is_okay: bool


@dataclass
class Question(Command):
    question: str
    q_id: str


@dataclass
class Retrieve(Command):
    question: str
    response: List[Dict]
    q_id: str


@dataclass
class Rerank(Command):
    question: str
    response: List[Dict]
    q_id: str


@dataclass
class UseTools(Command):
    question: str
    response: str
    q_id: str
