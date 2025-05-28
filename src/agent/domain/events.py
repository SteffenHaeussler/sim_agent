from dataclasses import dataclass
from typing import Dict, List, Optional


class Event:
    pass


@dataclass
class Evaluation(Event):
    question: str
    response: str
    q_id: str
    approved: bool
    summary: str
    issues: List[str]
    plausibility: str
    factual_consistency: str
    clarity: str
    completeness: str


@dataclass
class FailedRequest(Event):
    question: str
    exception: Exception
    q_id: str


@dataclass
class RejectedRequest(Event):
    question: str
    response: str
    q_id: str


@dataclass
class Response(Event):
    question: str
    response: str
    q_id: str


@dataclass
class RejectedAnswer(Response):
    question: str
    response: str
    rejection: str
    q_id: str


@dataclass
class ResponseWithData(Response):
    question: str
    response: str
    q_id: str
    data: Optional[Dict[str, str]] = None
