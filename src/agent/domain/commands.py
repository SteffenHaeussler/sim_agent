from dataclasses import dataclass
from typing import Dict, List, Optional

from pydantic import BaseModel


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
    answer: Optional[List[Dict]] = None
    q_id: str


@dataclass
class Rerank(Command):
    question: str
    answer: Optional[List[Dict]] = None
    q_id: str


@dataclass
class UseTools(Command):
    question: str
    response: Optional[str] = None
    q_id: str


class LLMResponse(Command):
    question: str
    chain_of_thought: str
    q_id: str
    response: Optional[str] = None


################################################################################
# pydantic models
################################################################################


class LLMResponseModel(BaseModel):
    chain_of_thought: str
    response: str
