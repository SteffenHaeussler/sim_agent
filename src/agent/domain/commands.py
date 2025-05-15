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
    q_id: str
    answer: Optional[List[Dict]] = None


@dataclass
class Rerank(Command):
    question: str
    q_id: str
    answer: Optional[List[Dict]] = None


@dataclass
class UseTools(Command):
    question: str
    q_id: str
    response: Optional[str] = None


@dataclass
class LLMResponse(Command):
    question: str
    q_id: str
    response: Optional[str] = None
    chain_of_thought: Optional[str] = None


################################################################################
# pydantic models
################################################################################


class LLMResponseModel(BaseModel):
    chain_of_thought: str
    response: str
