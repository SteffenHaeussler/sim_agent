from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel

################################################################################
# pydantic models - between agent and adapters
################################################################################


class LLMResponseModel(BaseModel):
    chain_of_thought: str
    response: str


class KBResponse(BaseModel):
    description: str
    score: float
    id: str
    tag: str
    name: str


class RerankResponse(BaseModel):
    question: str
    text: str
    score: float
    id: str
    tag: str
    name: str


################################################################################
# Internal Commands
################################################################################


class Command:
    pass


@dataclass
class Check(Command):
    question: str
    q_id: str
    is_okay: bool


@dataclass
class Enhance(Command):
    question: str
    q_id: str
    response: Optional[str] = None
    chain_of_thought: Optional[str] = None


@dataclass
class Question(Command):
    question: str
    q_id: str


@dataclass
class Retrieve(Command):
    question: str
    q_id: str
    candidates: Optional[List[KBResponse]] = None


@dataclass
class Rerank(Command):
    question: str
    q_id: str
    candidates: Optional[List[KBResponse]] = None


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
