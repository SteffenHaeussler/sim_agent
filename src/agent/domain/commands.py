from typing import Dict, List, Optional

from pydantic import BaseModel

################################################################################
# pydantic models - between agent and adapters
################################################################################


class GuardrailPreCheckModel(BaseModel):
    approved: bool
    chain_of_thought: str
    response: str


class GuardrailPostCheckModel(BaseModel):
    chain_of_thought: str
    approved: bool
    summary: str
    issues: List[str]
    plausibility: str
    factual_consistency: str
    clarity: str
    completeness: str


class KBResponse(BaseModel):
    description: str
    score: float
    id: str
    tag: str
    name: str


class LLMResponseModel(BaseModel):
    chain_of_thought: str
    response: str


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


class Command(BaseModel):
    pass


class Check(Command):
    question: str
    q_id: str
    approved: Optional[bool] = None
    chain_of_thought: Optional[str] = None
    response: Optional[str] = None


class Enhance(Command):
    question: str
    q_id: str
    response: Optional[str] = None
    chain_of_thought: Optional[str] = None


class FinalCheck(Command):
    question: str
    q_id: str
    chain_of_thought: Optional[str] = None
    approved: Optional[bool] = None
    summary: Optional[str] = None
    issues: Optional[List[str]] = None
    plausibility: Optional[str] = None
    factual_consistency: Optional[str] = None
    clarity: Optional[str] = None
    completeness: Optional[str] = None
    data: Optional[Dict[str, str]] = None


class LLMResponse(Command):
    question: str
    q_id: str
    response: Optional[str] = None
    data: Optional[Dict[str, str]] = None
    chain_of_thought: Optional[str] = None


class Question(Command):
    question: str
    q_id: str


class Rerank(Command):
    question: str
    q_id: str
    candidates: Optional[List[KBResponse]] = None


class Retrieve(Command):
    question: str
    q_id: str
    candidates: Optional[List[KBResponse]] = None


class UseTools(Command):
    question: str
    q_id: str
    response: Optional[str] = None
    memory: Optional[List[str]] = None
    data: Optional[Dict[str, str]] = None
