from typing import Any, Dict, List, Optional

from pydantic import BaseModel

################################################################################
# Tools pydantic models - between agent and adapters
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
# SQL building blocks
################################################################################


class AggregationFunction(BaseModel):
    function: str  # COUNT, SUM, AVG, etc.
    column: Optional[str] = None
    alias: Optional[str] = None


class ColumnMapping(BaseModel):
    user_term: str
    table_name: str
    column_name: str
    confidence: float


class FilterCondition(BaseModel):
    column: str
    operator: str  # =, >, <, LIKE, etc.
    value: str
    chain_of_thought: Optional[str] = None


class JoinPath(BaseModel):
    from_table: str
    to_table: str
    from_column: str
    to_column: str
    join_type: str = "INNER"


class TableMapping(BaseModel):
    user_term: str
    table_name: str
    confidence: float


################################################################################
# SQL pydantic models - between agent and adapters
################################################################################


class AggregationResponse(BaseModel):
    """Response from Aggregation Agent."""

    aggregations: List[AggregationFunction]
    group_by_columns: List[str] = []
    is_aggregation_query: bool
    chain_of_thought: Optional[str] = None


class ConstructionResponse(BaseModel):
    sql_query: str
    chain_of_thought: Optional[str] = None


class FilterResponse(BaseModel):
    """Response from Filter Agent."""

    conditions: List[FilterCondition]
    chain_of_thought: Optional[str] = None


class GroundingResponse(BaseModel):
    table_mappings: List[TableMapping]
    column_mappings: List[ColumnMapping]
    chain_of_thought: Optional[str] = None


class JoinInferenceResponse(BaseModel):
    joins: List[JoinPath]
    chain_of_thought: Optional[str] = None


class ValidationResponse(BaseModel):
    is_valid: bool
    issues: Optional[List[str]] = None
    corrected_sql: Optional[str] = None
    confidence: float
    chain_of_thought: Optional[str] = None


################################################################################
# Internal Tool Commands
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


################################################################################
# Internal SQL Commands
################################################################################


class SQLAggregation(Command):
    question: str
    q_id: str
    aggregations: Optional[List[AggregationFunction]] = None
    group_by_columns: Optional[List[str]] = None
    is_aggregation_query: Optional[bool] = None
    chain_of_thought: Optional[str] = None


class SQLCheck(Command):
    question: str
    q_id: str
    approved: Optional[bool] = None
    chain_of_thought: Optional[str] = None
    response: Optional[str] = None
    schema_info: Any


class SQLConstruction(Command):
    question: str
    q_id: str
    schema_info: Optional[Any] = None
    column_mappings: Optional[List[ColumnMapping]] = None
    table_mappings: Optional[List[TableMapping]] = None
    conditions: Optional[List[FilterCondition]] = None
    joins: Optional[List[JoinPath]] = None
    aggregations: Optional[List[AggregationFunction]] = None
    group_by_columns: Optional[List[str]] = None
    is_aggregation_query: Optional[bool] = None
    sql_query: Optional[str] = None
    chain_of_thought: Optional[str] = None


class SQLExecution(Command):
    question: str
    q_id: str


class SQLFilter(Command):
    question: str
    q_id: str
    table_mappings: Optional[List[TableMapping]] = None
    schema_info: Optional[Any] = None
    conditions: Optional[List[FilterCondition]] = None
    chain_of_thought: Optional[str] = None


class SQLGrounding(Command):
    question: str
    q_id: str
    schema_info: Any
    table_mappings: Optional[List[TableMapping]] = None
    column_mappings: Optional[List[ColumnMapping]] = None
    chain_of_thought: Optional[str] = None


class SQLJoinInference(Command):
    question: str
    q_id: str
    schema_info: Any
    table_mappings: Optional[List[TableMapping]]
    joins: Optional[List[JoinPath]] = None
    chain_of_thought: Optional[str] = None


class SQLQuestion(Command):
    question: str
    q_id: str
    schema_info: Optional[Any] = None


class SQLValidation(Command):
    question: str
    q_id: str
    is_valid: Optional[bool] = None
    issues: Optional[List[str]] = None
    corrected_sql: Optional[str] = None
    confidence: Optional[float] = None
    chain_of_thought: Optional[str] = None
