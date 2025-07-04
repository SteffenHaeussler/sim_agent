"""
Simplified Pydantic response models for SQL agents.
"""

from typing import List, Optional


from pydantic import BaseModel


class TableMapping(BaseModel):
    """Maps user term to database table."""

    user_term: str
    table_name: str
    confidence: float


class ColumnMapping(BaseModel):
    """Maps user term to database column."""

    user_term: str
    table_name: str
    column_name: str
    confidence: float


class GroundingResponse(BaseModel):
    """Response from Grounding Agent."""

    table_mappings: List[TableMapping]
    column_mappings: List[ColumnMapping]
    reasoning: str


class FilterCondition(BaseModel):
    """Represents a WHERE clause condition."""

    column: str
    operator: str  # =, >, <, LIKE, etc.
    value: str
    reasoning: str


class FilterResponse(BaseModel):
    """Response from Filter Agent."""

    conditions: List[FilterCondition]
    reasoning: str


class JoinPath(BaseModel):
    """Represents a join between two tables."""

    from_table: str
    to_table: str
    from_column: str
    to_column: str
    join_type: str = "INNER"


class JoinInferenceResponse(BaseModel):
    """Response from Join Inference Agent."""

    joins: List[JoinPath]
    reasoning: str


class AggregationFunction(BaseModel):
    """Represents an aggregation function."""

    function: str  # COUNT, SUM, AVG, etc.
    column: Optional[str] = None
    alias: Optional[str] = None


class AggregationResponse(BaseModel):
    """Response from Aggregation Agent."""

    aggregations: List[AggregationFunction]
    group_by_columns: List[str] = []
    is_aggregation_query: bool
    reasoning: str


class SQLConstructionResponse(BaseModel):
    """Response from SQL Construction Agent."""

    sql_query: str
    explanation: str
    reasoning: str


class ValidationIssue(BaseModel):
    """Represents a validation issue."""

    severity: str  # error, warning, info
    message: str


class ValidationResponse(BaseModel):
    """Response from Validation Agent."""

    is_valid: bool
    issues: List[ValidationIssue]
    corrected_sql: Optional[str] = None
    confidence_score: float
    reasoning: str


class SQLGenerationResponse(BaseModel):
    """Final response from SQL generation workflow."""

    sql_query: str
    explanation: str
    confidence: float
    validation_passed: bool
    execution_time_ms: float
