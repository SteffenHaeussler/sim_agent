"""
Simplified Validation Agent - Validates generated SQL queries.
"""

from src.agent.adapters.llm import LLM
from src.agent.adapters.database import BaseDatabaseAdapter
from .models import ValidationResponse, ValidationIssue
from .prompts import VALIDATION_AGENT_PROMPT


class ValidationAgent:
    """Agent that validates and reflects on generated SQL queries."""
    
    def __init__(self, llm: LLM, db_adapter: BaseDatabaseAdapter):
        self.llm = llm
        self.db_adapter = db_adapter
        
    def _get_schema_info(self) -> str:
        """Get schema information for validation."""
        with self.db_adapter as db:
            schema = db.get_schema()
            
        if not schema:
            return "No schema available"
            
        schema_lines = []
        for table_name, table in schema.tables.items():
            schema_lines.append(f"Table: {table_name}")
            for column in table.columns:
                schema_lines.append(f"  - {column.name} ({column.type})")
            schema_lines.append("")
            
        return "\n".join(schema_lines)
    
    def process(self, question: str, sql_query: str) -> ValidationResponse:
        """Validate the generated SQL query."""
        schema_info = self._get_schema_info()
        
        prompt = VALIDATION_AGENT_PROMPT.format(
            question=question,
            sql_query=sql_query,
            schema_info=schema_info
        )
        
        try:
            response = self.llm.use(prompt, ValidationResponse)
            return response
        except Exception as e:
            # Fallback response
            return ValidationResponse(
                is_valid=False,
                issues=[ValidationIssue(severity="error", message=f"Validation failed: {str(e)}")],
                corrected_sql=None,
                confidence_score=0.0,
                reasoning="Validation agent failed"
            )