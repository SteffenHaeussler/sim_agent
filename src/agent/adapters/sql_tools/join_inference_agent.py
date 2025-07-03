"""
Simplified Join Inference Agent - Determines necessary table joins.
"""

from typing import List

from src.agent.adapters.llm import LLM
from src.agent.adapters.database import BaseDatabaseAdapter
from .models import JoinInferenceResponse, JoinPath, TableMapping
from .prompts import JOIN_INFERENCE_AGENT_PROMPT


class JoinInferenceAgent:
    """Agent that determines necessary table joins for SQL queries."""
    
    def __init__(self, llm: LLM, db_adapter: BaseDatabaseAdapter):
        self.llm = llm
        self.db_adapter = db_adapter
        
    def _get_schema_relationships(self) -> str:
        """Get foreign key relationships from schema."""
        with self.db_adapter as db:
            schema = db.get_schema()
            
        if not schema:
            return "No relationships available"
            
        relationships = []
        for table_name, table in schema.tables.items():
            for fk in table.foreign_keys:
                relationships.append(
                    f"{table_name}.{fk.parent.name} -> {fk.column.table.name}.{fk.column.name}"
                )
        
        return "\n".join(relationships) if relationships else "No foreign keys found"
    
    def process(self, question: str, table_mappings: List[TableMapping]) -> JoinInferenceResponse:
        """Determine necessary joins between tables."""
        required_tables = [mapping.table_name for mapping in table_mappings]
        relationships = self._get_schema_relationships()
        
        prompt = JOIN_INFERENCE_AGENT_PROMPT.format(
            question=question,
            required_tables=", ".join(required_tables),
            schema_relationships=relationships
        )
        
        try:
            response = self.llm.use(prompt, JoinInferenceResponse)
            return response
        except Exception as e:
            # Fallback response
            return JoinInferenceResponse(
                joins=[],
                reasoning=f"Join inference agent failed: {str(e)}"
            )