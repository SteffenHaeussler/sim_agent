"""
Simplified Grounding Agent - Maps user terms to database schema elements.
"""

from typing import List

from src.agent.adapters.llm import LLM
from src.agent.adapters.database import BaseDatabaseAdapter
from .models import GroundingResponse, TableMapping, ColumnMapping
from .prompts import GROUNDING_AGENT_PROMPT


class GroundingAgent:
    """Agent that maps natural language terms to database schema elements."""
    
    def __init__(self, llm: LLM, db_adapter: BaseDatabaseAdapter):
        self.llm = llm
        self.db_adapter = db_adapter
        
    def _get_schema_info(self) -> str:
        """Get formatted schema information."""
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
    
    def _create_fallback_grounding_response(self, question: str, error: str) -> GroundingResponse:
        """Create a fallback response when LLM fails."""
        words = question.lower().split()
        table_mappings = []
        column_mappings = []
        
        # Simple word-to-table mapping
        if any(word in ["customer", "customers"] for word in words):
            table_mappings.append(TableMapping(user_term="customers", table_name="customers", confidence=0.7))
        if any(word in ["order", "orders"] for word in words):
            table_mappings.append(TableMapping(user_term="orders", table_name="orders", confidence=0.7))
        if any(word in ["product", "products"] for word in words):
            table_mappings.append(TableMapping(user_term="products", table_name="products", confidence=0.7))
        
        return GroundingResponse(
            table_mappings=table_mappings,
            column_mappings=column_mappings,
            reasoning=f"Fallback mapping due to LLM error: {error}"
        )
    
    def process(self, question: str) -> GroundingResponse:
        """Map user question to schema elements."""
        schema_info = self._get_schema_info()
        
        prompt = GROUNDING_AGENT_PROMPT.format(
            question=question,
            schema_info=schema_info
        )
        
        try:
            # Use structured output with the LLM
            response = self.llm.use(prompt, GroundingResponse)
            return response
        except Exception as e:
            # Fallback response
            return self._create_fallback_grounding_response(question, str(e))