"""
Simplified Filter Agent - Extracts WHERE clause conditions.
"""

from typing import List

from src.agent.adapters.llm import LLM
from .models import FilterResponse, FilterCondition, ColumnMapping
from .prompts import FILTER_AGENT_PROMPT


class FilterAgent:
    """Agent that extracts filter conditions from natural language questions."""
    
    def __init__(self, llm: LLM):
        self.llm = llm
        
    def _format_mapped_columns(self, column_mappings: List[ColumnMapping]) -> str:
        """Format column mappings for the prompt."""
        if not column_mappings:
            return "No columns mapped"
            
        formatted = []
        for mapping in column_mappings:
            formatted.append(f"- {mapping.table_name}.{mapping.column_name}")
            
        return "\n".join(formatted)
    
    def process(self, question: str, column_mappings: List[ColumnMapping]) -> FilterResponse:
        """Extract filter conditions from question."""
        mapped_columns_str = self._format_mapped_columns(column_mappings)
        
        prompt = FILTER_AGENT_PROMPT.format(
            question=question,
            mapped_columns=mapped_columns_str
        )
        
        try:
            response = self.llm.use(prompt, FilterResponse)
            return response
        except Exception as e:
            # Fallback response
            return FilterResponse(
                conditions=[],
                reasoning=f"Filter agent failed: {str(e)}"
            )