"""
Simplified SQL Construction Agent - Builds the final SQL query.
"""

from src.agent.adapters.llm import LLM
from .models import (
    SQLConstructionResponse, 
    GroundingResponse, 
    FilterResponse, 
    JoinInferenceResponse, 
    AggregationResponse
)
from .prompts import SQL_CONSTRUCTION_AGENT_PROMPT


class SQLConstructionAgent:
    """Agent that builds the final SQL query from analyzed components."""
    
    def __init__(self, llm: LLM):
        self.llm = llm
        
    def process(
        self, 
        question: str, 
        grounding_results: GroundingResponse,
        filter_results: FilterResponse,
        join_results: JoinInferenceResponse,
        aggregation_results: AggregationResponse
    ) -> SQLConstructionResponse:
        """Build the final SQL query from all components."""
        
        prompt = SQL_CONSTRUCTION_AGENT_PROMPT.format(
            question=question,
            grounding_results=grounding_results.model_dump(),
            filter_results=filter_results.model_dump(),
            join_results=join_results.model_dump(),
            aggregation_results=aggregation_results.model_dump()
        )
        
        try:
            response = self.llm.use(prompt, SQLConstructionResponse)
            return response
        except Exception as e:
            # Fallback response
            return SQLConstructionResponse(
                sql_query="SELECT 'Error: Could not construct SQL' as error_message",
                explanation=f"SQL construction failed: {str(e)}",
                reasoning="SQL construction agent failed"
            )