"""
Simplified SQL Controller - Orchestrates the SQL generation workflow.
"""

import time
from typing import Optional

from src.agent.adapters.llm import LLM
from src.agent.adapters.database import BaseDatabaseAdapter
from src.agent.config import get_llm_config, get_database_config

from .models import SQLGenerationResponse
from .grounding_agent import GroundingAgent
from .filter_agent import FilterAgent
from .join_inference_agent import JoinInferenceAgent
from .aggregation_agent import AggregationAgent
from .sql_construction_agent import SQLConstructionAgent
from .validation_agent import ValidationAgent


class SQLController:
    """Main controller that orchestrates the SQL generation workflow."""
    
    def __init__(self, llm: Optional[LLM] = None, db_adapter: Optional[BaseDatabaseAdapter] = None):
        # Initialize adapters
        self.llm = llm or LLM(get_llm_config())
        self.db_adapter = db_adapter or BaseDatabaseAdapter(get_database_config())
        
        # Initialize agents
        self.grounding_agent = GroundingAgent(self.llm, self.db_adapter)
        self.filter_agent = FilterAgent(self.llm)
        self.join_inference_agent = JoinInferenceAgent(self.llm, self.db_adapter)
        self.aggregation_agent = AggregationAgent(self.llm)
        self.sql_construction_agent = SQLConstructionAgent(self.llm)
        self.validation_agent = ValidationAgent(self.llm, self.db_adapter)
    
    def process_question(self, question: str) -> SQLGenerationResponse:
        """Generate SQL query from natural language question."""
        start_time = time.time()
        
        try:
            # Step 1: Ground the question to schema elements
            grounding_result = self.grounding_agent.process(question)
            
            # Step 2: Extract filter conditions
            filter_result = self.filter_agent.process(question, grounding_result.column_mappings)
            
            # Step 3: Determine necessary joins
            join_result = self.join_inference_agent.process(question, grounding_result.table_mappings)
            
            # Step 4: Detect aggregation requirements
            aggregation_result = self.aggregation_agent.process(question, grounding_result.column_mappings)
            
            # Step 5: Construct the SQL query
            construction_result = self.sql_construction_agent.process(
                question, grounding_result, filter_result, join_result, aggregation_result
            )
            
            # Step 6: Validate the generated SQL
            validation_result = self.validation_agent.process(question, construction_result.sql_query)
            
            # Determine final SQL
            if validation_result.is_valid:
                final_sql = construction_result.sql_query
            elif validation_result.corrected_sql:
                final_sql = validation_result.corrected_sql
            else:
                final_sql = construction_result.sql_query  # Use original even if invalid
            
            # Create response
            execution_time_ms = (time.time() - start_time) * 1000
            
            return SQLGenerationResponse(
                sql_query=final_sql,
                explanation=construction_result.explanation,
                confidence=validation_result.confidence_score,
                validation_passed=validation_result.is_valid,
                execution_time_ms=execution_time_ms
            )
            
        except Exception as e:
            # Error handling
            execution_time_ms = (time.time() - start_time) * 1000
            
            return SQLGenerationResponse(
                sql_query=f"/* Error: {str(e)} */",
                explanation=f"SQL generation failed: {str(e)}",
                confidence=0.0,
                validation_passed=False,
                execution_time_ms=execution_time_ms
            )


def create_sql_controller() -> SQLController:
    """Create a SQL Controller with default configuration."""
    return SQLController()