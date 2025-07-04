"""
Simplified Aggregation Agent - Detects aggregation requirements.
"""

from typing import List

from src.agent.adapters.llm import LLM
from src.agent.adapters.sql_tools.models import AggregationResponse, ColumnMapping
from src.agent.adapters.sql_tools.prompts import AGGREGATION_AGENT_PROMPT


class AggregationAgent:
    """Agent that detects aggregation requirements in natural language questions."""

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

    def process(
        self, question: str, column_mappings: List[ColumnMapping]
    ) -> AggregationResponse:
        """Detect aggregation requirements from question."""
        mapped_columns_str = self._format_mapped_columns(column_mappings)

        prompt = AGGREGATION_AGENT_PROMPT.format(
            question=question, mapped_columns=mapped_columns_str
        )

        try:
            response = self.llm.use(prompt, AggregationResponse)
            return response
        except Exception as e:
            # Fallback response
            return AggregationResponse(
                aggregations=[],
                group_by_columns=[],
                is_aggregation_query=False,
                reasoning=f"Aggregation agent failed: {str(e)}",
            )
