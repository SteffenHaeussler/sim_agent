"""Base classes and utilities for SQL evaluation tests."""

from pathlib import Path
from typing import List

from src.agent.domain import commands
from tests.evals.base_eval_db import BaseEvaluationTest


class BaseSQLEvalTest(BaseEvaluationTest):
    """Base class for SQL evaluation tests with common functionality."""

    # SQL-specific configuration
    EVALUATION_CATEGORY = "sql"

    def setup_method(self):
        """Setup method called before each test method."""
        super().setup_method()
        self.current_path = Path(__file__).parent

    @staticmethod
    def create_grounding_result(
        grounding_tables: List[str], grounding_columns: List[str]
    ) -> commands.GroundingResponse:
        """Create a GroundingResponse from table and column lists.

        Args:
            grounding_tables: List of table names
            grounding_columns: List of column names (can be "table.column" format)

        Returns:
            GroundingResponse with table and column mappings
        """
        # Create table mappings
        table_mappings = [
            commands.TableMapping(question_term=table, table_name=table, confidence=0.9)
            for table in grounding_tables
        ]

        # Create column mappings
        column_mappings = []
        for col in grounding_columns:
            parts = col.split(".", 1)
            table_name = (
                parts[0]
                if len(parts) > 1
                else grounding_tables[0]
                if grounding_tables
                else "unknown"
            )
            column_name = parts[1] if len(parts) > 1 else col

            column_mappings.append(
                commands.ColumnMapping(
                    question_term=column_name,
                    table_name=table_name,
                    column_name=column_name,
                    confidence=0.9,
                )
            )

        return commands.GroundingResponse(
            table_mapping=table_mappings,
            column_mapping=column_mappings,
        )


# Command handler mapping for E2E tests
SQL_COMMAND_HANDLERS = {
    commands.SQLCheck: "check",
    commands.SQLGrounding: "grounding",
    commands.SQLFilter: "filter",
    commands.SQLJoinInference: "join_inference",
    commands.SQLAggregation: "aggregation",
    commands.SQLConstruction: "construction",
}
