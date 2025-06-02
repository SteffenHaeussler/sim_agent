from typing import Any

import numpy as np

from src.agent.adapters.tools.base import BaseTool


# overwrites default smolagents tool
class FinalAnswerTool(BaseTool):
    name = "final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {
        "answer": {"type": "any", "description": "The final answer to the problem"}
    }
    outputs = {
        "result": {"type": "any", "description": "The final response to the problem"}
    }
    output_type = "any"

    def forward(self, answer: Any) -> Any:
        """
        Provides a final answer to the given problem.

        Args:
            answer: Any: The final answer to the problem.

        Returns:
            answer: Any: The final answer to the problem.
        """
        if isinstance(answer, (int, float, bool, np.int64, np.float64)):
            answer = str(round(answer, 6))

        return answer
