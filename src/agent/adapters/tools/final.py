import base64
import re
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

        if self.is_base64(answer):
            answer = {"plot": answer}

        if isinstance(answer, (int, float, bool, np.int64, np.float64)):
            answer = str(round(answer, 6))

        return answer

    def is_base64(self, s: str) -> bool:
        # Check if length is multiple of 4
        if not isinstance(s, str) or len(s) % 4 != 0 or len(s) < 50:
            return False

        # Check if it matches base64 regex
        if not re.fullmatch(r"^[A-Za-z0-9+/]*={0,2}$", s):
            return False

        try:
            # Try decoding and re-encoding to confirm match
            return base64.b64encode(base64.b64decode(s)).decode() == s
        except Exception:
            return False
