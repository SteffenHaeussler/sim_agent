from typing import Any

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
        return answer
