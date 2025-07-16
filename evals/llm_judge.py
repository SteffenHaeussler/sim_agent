"""Simplified LLM Judge for evaluating test responses."""

from dataclasses import dataclass
from typing import Dict, Optional
from pydantic import BaseModel, Field

from src.agent.adapters.llm import LLM
from src.agent.config import get_llm_config


class JudgeScores(BaseModel):
    """Scoring dimensions for evaluation."""

    accuracy: float = Field(ge=0, le=10, description="Factual correctness")
    relevance: float = Field(ge=0, le=10, description="Relevance to question")
    completeness: float = Field(ge=0, le=10, description="Completeness of answer")
    hallucination: float = Field(
        ge=0, le=10, description="Absence of hallucination (10=none)"
    )


class JudgeResult(BaseModel):
    """Result from LLM judge evaluation."""

    scores: JudgeScores
    reasoning: Dict[str, str] = Field(description="Reasoning for each score")
    overall_assessment: str = Field(description="Overall assessment")
    passed: bool = Field(description="Whether the response passes")


class JudgeCriteria(BaseModel):
    """Criteria for evaluation."""

    accuracy_threshold: float = Field(default=7.0, ge=0, le=10)
    relevance_threshold: float = Field(default=8.0, ge=0, le=10)
    completeness_threshold: float = Field(default=7.0, ge=0, le=10)
    hallucination_threshold: float = Field(default=8.0, ge=0, le=10)


@dataclass
class LLMJudge:
    """Simplified LLM-based judge."""

    llm: LLM = None

    def __post_init__(self):
        if self.llm is None:
            self.llm = LLM(get_llm_config())

    def evaluate(
        self,
        question: str,
        expected: str,
        actual: str,
        criteria: Optional[JudgeCriteria] = None,
        test_type: str = "general",
    ) -> JudgeResult:
        """Evaluate actual response compared to expected."""
        if criteria is None:
            criteria = JudgeCriteria()

        prompt = f"""You are evaluating an AI response.

Question: {question}
Expected Response: {expected}
Actual Response: {actual}

Evaluate on these dimensions (0-10):
1. Accuracy: How factually correct is the actual response?
2. Relevance: Does it answer the question?
3. Completeness: Is the answer complete?
4. Hallucination: Rate absence of made-up info (10=no hallucination)

Provide scores and brief reasoning for each dimension, then an overall assessment.
"""

        # Use LLM to evaluate
        judge_response = self.llm.use(prompt, response_model=JudgeResult)

        # Check if passes thresholds
        judge_response.passed = all(
            [
                judge_response.scores.accuracy >= criteria.accuracy_threshold,
                judge_response.scores.relevance >= criteria.relevance_threshold,
                judge_response.scores.completeness >= criteria.completeness_threshold,
                judge_response.scores.hallucination >= criteria.hallucination_threshold,
            ]
        )

        return judge_response
