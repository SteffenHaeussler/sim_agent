"""LLM Judge for evaluating test responses."""

from dataclasses import dataclass
from typing import Dict, Optional

from pydantic import BaseModel, Field

from src.agent.adapters.llm import LLM
from src.agent.config import get_llm_config


class JudgeScores(BaseModel):
    """Individual scoring dimensions for LLM judge evaluation."""

    accuracy: float = Field(ge=0, le=10, description="Factual correctness score")
    relevance: float = Field(ge=0, le=10, description="Relevance to the question")
    completeness: float = Field(ge=0, le=10, description="Completeness of answer")
    hallucination: float = Field(
        ge=0, le=10, description="Absence of hallucination (10=no hallucination)"
    )
    format_compliance: Optional[float] = Field(
        default=None,
        ge=0,
        le=10,
        description="Format compliance for structured outputs",
    )

    @property
    def average_score(self) -> float:
        """Calculate average score across all dimensions."""
        scores = [self.accuracy, self.relevance, self.completeness, self.hallucination]
        if self.format_compliance is not None:
            scores.append(self.format_compliance)
        return sum(scores) / len(scores)


class JudgeResult(BaseModel):
    """Complete result from LLM judge evaluation."""

    scores: JudgeScores
    reasoning: Dict[str, str] = Field(description="Reasoning for each score dimension")
    overall_assessment: str = Field(description="Overall assessment of the response")
    passed: bool = Field(description="Whether the response passes based on thresholds")


class JudgeCriteria(BaseModel):
    """Criteria and thresholds for judge evaluation."""

    accuracy_threshold: float = Field(default=7.0, ge=0, le=10)
    relevance_threshold: float = Field(default=8.0, ge=0, le=10)
    completeness_threshold: float = Field(default=7.0, ge=0, le=10)
    hallucination_threshold: float = Field(default=8.0, ge=0, le=10)
    format_compliance_threshold: Optional[float] = Field(default=None, ge=0, le=10)
    require_exact_match: bool = Field(default=False)
    allow_additional_context: bool = Field(default=True)


@dataclass
class LLMJudge:
    """LLM-based judge for evaluating test responses."""

    llm: LLM = None

    def __post_init__(self):
        """Initialize LLM if not provided."""
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
        """
        Evaluate the quality of an actual response compared to expected.

        Args:
            question: The original question
            expected: The expected response
            actual: The actual response from the system
            criteria: Evaluation criteria and thresholds
            test_type: Type of test for context-specific evaluation

        Returns:
            JudgeResult with scores, reasoning, and pass/fail determination
        """
        if criteria is None:
            criteria = JudgeCriteria()

        prompt = self._build_evaluation_prompt(
            question, expected, actual, criteria, test_type
        )

        # Use LLM to evaluate
        judge_response = self.llm.use(prompt, response_model=JudgeResult)

        # Determine if response passes based on thresholds
        judge_response.passed = self._check_thresholds(judge_response.scores, criteria)

        return judge_response

    def _build_evaluation_prompt(
        self,
        question: str,
        expected: str,
        actual: str,
        criteria: JudgeCriteria,
        test_type: str,
    ) -> str:
        """Build the evaluation prompt for the LLM judge."""

        context_instructions = self._get_test_type_instructions(test_type)

        prompt = f"""You are an expert evaluator assessing the quality of AI-generated responses.

{context_instructions}

Evaluate the actual response compared to the expected response for the given question.

Question: {question}

Expected Response: {expected}

Actual Response: {actual}

Evaluation Criteria:
- Exact Match Required: {criteria.require_exact_match}
- Additional Context Allowed: {criteria.allow_additional_context}

Please evaluate on the following dimensions:

1. **Accuracy (0-10)**: How factually correct is the actual response compared to expected?
   - Consider if key facts, numbers, and information align
   - For numerical values, allow reasonable rounding unless exact match is required

2. **Relevance (0-10)**: How well does the actual response address the question?
   - Does it stay on topic?
   - Does it answer what was asked?

3. **Completeness (0-10)**: Does the actual response fully answer the question?
   - Are all parts of the question addressed?
   - Is critical information missing?

4. **Hallucination (0-10)**: Rate the absence of made-up information
   - 10 = No hallucination at all
   - 0 = Severe hallucination with fabricated facts
   - Consider if actual response contains information not derivable from context

{self._get_format_compliance_instruction(test_type)}

For each dimension, provide:
- A numerical score (0-10)
- Clear reasoning explaining the score

Finally, provide an overall assessment summarizing the response quality.
"""
        return prompt

    def _get_test_type_instructions(self, test_type: str) -> str:
        """Get specific instructions based on test type."""

        instructions = {
            "e2e": "This is an end-to-end test. Focus on whether the final answer is correct and useful to the user.",
            "tool_agent": "This tests tool usage. Verify the correct tools were used and data was properly retrieved/processed.",
            "ir": "This tests information retrieval. Check if the correct documents/data were retrieved and ranked appropriately.",
            "enhance": "This tests question enhancement. Verify the enhanced question maintains the original intent while adding clarity.",
            "pre_check": "This tests pre-processing guardrails. You are evaluating whether the guardrail correctly identified inappropriate requests. Focus on whether the guardrail decision (Approved: True/False) matches the expected decision, not on answering the original question.",
            "post_check": "This tests post-processing guardrails. You are evaluating whether the guardrail correctly approved or rejected a response, NOT whether the response answers the original question. Focus on whether the guardrail decision (Approved: True/False) matches the expected decision.",
            "sql_e2e": """This tests SQL query generation. You are evaluating SQL queries, not natural language responses.
Focus on:
- SQL Correctness: Does the generated SQL query correctly answer the natural language question?
- SQL Syntax: Is the SQL syntactically valid?
- Table/Column Usage: Are the correct tables and columns used?
- Join Logic: Are joins correctly specified?
- Filter Conditions: Are WHERE clauses accurate?
- Aggregations: Are GROUP BY and aggregate functions used correctly?
- Order and Limits: Are ORDER BY and LIMIT clauses appropriate?
Compare the actual SQL to the expected SQL for exact match as specified in criteria.""",
            "sql_grounding": "This tests SQL grounding - identifying relevant tables and columns for a query. Verify the correct database objects were identified.",
            "sql_filter": "This tests SQL filter extraction - identifying WHERE clause conditions from the question.",
            "sql_aggregation": "This tests SQL aggregation detection - identifying GROUP BY needs and aggregate functions.",
            "sql_join": "This tests SQL join inference - determining which tables need to be joined and how.",
        }

        return instructions.get(
            test_type, "Evaluate the response quality comprehensively."
        )

    def _get_format_compliance_instruction(self, test_type: str) -> str:
        """Get format compliance instruction if needed for test type."""

        format_tests = [
            "tool_agent",
            "ir",
            "sql_grounding",
            "sql_filter",
            "sql_aggregation",
            "sql_join",
        ]

        if test_type in format_tests:
            return """
5. **Format Compliance (0-10)**: For structured outputs, does the format match expectations?
   - JSON structure correctness
   - Required fields present
   - Data types match
"""
        elif test_type == "sql_e2e":
            return """
5. **SQL Efficiency (0-10)**: Is the SQL query efficient and well-structured?
   - Avoids unnecessary complexity
   - Uses appropriate indexes (implied by column choices)
   - Minimizes redundant operations
   - Follows SQL best practices
"""
        return ""

    def _check_thresholds(self, scores: JudgeScores, criteria: JudgeCriteria) -> bool:
        """Check if scores meet the threshold criteria."""

        checks = [
            scores.accuracy >= criteria.accuracy_threshold,
            scores.relevance >= criteria.relevance_threshold,
            scores.completeness >= criteria.completeness_threshold,
            scores.hallucination >= criteria.hallucination_threshold,
        ]

        if (
            criteria.format_compliance_threshold is not None
            and scores.format_compliance is not None
        ):
            checks.append(
                scores.format_compliance >= criteria.format_compliance_threshold
            )

        return all(checks)


class BatchJudgeResult(BaseModel):
    """Results from batch evaluation of multiple test cases."""

    total_tests: int
    passed_tests: int
    failed_tests: int
    average_scores: JudgeScores
    test_results: Dict[str, JudgeResult]

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate percentage."""
        return (
            (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        )
