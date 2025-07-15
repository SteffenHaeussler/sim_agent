"""Domain models for evaluation tracking following DDD principles."""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class JudgeScores(BaseModel):
    """Value object for judge evaluation scores."""

    accuracy: float = Field(ge=0, le=10)
    relevance: float = Field(ge=0, le=10)
    completeness: float = Field(ge=0, le=10)
    hallucination: float = Field(ge=0, le=10)
    format_compliance: Optional[float] = Field(default=None, ge=0, le=10)

    @property
    def average_score(self) -> float:
        """Calculate average score across all dimensions."""
        scores = [self.accuracy, self.relevance, self.completeness, self.hallucination]
        if self.format_compliance is not None:
            scores.append(self.format_compliance)
        return sum(scores) / len(scores)


class EvaluationMetric(BaseModel):
    """Value object for aggregated evaluation metrics."""

    metric_type: str  # 'accuracy', 'relevance', 'completeness', 'hallucination'
    average_score: float
    min_score: float
    max_score: float
    std_deviation: float
    percentile_25: Optional[float] = None
    percentile_50: Optional[float] = None
    percentile_75: Optional[float] = None


class TestResult(BaseModel):
    """Entity representing an individual test result."""

    id: UUID = Field(default_factory=uuid4)
    test_name: str
    test_type: Optional[str] = None
    question: str
    expected_response: str
    actual_response: str
    passed: bool
    execution_time_ms: Optional[int] = None
    judge_scores: Optional[JudgeScores] = None
    judge_reasoning: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Additional fields for specific test types
    tools_used: Optional[List[str]] = None  # For tool agent tests
    tool_outputs: Optional[Dict[str, Any]] = None
    sql_stage: Optional[str] = None  # For SQL tests
    sql_query: Optional[str] = None
    schema_context: Optional[Dict[str, Any]] = None

    def is_judge_evaluated(self) -> bool:
        """Check if this result has been evaluated by the judge."""
        return self.judge_scores is not None


class EvaluationRun(BaseModel):
    """Aggregate root for an evaluation run."""

    id: UUID = Field(default_factory=uuid4)
    run_type: str  # 'e2e', 'sql_stages', 'tool_agent', etc.
    evaluation_category: Optional[str] = None  # 'sql', 'tool_agent', 'guardrails'
    stage: Optional[str] = None  # For SQL: 'grounding', 'filter', etc.
    git_commit: Optional[str] = None
    branch: Optional[str] = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    model_name: Optional[str] = None
    model_temperature: Optional[float] = None
    prompt_version: Optional[str] = None
    fixtures_used: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None

    # Relationships
    test_results: List[TestResult] = Field(default_factory=list)
    metrics: List[EvaluationMetric] = Field(default_factory=list)

    def add_test_result(self, result: TestResult) -> None:
        """Add a test result to this run."""
        self.test_results.append(result)
        self.total_tests += 1
        if result.passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1

    def complete(self) -> None:
        """Mark the evaluation run as complete."""
        self.completed_at = datetime.now(timezone.utc)
        self._calculate_metrics()

    def _calculate_metrics(self) -> None:
        """Calculate aggregate metrics for this run."""
        if not self.test_results:
            return

        # Only calculate for judge-evaluated results
        judge_results = [r for r in self.test_results if r.is_judge_evaluated()]
        if not judge_results:
            return

        # Calculate metrics for each score dimension
        score_types = ["accuracy", "relevance", "completeness", "hallucination"]

        for score_type in score_types:
            scores = [getattr(r.judge_scores, score_type) for r in judge_results]

            if scores:
                import statistics

                metric = EvaluationMetric(
                    metric_type=score_type,
                    average_score=statistics.mean(scores),
                    min_score=min(scores),
                    max_score=max(scores),
                    std_deviation=statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    percentile_25=statistics.quantiles(scores, n=4)[0]
                    if len(scores) > 1
                    else scores[0],
                    percentile_50=statistics.median(scores),
                    percentile_75=statistics.quantiles(scores, n=4)[2]
                    if len(scores) > 1
                    else scores[0],
                )
                self.metrics.append(metric)

    @property
    def pass_rate(self) -> float:
        """Calculate the pass rate for this run."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate the duration of the run in seconds."""
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to a summary dictionary for reporting."""
        summary = {
            "run_id": str(self.id),
            "run_type": self.run_type,
            "evaluation_category": self.evaluation_category,
            "stage": self.stage,
            "git_commit": self.git_commit,
            "branch": self.branch,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "duration_seconds": self.duration_seconds,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "pass_rate": f"{self.pass_rate:.1f}%",
            "model_name": self.model_name,
            "model_temperature": self.model_temperature,
        }

        # Add average scores if available
        if self.metrics:
            summary["average_scores"] = {
                metric.metric_type: round(metric.average_score, 2)
                for metric in self.metrics
            }

        # Add failed test details
        failed_tests = [r for r in self.test_results if not r.passed]
        if failed_tests:
            summary["failed_tests_details"] = [
                {
                    "test_name": test.test_name,
                    "error": test.error_message,
                    "scores": test.judge_scores.model_dump()
                    if test.judge_scores
                    else None,
                }
                for test in failed_tests
            ]

        return summary


class EvaluationComparison(BaseModel):
    """Value object for comparing evaluation runs."""

    baseline_run_id: UUID
    comparison_run_id: UUID
    metric_improvements: Dict[str, float]  # Metric type -> improvement percentage
    pass_rate_change: float
    execution_time_change: Optional[float] = None
    summary: str

    def is_improvement(self) -> bool:
        """Check if the comparison shows overall improvement."""
        # Consider it an improvement if pass rate increased or average scores improved
        if self.pass_rate_change > 0:
            return True

        avg_improvement = sum(self.metric_improvements.values()) / len(
            self.metric_improvements
        )
        return avg_improvement > 0
