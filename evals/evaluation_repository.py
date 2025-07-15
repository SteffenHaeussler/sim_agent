"""Repository adapter for evaluation persistence following DDD patterns."""

from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
from uuid import UUID
import os
import subprocess

from sqlalchemy import create_engine, desc, and_
from sqlalchemy.orm import Session, sessionmaker
from loguru import logger

from src.agent.adapters.orm import (
    Base,
    EvaluationRunORM,
    TestResultORM,
    EvaluationMetricORM,
    ToolAgentResultORM,
    SQLTestResultORM,
)
from src.agent.domain.evaluation_model import (
    EvaluationRun,
    TestResult,
    EvaluationMetric,
    JudgeScores,
)


class EvaluationRepository:
    """Repository for managing evaluation data persistence."""

    def __init__(self, connection_string: Optional[str] = None):
        """Initialize the repository with database connection."""
        if connection_string is None:
            # Build connection string from environment variables
            pg_host = os.getenv("PG_HOST", "localhost")
            pg_port = os.getenv("PG_PORT", "5432")
            pg_user = os.getenv("PG_USER", "postgres")
            pg_password = os.getenv("PG_PASSWORD", "example")
            pg_eval_db = os.getenv("PG_EVAL_DB", "evaluation")
            connection_string = (
                f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_eval_db}"
            )

        self.engine = create_engine(connection_string)
        Base.metadata.bind = self.engine
        self.SessionLocal = sessionmaker(bind=self.engine)

    def _get_git_info(self) -> Dict[str, str]:
        """Get current git commit and branch information."""
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()

            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
            ).strip()

            return {"commit": commit[:8], "branch": branch}
        except Exception as e:
            logger.warning(f"Could not get git info: {e}")
            return {"commit": None, "branch": None}

    def create_evaluation_run(
        self,
        run_type: str,
        evaluation_category: Optional[str] = None,
        stage: Optional[str] = None,
        model_name: Optional[str] = None,
        model_temperature: Optional[float] = None,
        prompt_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationRun:
        """Create a new evaluation run."""
        git_info = self._get_git_info()

        evaluation_run = EvaluationRun(
            run_type=run_type,
            evaluation_category=evaluation_category,
            stage=stage,
            git_commit=git_info["commit"],
            branch=git_info["branch"],
            model_name=model_name,
            model_temperature=model_temperature,
            prompt_version=prompt_version,
            metadata=metadata or {},
        )

        with self.SessionLocal() as session:
            orm_run = EvaluationRunORM(
                id=evaluation_run.id,
                run_type=evaluation_run.run_type,
                evaluation_category=evaluation_run.evaluation_category,
                stage=evaluation_run.stage,
                git_commit=evaluation_run.git_commit,
                branch=evaluation_run.branch,
                started_at=evaluation_run.started_at,
                model_name=evaluation_run.model_name,
                model_temperature=evaluation_run.model_temperature,
                prompt_version=evaluation_run.prompt_version,
                meta_data=evaluation_run.metadata,
            )
            session.add(orm_run)
            session.commit()

        logger.info(f"Created evaluation run {evaluation_run.id} for {run_type}")
        return evaluation_run

    def add_test_result(self, run_id: UUID, test_result: TestResult) -> None:
        """Add a test result to an evaluation run."""
        with self.SessionLocal() as session:
            # Create test result
            orm_result = TestResultORM(
                id=test_result.id,
                run_id=run_id,
                test_name=test_result.test_name,
                test_type=test_result.test_type,
                question=test_result.question,
                expected_response=test_result.expected_response,
                actual_response=test_result.actual_response,
                passed=test_result.passed,
                execution_time_ms=test_result.execution_time_ms,
                judge_scores=test_result.judge_scores.model_dump()
                if test_result.judge_scores
                else None,
                judge_reasoning=test_result.judge_reasoning,
                error_message=test_result.error_message,
                meta_data=test_result.metadata,
            )
            session.add(orm_result)

            # Add tool agent specific data if present
            if test_result.tools_used is not None:
                tool_result = ToolAgentResultORM(
                    test_result_id=test_result.id,
                    tools_used=test_result.tools_used,
                    tool_outputs=test_result.tool_outputs,
                    execution_delay_ms=test_result.metadata.get("execution_delay_ms")
                    if test_result.metadata
                    else None,
                )
                session.add(tool_result)

            # Add SQL specific data if present
            if test_result.sql_stage is not None or test_result.sql_query is not None:
                sql_result = SQLTestResultORM(
                    test_result_id=test_result.id,
                    stage=test_result.sql_stage,
                    schema_context=test_result.schema_context,
                    sql_query=test_result.sql_query,
                )
                session.add(sql_result)

            # Update run statistics
            run = session.query(EvaluationRunORM).filter_by(id=run_id).first()
            if run:
                run.total_tests += 1
                if test_result.passed:
                    run.passed_tests += 1
                else:
                    run.failed_tests += 1

            session.commit()

    def complete_evaluation_run(
        self, run_id: UUID, fixtures_used: Optional[List[str]] = None
    ) -> None:
        """Mark an evaluation run as complete and calculate metrics."""
        with self.SessionLocal() as session:
            run = session.query(EvaluationRunORM).filter_by(id=run_id).first()
            if not run:
                raise ValueError(f"Evaluation run {run_id} not found")

            run.completed_at = datetime.now(timezone.utc)
            if fixtures_used:
                run.fixtures_used = fixtures_used

            # Calculate and store metrics
            self._calculate_and_store_metrics(session, run)

            session.commit()
            logger.info(f"Completed evaluation run {run_id}")

    def _calculate_and_store_metrics(
        self, session: Session, run: EvaluationRunORM
    ) -> None:
        """Calculate aggregate metrics for a run."""
        # Get all test results with judge scores
        results_with_scores = (
            session.query(TestResultORM)
            .filter(
                and_(
                    TestResultORM.run_id == run.id,
                    TestResultORM.judge_scores.isnot(None),
                )
            )
            .all()
        )

        if not results_with_scores:
            return

        # Calculate metrics for each dimension
        dimensions = ["accuracy", "relevance", "completeness", "hallucination"]

        for dimension in dimensions:
            scores = [r.judge_scores.get(dimension, 0) for r in results_with_scores]

            if scores:
                import statistics

                metric = EvaluationMetricORM(
                    run_id=run.id,
                    metric_type=dimension,
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
                session.add(metric)

    def get_evaluation_run(self, run_id: UUID) -> Optional[EvaluationRun]:
        """Retrieve an evaluation run by ID."""
        with self.SessionLocal() as session:
            orm_run = session.query(EvaluationRunORM).filter_by(id=run_id).first()
            if not orm_run:
                return None

            # Convert ORM to domain model
            run = EvaluationRun(
                id=orm_run.id,
                run_type=orm_run.run_type,
                evaluation_category=orm_run.evaluation_category,
                stage=orm_run.stage,
                git_commit=orm_run.git_commit,
                branch=orm_run.branch,
                started_at=orm_run.started_at,
                completed_at=orm_run.completed_at,
                total_tests=orm_run.total_tests,
                passed_tests=orm_run.passed_tests,
                failed_tests=orm_run.failed_tests,
                model_name=orm_run.model_name,
                model_temperature=orm_run.model_temperature,
                prompt_version=orm_run.prompt_version,
                fixtures_used=orm_run.fixtures_used or [],
                metadata=orm_run.meta_data,
            )

            # Load test results
            for orm_result in orm_run.test_results:
                test_result = TestResult(
                    id=orm_result.id,
                    test_name=orm_result.test_name,
                    test_type=orm_result.test_type,
                    question=orm_result.question,
                    expected_response=orm_result.expected_response,
                    actual_response=orm_result.actual_response,
                    passed=orm_result.passed,
                    execution_time_ms=orm_result.execution_time_ms,
                    judge_scores=JudgeScores(**orm_result.judge_scores)
                    if orm_result.judge_scores
                    else None,
                    judge_reasoning=orm_result.judge_reasoning,
                    error_message=orm_result.error_message,
                    metadata=orm_result.meta_data,
                    created_at=orm_result.created_at,
                )
                run.test_results.append(test_result)

            # Load metrics
            for orm_metric in orm_run.metrics:
                metric = EvaluationMetric(
                    metric_type=orm_metric.metric_type,
                    average_score=orm_metric.average_score,
                    min_score=orm_metric.min_score,
                    max_score=orm_metric.max_score,
                    std_deviation=orm_metric.std_deviation,
                    percentile_25=orm_metric.percentile_25,
                    percentile_50=orm_metric.percentile_50,
                    percentile_75=orm_metric.percentile_75,
                )
                run.metrics.append(metric)

            return run

    def get_recent_runs(
        self, run_type: Optional[str] = None, limit: int = 10
    ) -> List[EvaluationRun]:
        """Get recent evaluation runs, optionally filtered by type."""
        with self.SessionLocal() as session:
            query = session.query(EvaluationRunORM)

            if run_type:
                query = query.filter_by(run_type=run_type)

            orm_runs = (
                query.order_by(desc(EvaluationRunORM.started_at)).limit(limit).all()
            )

            runs = []
            for orm_run in orm_runs:
                # Create simplified version without loading all test results
                run = EvaluationRun(
                    id=orm_run.id,
                    run_type=orm_run.run_type,
                    evaluation_category=orm_run.evaluation_category,
                    stage=orm_run.stage,
                    git_commit=orm_run.git_commit,
                    branch=orm_run.branch,
                    started_at=orm_run.started_at,
                    completed_at=orm_run.completed_at,
                    total_tests=orm_run.total_tests,
                    passed_tests=orm_run.passed_tests,
                    failed_tests=orm_run.failed_tests,
                    model_name=orm_run.model_name,
                    model_temperature=orm_run.model_temperature,
                    prompt_version=orm_run.prompt_version,
                    metadata=orm_run.meta_data,
                )
                runs.append(run)

            return runs

    def get_comparison_metrics(
        self, baseline_run_id: UUID, comparison_run_id: UUID
    ) -> Dict[str, Any]:
        """Compare metrics between two evaluation runs."""
        baseline = self.get_evaluation_run(baseline_run_id)
        comparison = self.get_evaluation_run(comparison_run_id)

        if not baseline or not comparison:
            raise ValueError("One or both runs not found")

        result = {
            "baseline": {
                "run_id": str(baseline_run_id),
                "pass_rate": baseline.pass_rate,
                "total_tests": baseline.total_tests,
            },
            "comparison": {
                "run_id": str(comparison_run_id),
                "pass_rate": comparison.pass_rate,
                "total_tests": comparison.total_tests,
            },
            "improvements": {},
        }

        # Compare pass rates
        result["pass_rate_change"] = comparison.pass_rate - baseline.pass_rate

        # Compare metric scores
        baseline_metrics = {m.metric_type: m.average_score for m in baseline.metrics}
        comparison_metrics = {
            m.metric_type: m.average_score for m in comparison.metrics
        }

        for metric_type in baseline_metrics:
            if metric_type in comparison_metrics:
                baseline_score = baseline_metrics[metric_type]
                comparison_score = comparison_metrics[metric_type]
                improvement = (
                    (comparison_score - baseline_score) / baseline_score
                ) * 100
                result["improvements"][metric_type] = round(improvement, 2)

        return result
