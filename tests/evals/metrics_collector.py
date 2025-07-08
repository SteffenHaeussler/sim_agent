"""Metrics collection and reporting for evaluation tests."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field

from tests.evals.llm_judge import JudgeResult, JudgeScores


class TestMetric(BaseModel):
    """Individual test metric entry."""

    test_id: str
    test_type: str
    timestamp: datetime = Field(default_factory=datetime.now)
    duration_seconds: Optional[float] = None
    passed: bool
    judge_scores: Optional[JudgeScores] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MetricsSummary(BaseModel):
    """Summary metrics for a test run."""

    test_type: str
    run_timestamp: datetime
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_rate: float
    average_duration: Optional[float] = None
    average_scores: Optional[Dict[str, float]] = None
    score_distributions: Optional[Dict[str, Dict[str, float]]] = None
    common_failures: Optional[List[Dict[str, Any]]] = None


class MetricsCollector:
    """Collects and aggregates metrics from evaluation tests."""

    def __init__(self, output_dir: Path = Path("test_metrics")):
        """Initialize metrics collector."""
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.metrics: List[TestMetric] = []
        self.current_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def add_metric(self, metric: TestMetric) -> None:
        """Add a single test metric."""
        self.metrics.append(metric)

    def add_test_result(
        self,
        test_id: str,
        test_type: str,
        passed: bool,
        duration: Optional[float] = None,
        judge_result: Optional[JudgeResult] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a test result to metrics."""

        metric = TestMetric(
            test_id=test_id,
            test_type=test_type,
            passed=passed,
            duration_seconds=duration,
            judge_scores=judge_result.scores if judge_result else None,
            error_message=error,
            metadata=metadata or {},
        )
        self.add_metric(metric)

    def generate_summary(self, test_type: str) -> MetricsSummary:
        """Generate summary metrics for a specific test type."""

        # Filter metrics by test type
        type_metrics = [m for m in self.metrics if m.test_type == test_type]

        if not type_metrics:
            raise ValueError(f"No metrics found for test type: {test_type}")

        # Basic counts
        total_tests = len(type_metrics)
        passed_tests = sum(1 for m in type_metrics if m.passed)
        failed_tests = total_tests - passed_tests

        # Duration stats
        durations = [m.duration_seconds for m in type_metrics if m.duration_seconds]
        avg_duration = sum(durations) / len(durations) if durations else None

        # Judge score stats
        judge_metrics = [m for m in type_metrics if m.judge_scores]
        avg_scores = None
        score_distributions = None

        if judge_metrics:
            # Calculate average scores
            avg_scores = {
                "accuracy": sum(m.judge_scores.accuracy for m in judge_metrics)
                / len(judge_metrics),
                "relevance": sum(m.judge_scores.relevance for m in judge_metrics)
                / len(judge_metrics),
                "completeness": sum(m.judge_scores.completeness for m in judge_metrics)
                / len(judge_metrics),
                "hallucination": sum(
                    m.judge_scores.hallucination for m in judge_metrics
                )
                / len(judge_metrics),
            }

            # Add format compliance if present
            format_scores = [
                m.judge_scores.format_compliance
                for m in judge_metrics
                if m.judge_scores.format_compliance is not None
            ]
            if format_scores:
                avg_scores["format_compliance"] = sum(format_scores) / len(
                    format_scores
                )

            # Calculate score distributions (percentiles)
            score_data = {
                "accuracy": [m.judge_scores.accuracy for m in judge_metrics],
                "relevance": [m.judge_scores.relevance for m in judge_metrics],
                "completeness": [m.judge_scores.completeness for m in judge_metrics],
                "hallucination": [m.judge_scores.hallucination for m in judge_metrics],
            }

            score_distributions = {}
            for metric_name, scores in score_data.items():
                if scores:
                    score_distributions[metric_name] = {
                        "min": min(scores),
                        "p25": pd.Series(scores).quantile(0.25),
                        "median": pd.Series(scores).quantile(0.50),
                        "p75": pd.Series(scores).quantile(0.75),
                        "max": max(scores),
                        "std": pd.Series(scores).std(),
                    }

        # Analyze common failures
        failed_metrics = [m for m in type_metrics if not m.passed]
        common_failures = []

        if failed_metrics:
            # Group failures by error pattern
            error_counts: Dict[str, int] = {}
            for m in failed_metrics:
                error_key = m.error_message or "unknown_error"
                # Simplify error message for grouping
                error_key = error_key.split("\n")[0][:100]
                error_counts[error_key] = error_counts.get(error_key, 0) + 1

            # Sort by frequency
            common_failures = [
                {
                    "error": error,
                    "count": count,
                    "percentage": count / len(failed_metrics) * 100,
                }
                for error, count in sorted(
                    error_counts.items(), key=lambda x: x[1], reverse=True
                )
            ][:5]  # Top 5 failure reasons

        return MetricsSummary(
            test_type=test_type,
            run_timestamp=datetime.now(),
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            pass_rate=passed_tests / total_tests,
            average_duration=avg_duration,
            average_scores=avg_scores,
            score_distributions=score_distributions,
            common_failures=common_failures,
        )

    def save_metrics(self) -> Path:
        """Save all metrics to JSON file."""

        output_file = self.output_dir / f"metrics_{self.current_run_id}.json"

        metrics_data = {
            "run_id": self.current_run_id,
            "timestamp": datetime.now().isoformat(),
            "total_metrics": len(self.metrics),
            "metrics": [m.model_dump() for m in self.metrics],
        }

        with open(output_file, "w") as f:
            json.dump(metrics_data, f, indent=2, default=str)

        return output_file

    def save_summary(self, summary: MetricsSummary) -> Path:
        """Save summary metrics to JSON file."""

        output_file = (
            self.output_dir / f"summary_{summary.test_type}_{self.current_run_id}.json"
        )

        with open(output_file, "w") as f:
            json.dump(summary.model_dump(), f, indent=2, default=str)

        return output_file

    def generate_report(self) -> Path:
        """Generate comprehensive HTML report."""

        # Group metrics by test type
        test_types = list(set(m.test_type for m in self.metrics))
        summaries = []

        for test_type in test_types:
            try:
                summary = self.generate_summary(test_type)
                summaries.append(summary)
            except ValueError:
                continue

        # Generate HTML report
        html_content = self._generate_html_report(summaries)

        output_file = self.output_dir / f"report_{self.current_run_id}.html"
        with open(output_file, "w") as f:
            f.write(html_content)

        return output_file

    def _generate_html_report(self, summaries: List[MetricsSummary]) -> str:
        """Generate HTML report content."""

        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Evaluation Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .pass { color: green; font-weight: bold; }
        .fail { color: red; font-weight: bold; }
        .metric-card {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            background-color: #f9f9f9;
        }
        .score-bar {
            background-color: #e0e0e0;
            border-radius: 3px;
            height: 20px;
            position: relative;
            margin: 5px 0;
        }
        .score-fill {
            background-color: #4CAF50;
            height: 100%;
            border-radius: 3px;
            text-align: center;
            color: white;
            line-height: 20px;
        }
    </style>
</head>
<body>
    <h1>Evaluation Test Report</h1>
    <p>Generated: {timestamp}</p>
    <p>Total Test Types: {total_types}</p>

    {content}
</body>
</html>
"""

        content_parts = []

        for summary in summaries:
            pass_rate_class = "pass" if summary.pass_rate >= 0.8 else "fail"

            section = f"""
    <div class="metric-card">
        <h2>{summary.test_type.upper()} Tests</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Tests</td>
                <td>{summary.total_tests}</td>
            </tr>
            <tr>
                <td>Passed</td>
                <td class="pass">{summary.passed_tests}</td>
            </tr>
            <tr>
                <td>Failed</td>
                <td class="fail">{summary.failed_tests}</td>
            </tr>
            <tr>
                <td>Pass Rate</td>
                <td class="{pass_rate_class}">{summary.pass_rate:.1%}</td>
            </tr>
"""

            if summary.average_duration:
                section += f"""
            <tr>
                <td>Average Duration</td>
                <td>{summary.average_duration:.2f}s</td>
            </tr>
"""

            section += "</table>"

            # Add score visualization if available
            if summary.average_scores:
                section += "<h3>Average Judge Scores</h3>"
                for metric, score in summary.average_scores.items():
                    percentage = score * 10  # Convert to percentage
                    section += f"""
        <div>
            <strong>{metric.replace("_", " ").title()}:</strong>
            <div class="score-bar">
                <div class="score-fill" style="width: {percentage}%">{score:.1f}/10</div>
            </div>
        </div>
"""

            section += "</div>"
            content_parts.append(section)

        return html.format(
            timestamp=datetime.now().isoformat(),
            total_types=len(summaries),
            content="\n".join(content_parts),
        )

    def export_to_csv(self) -> Path:
        """Export metrics to CSV for further analysis."""

        if not self.metrics:
            raise ValueError("No metrics to export")

        # Convert to dataframe
        data = []
        for m in self.metrics:
            row = {
                "test_id": m.test_id,
                "test_type": m.test_type,
                "timestamp": m.timestamp,
                "duration_seconds": m.duration_seconds,
                "passed": m.passed,
                "error_message": m.error_message,
            }

            # Add judge scores if available
            if m.judge_scores:
                row.update(
                    {
                        "score_accuracy": m.judge_scores.accuracy,
                        "score_relevance": m.judge_scores.relevance,
                        "score_completeness": m.judge_scores.completeness,
                        "score_hallucination": m.judge_scores.hallucination,
                        "score_format": m.judge_scores.format_compliance,
                    }
                )

            data.append(row)

        df = pd.DataFrame(data)
        output_file = self.output_dir / f"metrics_{self.current_run_id}.csv"
        df.to_csv(output_file, index=False)

        return output_file
