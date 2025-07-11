from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from pydantic import BaseModel


class Event(BaseModel, ABC):
    @abstractmethod
    def to_event_string(self) -> str:
        pass

    @abstractmethod
    def to_message(self) -> str:
        pass

    @abstractmethod
    def to_markdown(self) -> str:
        pass

    def __str__(self):
        return f"q_id: {self.q_id}"


class EndOfEvent(Event):
    q_id: str
    response: str = "end"

    def to_event_string(self) -> str:
        return f"event: {self.to_message()}"

    def to_message(self) -> str:
        return f"{self.response}"

    def to_markdown(self) -> str:
        return f"## End of Event\n\n{self.response}"


class Evaluation(Event):
    question: str
    response: str
    q_id: str
    approved: bool
    summary: str
    issues: Optional[List[str]] = None
    plausibility: Optional[str] = None
    factual_consistency: Optional[str] = None
    clarity: Optional[str] = None
    completeness: Optional[str] = None

    def to_event_string(self) -> str:
        return f"data: {self.to_markdown()}"

    def to_message(self) -> str:
        return f"Question: {self.question}\nResponse: {self.response}\nSummary: {self.summary}\nIssues: {self.issues}\nPlausibility: {self.plausibility}\nFactual Consistency: {self.factual_consistency}\nClarity: {self.clarity}\nCompleteness: {self.completeness}"

    def to_markdown(self) -> str:
        markdown = f"## Evaluation\n\n{self.summary}\n\n"

        if self.issues:
            markdown += "**Issues:**\n"
            if isinstance(self.issues, list):
                for issue in self.issues:
                    markdown += f"- {issue}\n"
            else:
                markdown += f"{self.issues}\n"
            markdown += "\n"

        if self.plausibility:
            markdown += f"**Plausibility:** {self.plausibility}\n\n"
        if self.factual_consistency:
            markdown += f"**Factual Consistency:** {self.factual_consistency}\n\n"
        if self.clarity:
            markdown += f"**Clarity:** {self.clarity}\n\n"
        if self.completeness:
            markdown += f"**Completeness:** {self.completeness}\n\n"

        return markdown.strip()


class FailedRequest(Event):
    question: str
    exception: str
    q_id: str

    def to_event_string(self) -> str:
        return f"data: {self.to_markdown()}"

    def to_message(self) -> str:
        return f"\nQuestion:\n{self.question}\nException:\n{self.exception}"

    def to_markdown(self) -> str:
        return f"## Failed Request\n\n```\n{self.exception}\n```"


class RejectedRequest(Event):
    question: str
    response: str
    q_id: str

    def to_event_string(self) -> str:
        return f"data: {self.to_markdown()}"

    def to_message(self) -> str:
        return (
            f"\nQuestion:\n{self.question}\n was rejected. Response:\n{self.response}"
        )

    def to_markdown(self) -> str:
        return f"## Rejected Request\n\n{self.response}"


class RejectedAnswer(Event):
    question: str
    response: str
    rejection: str
    q_id: str

    def to_event_string(self) -> str:
        return f"data: {self.to_markdown()}"

    def to_message(self) -> str:
        return f"Question:\n{self.question}\nResponse:\n{self.response}\nRejection Reason:\n{self.rejection}"

    def to_markdown(self) -> str:
        return f"## Rejected Answer\n\n{self.response}\n\n### Rejection Reason\n\n{self.rejection}"


class StatusUpdate(Event):
    step_name: str
    q_id: str

    def to_event_string(self) -> str:
        return f"event: {self.to_message()}"

    def to_message(self) -> str:
        return f"{self.step_name}"

    def to_markdown(self) -> str:
        return f"## Status Update\n\n{self.step_name}"


class Response(Event):
    question: str
    response: str
    q_id: str
    data: Optional[Dict[str, str]] = None

    def to_event_string(self) -> str:
        return f"data: {self.to_markdown()}\n\n"

    def to_message(self) -> str:
        message = f"\nQuestion:\n{self.question}\nResponse:\n{self.response}"

        if self.data:
            for key, value in self.data.items():
                message += f"\n{key.capitalize()}:\n{value}"

        return message

    def to_markdown(self) -> str:
        message = f"## Response\n\n{self.response}"

        if self.data:
            for key, value in self.data.items():
                message += f"$%$%{key.capitalize()}:{value}"
        return message


class EvaluationStarted(Event):
    """Event when an evaluation run starts."""

    run_id: str
    run_type: str
    evaluation_category: Optional[str] = None
    stage: Optional[str] = None
    model_name: Optional[str] = None

    def to_event_string(self) -> str:
        return f"event: {self.to_message()}"

    def to_message(self) -> str:
        stage_info = f" - {self.stage}" if self.stage else ""
        return f"Evaluation started: {self.run_type}{stage_info} (run_id: {self.run_id[:8]})"

    def to_markdown(self) -> str:
        return f"## Evaluation Started\n\nType: {self.run_type}\nRun ID: {self.run_id}"


class TestResultRecorded(Event):
    """Event when a test result is recorded."""

    run_id: str
    test_name: str
    passed: bool
    q_id: Optional[str] = None

    def to_event_string(self) -> str:
        return f"event: {self.to_message()}"

    def to_message(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return f"Test {self.test_name}: {status}"

    def to_markdown(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        return f"## Test Result\n\n{self.test_name}: {status}"


class EvaluationCompleted(Event):
    """Event when an evaluation run completes."""

    run_id: str
    run_type: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_rate: float
    q_id: Optional[str] = None

    def to_event_string(self) -> str:
        return f"data: {self.to_markdown()}"

    def to_message(self) -> str:
        return (
            f"Evaluation completed: {self.run_type}\n"
            f"Total: {self.total_tests}, Passed: {self.passed_tests}, "
            f"Failed: {self.failed_tests}, Pass Rate: {self.pass_rate:.1f}%"
        )

    def to_markdown(self) -> str:
        return (
            f"## Evaluation Completed\n\n"
            f"**Type:** {self.run_type}\n"
            f"**Total Tests:** {self.total_tests}\n"
            f"**Passed:** {self.passed_tests}\n"
            f"**Failed:** {self.failed_tests}\n"
            f"**Pass Rate:** {self.pass_rate:.1f}%"
        )
