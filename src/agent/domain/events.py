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
        return f"Starting step: {self.step_name}"

    def to_markdown(self) -> str:
        return f"## Status Update\n\n**Starting:**\n\n{self.step_name}"


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
