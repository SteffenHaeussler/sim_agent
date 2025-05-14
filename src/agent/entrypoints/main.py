import argparse
from uuid import uuid4

import src.agent.service_layer.handlers as handlers
from src.agent.adapters.adapter import AgentAdapter
from src.agent.adapters.notifications import CliNotifications
from src.agent.bootstrap import bootstrap
from src.agent.domain.commands import Question

bus = bootstrap(adapter=AgentAdapter(), notifications=(CliNotifications()))


def answer(question: str, q_id: str) -> str:
    try:
        command = Question(question, q_id)
        bus.handle(command)
    except (handlers.InvalidQuestion, ValueError) as e:
        raise Exception(str(e))
    return "done"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get question.")
    parser.add_argument("question", nargs="?", type=str, help="question")
    parser.add_argument("--q", type=str, help="question")

    args = parser.parse_args()

    question = args.q if args.q else args.question

    if question and question.startswith("question="):
        question = question.removeprefix("question=")

    q_id = uuid4().hex

    answer(question, q_id)
