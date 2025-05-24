import os
from time import time
from typing import Optional
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from loguru import logger

import src.agent.service_layer.handlers as handlers
from src.agent.adapters.adapter import AgentAdapter
from src.agent.adapters.notifications import ApiNotifications, SlackNotifications
from src.agent.bootstrap import bootstrap
from src.agent.config import get_logging_config, get_tracing_config
from src.agent.domain.commands import Question
from src.agent.observability.context import ctx_query_id
from src.agent.observability.logging import setup_logging
from src.agent.observability.tracing import setup_tracing

if os.getenv("IS_TESTING") != "true":
    load_dotenv(".env")

setup_tracing(get_tracing_config())
setup_logging(get_logging_config())


app = FastAPI()

bus = bootstrap(
    adapter=AgentAdapter(),
    notifications=[ApiNotifications(), SlackNotifications()],
)


@app.get("/answer")
def answer(question: str, q_id: Optional[str] = None):
    """
    Entrypoint for the agent.

    Args:
        question: str: The question to answer.
        q_id: Optional[str]: The id of the question.

    Returns:
        response: str: The response to the question.

    Raises:
        HTTPException: If the question is invalid.
        ValueError: If the question is invalid.
    """
    if not q_id:
        q_id = uuid4().hex

    ctx_query_id.set(q_id)
    try:
        command = Question(question, q_id)
        bus.handle(command)

    except (handlers.InvalidQuestion, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    message = ApiNotifications().temp.pop(q_id, None)
    return {"response": message}


@app.get("/health")
def health(request: Request):
    logger.debug(f"Methode: {request.method} on {request.url.path}")
    return {"version": "0.0.1", "timestamp": time()}
