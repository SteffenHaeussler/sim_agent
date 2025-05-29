import asyncio
import os
from time import time
from typing import Optional
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from loguru import logger
from starlette.websockets import WebSocketState

import src.agent.service_layer.handlers as handlers
from src.agent.adapters.adapter import AgentAdapter
from src.agent.adapters.notifications import (
    ApiNotifications,
    SlackNotifications,
    WSNotifications,
)
from src.agent.bootstrap import bootstrap
from src.agent.config import get_logging_config, get_tracing_config
from src.agent.domain.commands import Question
from src.agent.observability.context import connected_clients, ctx_query_id
from src.agent.observability.logging import setup_logging
from src.agent.observability.tracing import setup_tracing

if os.getenv("IS_TESTING") != "true":
    load_dotenv(".env")

setup_tracing(get_tracing_config())
setup_logging(get_logging_config())


app = FastAPI()

bus = bootstrap(
    adapter=AgentAdapter(),
    notifications=[ApiNotifications(), SlackNotifications(), WSNotifications()],
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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, session_id: str = Query(...)):
    await websocket.accept()

    current_loop = asyncio.get_running_loop()
    connected_time = time()
    connected_clients[session_id] = {
        "ws": websocket,
        "loop": current_loop,
        "last_event_time": connected_time,  # Initialize last_event_time
    }

    logger.info(f"Client connected: {session_id}, loop: {id(current_loop)}")

    timeout = 90

    try:
        while True:
            await asyncio.sleep(1)

            client_info = connected_clients.get(session_id)
            if (
                not client_info
                or client_info["ws"].client_state == WebSocketState.DISCONNECTED
            ):
                logger.info(
                    f"Client {session_id} no longer in registry or disconnected, breaking loop."
                )
                break

            if time() - client_info["last_event_time"] > timeout:
                logger.info(f"Session timeout: {session_id}")
                await websocket.close(code=1000, reason="Idle timeout")
                break

    except WebSocketDisconnect:
        logger.info(f"Disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: session_id={session_id}: {e}")
    finally:
        connected_clients.pop(session_id, None)
        logger.info(f"WebSocket removed from registry: session_id={session_id}")

        if websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close(
                    code=1001, reason="Server shutting down connection"
                )
            except Exception:
                pass  # May already be closed or in a bad state
        logger.info(f"Connection closed for {session_id}")
