import asyncio
from time import time

import src.agent.service_layer.handlers as handlers
from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from loguru import logger
from src.agent.adapters.adapter import RouterAdapter
from src.agent.adapters.notifications import SlackNotifications, WSNotifications
from src.agent.bootstrap import bootstrap
from src.agent.config import get_logging_config, get_tracing_config
from src.agent.domain.commands import Question, SQLQuestion
from src.agent.observability.context import connected_clients, ctx_query_id
from src.agent.observability.logging import setup_logging
from src.agent.observability.tracing import setup_tracing
from starlette.responses import StreamingResponse
from starlette.websockets import WebSocketState

setup_tracing(get_tracing_config())
setup_logging(get_logging_config())

app = FastAPI()


bus = bootstrap(
    adapter=RouterAdapter(),
    notifications=[
        SlackNotifications(),
        WSNotifications(),
    ],
)


@app.get("/answer")
async def answer(
    question: str,
    x_session_id: str = Header(default="default-session", alias="X-Session-ID"),
):
    """
    Entrypoint for the agent.

    Args:
        question: str: The question to answer.
        x_session_id: str: Session ID from X-Session-ID header.

    Returns:
        response: str: The response to the question.

    Raises:
        HTTPException: If the question is invalid.
        ValueError: If the question is invalid.
    """
    session_id = x_session_id
    logger.info(f"session_id: {session_id}")

    if not question:
        raise HTTPException(status_code=400, detail="No question asked")

    ctx_query_id.set(session_id)

    try:
        command = Question(question=question, q_id=session_id)
        # Run the command handling in the background
        asyncio.create_task(asyncio.to_thread(bus.handle, command))
        return {"status": "processing", "message": "Event triggered successfully"}

    except (handlers.InvalidQuestion, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/query")
async def query(
    question: str,
    x_session_id: str = Header(default="default-session", alias="X-Session-ID"),
):
    """
    Entrypoint for the SQL agent.

    Args:
        question: str: The question to answer via SQL.
        x_session_id: str: Session ID from X-Session-ID header.

    Returns:
        response: str: The response to the question.

    Raises:
        HTTPException: If the question is invalid.
        ValueError: If the question is invalid.
    """
    session_id = x_session_id
    logger.info(f"session_id: {session_id}")

    if not question:
        raise HTTPException(status_code=400, detail="No question asked")

    ctx_query_id.set(session_id)

    try:
        command = SQLQuestion(question=question, q_id=session_id)
        # Run the command handling in the background
        asyncio.create_task(asyncio.to_thread(bus.handle, command))
        return {"status": "processing", "message": "Event triggered successfully"}

    except (handlers.InvalidQuestion, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


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


@app.get("/sse/{session_id}")
async def sse(request: Request, session_id: str):
    loop = asyncio.get_event_loop()

    if session_id not in connected_clients:
        connected_clients[session_id] = {
            "queue": asyncio.Queue(),
            "loop": loop,
            "last_event_time": time(),
        }

    queue = connected_clients[session_id]["queue"]

    async def event_stream():
        while True:
            if await request.is_disconnected():
                logger.info(f"Client {session_id} disconnected from SSE.")
                connected_clients.pop(session_id, None)
                break

            try:
                message = await asyncio.wait_for(queue.get(), timeout=20.0)
                yield message
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )
