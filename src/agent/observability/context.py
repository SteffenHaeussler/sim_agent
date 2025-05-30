import asyncio
from contextvars import ContextVar
from typing import Dict

from fastapi import WebSocket

ctx_query_id = ContextVar("query_id", default="-")


connected_clients: Dict[str, WebSocket] = {}
connected_streams: Dict[str, asyncio.Queue] = {}
