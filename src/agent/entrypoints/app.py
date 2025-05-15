from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException

import src.agent.service_layer.handlers as handlers
from src.agent.adapters.adapter import AgentAdapter
from src.agent.adapters.notifications import ApiNotifications
from src.agent.bootstrap import bootstrap
from src.agent.domain.commands import Question

app = FastAPI()

bus = bootstrap(
    adapter=AgentAdapter(),
    notifications=ApiNotifications(),
)


@app.get("/answer")
def answer(question: str, q_id: Optional[str] = None):
    if not q_id:
        q_id = uuid4().hex
    try:
        command = Question(question, q_id)
        bus.handle(command)

    except (handlers.InvalidQuestion, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    message = ApiNotifications().temp.pop(q_id, None)
    return {"response": message}
