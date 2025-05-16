from contextvars import ContextVar

ctx_query_id = ContextVar("query_id", default="-")
