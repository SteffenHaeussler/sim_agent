import json
import logging
import sys

from loguru import logger

from src.agent.observability.context import ctx_query_id


def query_id_filter(record):
    record["query_id"] = ctx_query_id.get()
    return record["query_id"]


class InterceptHandler(logging.Handler):
    """
    Default handler from examples in loguru documentaion.
    See https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging
    """

    def emit(self, record: logging.LogRecord):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def sink_serializer(message):
    record = message.record
    simplified = {
        "level": record["level"].name,
        "message": record["message"],
        "timestamp": record["time"].timestamp(),
        "query_id": record["query_id"],
    }
    serialized = json.dumps(simplified)
    print(serialized, file=sys.stdout)


def setup_logging(config: dict):
    logging_level = config.get("logging_level")
    logging_format = config.get("logging_format")

    intercept_handler = InterceptHandler()

    loggers = (
        logging.getLogger(name)
        for name in logging.root.manager.loggerDict
        if name.startswith("uvicorn.")
    )
    for uvicorn_logger in loggers:
        uvicorn_logger.handlers = [intercept_handler]

    if logging_level.lower() == "error":
        service_log_level = logging.ERROR
    elif logging_level.lower() == "warning":
        service_log_level = logging.WARNING
    elif logging_level.lower() == "info":
        service_log_level = logging.INFO
    else:
        service_log_level = logging.DEBUG

    if logging_format.lower() == "json":
        logger.configure(
            handlers=[
                {
                    "sink": sink_serializer,
                    "level": service_log_level,
                    "filter": query_id_filter,
                }
            ]
        )

    else:
        fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <red> {query_id} </red> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

        logger.configure(
            handlers=[
                {
                    "sink": sys.stdout,
                    "level": service_log_level,
                    "format": fmt,
                    "filter": query_id_filter,
                }
            ]
        )
