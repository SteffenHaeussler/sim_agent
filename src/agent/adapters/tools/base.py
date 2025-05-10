import json
from typing import List

import httpx
from loguru import logger
from smolagents import Tool, tools

## monkey patching

tools.AUTHORIZED_TYPES = [
    "string",
    "boolean",
    "integer",
    "number",
    "image",
    "audio",
    "array",
    "object",
    "any",
    "null",
    "list",
    "dict",
]


class BaseTool(Tool):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = kwargs.get("base_url")

    @staticmethod
    def format_input(ids: List[str]):
        ids = [str(i) for i in ids if i]
        ids = list(set(ids))

        return ids

    @staticmethod
    def call_api(api_url, body=None):
        try:
            out = httpx.get(api_url)
            out.raise_for_status()

            return out.json()

        except httpx.HTTPStatusError as e:
            logger.debug(
                f"HTTP error fetching name for {api_url}: {e.out.status_code} - {e.out.text}"
            )
        except httpx.RequestError as e:
            logger.debug(f"Request error fetching name for {api_url}: {e}")
        except json.JSONDecodeError as e:
            logger.debug(
                f"JSON decode error for {api_url}. Response text: {out.text}. Error: {e}"
            )
        except Exception as e:  # Catch any other unexpected errors
            logger.debug(f"An unexpected error occurred for {api_url}: {e}")

        return None
