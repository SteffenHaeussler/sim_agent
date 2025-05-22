import json
from datetime import datetime
from typing import Dict, List, Optional, Union

import httpx
from dateutil import parser
from loguru import logger
from smolagents import Tool, tools

TARGET_FORMAT = "%Y-%m-%dT%H:%M:%S"

FALLBACK_FORMATS = [
    "%Y-%m-%d %H:%M:%S.%f",  # With microseconds
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",  # Day first
    "%m/%d/%Y %H:%M:%S",  # Month first
    "%d-%b-%Y %I:%M:%S %p",  # e.g., 31-Dec-2025 11:59:00 PM
    "%Y-%m-%d",  # Date only
    "%d/%m/%Y",  # Date only (Day first)
    "%m/%d/%Y",  # Date only (Month first)
    "%b %d %Y",  # e.g., Dec 31 2025
    "%B %d, %Y",  # e.g., December 31, 2025
]

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
    "dataframe",
]


class BaseTool(Tool):
    """
    BaseTool is a class that implements the Tool interface.  All tools should inherit from this class.

    Methods:
        - format_input(ids: List[str]) -> List[str]: Format the input.
        - call_api(api_url: str, body: Dict = {}, method: str = "get") -> Optional[httpx.Response]: Call the API.
        - convert_to_iso_format(date_string: str) -> Optional[str]: Convert the date string to the target ISO-like format.
        - call_api(api_url: str, body: Dict = {}, method: str = "get") -> Optional[httpx.Response]: Call the API.
    """

    def __init__(self, **kwargs: Dict):
        """
        Initialize the BaseTool.

        Args:
            kwargs: Dict: The kwargs.
        """
        super().__init__(**kwargs)
        self.base_url = kwargs["tools_api_base"]
        self.limit = int(kwargs["tools_api_limit"])

    @staticmethod
    def format_input(ids: Union[List[str], str]) -> List[str]:
        """
        Harmonizes the input. Also removes duplicates, None and converts to list.

        Args:
            ids: Union[List[str], str]: The ids to format.

        Returns:
            ids: List[str]: The formatted ids.
        """
        if isinstance(ids, str):
            ids = [ids]

        ids = [str(i) for i in ids if i]
        ids = list(set(ids))

        return ids

    def call_api(self, api_url: str, body: Dict = {}) -> List[dict]:
        """
        Calls the specific API for each tool. Includes pagination logic.

        Args:
            api_url: str: The API URL.
            body: Dict: The body of the request.

        Returns:
            all_results: List[dict]: The results from the API.
        """
        all_results = []
        current_offset = 0

        while True:
            body.update(
                {
                    "offset": current_offset,
                    "limit": self.limit,
                }
            )

            try:
                logger.info(f"Fetching data from {api_url} with params: {body}")
                response = httpx.get(api_url, params=body, timeout=30.0)  # Add timeout

                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

                page_results = response.json()  # Expecting a list of Data objects

                if isinstance(page_results, list):
                    all_results.extend(page_results)
                else:
                    all_results.append(page_results)
                # --- Pagination Logic ---
                if len(page_results) < self.limit:
                    # If we received fewer items than we asked for, this must be the last page
                    logger.info("Reached the last page.")
                    break
                else:
                    # Otherwise, prepare for the next page
                    current_offset += self.limit

            except httpx.HTTPStatusError as e:
                logger.debug(
                    f"HTTP error fetching name for {api_url}: {e.response.status_code} - {e.response.text}"
                )
                break
            except httpx.RequestError as e:
                logger.debug(f"Request error fetching name for {api_url}: {e}")
                break
            except json.JSONDecodeError as e:
                logger.debug(
                    f"JSON decode error for {api_url}. Response text: {response.text}. Error: {e}"
                )
                break
            except Exception as e:  # Catch any other unexpected errors
                logger.debug(f"An unexpected error occurred for {api_url}: {e}")
                break

        return all_results

    @staticmethod
    def convert_to_iso_format(date_string: str) -> Optional[str]:
        """
        Tries to parse a date string from various common formats and
        converts it to the target ISO-like format: YYYY-MM-DDTHH:MM:SS.

        Args:
            date_string: The string representation of the date/time.

        Returns:
            The formatted date string or None if parsing fails.
        """
        if not isinstance(date_string, str):
            raise ValueError(f"Input '{date_string}' is not a string.")

        dt_object = None

        try:
            # dayfirst=None (default): Infer from context, often US-style (MM/DD) for ambiguous cases.
            dt_object = parser.parse(date_string)
        except (parser.ParserError, ValueError, TypeError):
            pass  # Continue to fallback methods

        # 2. If dateutil.parser failed, try with a list of specific formats
        if dt_object is None:
            for fmt in FALLBACK_FORMATS:
                try:
                    dt_object = datetime.strptime(date_string, fmt)
                    break  # Successfully parsed
                except ValueError:
                    continue  # Try next format

        try:
            dt_object = dt_object.strftime(TARGET_FORMAT)
        except (
            ValueError
        ):  # Can happen for dates before year 1900 with some strftime directives
            # print(f"Error formatting datetime object for '{date_string}': {e}")
            dt_object = None

        if not dt_object:
            raise ValueError(
                f"Could not parse date string: '{date_string}' with any known format."
            )

        return dt_object
