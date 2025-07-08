"""Utilities for evaluation tests."""

import asyncio
import time
from typing import Any, Callable, Optional, Tuple

from fastapi.testclient import TestClient


class TestTimer:
    """Context manager for timing test execution."""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()

    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


async def wait_for_response(
    check_func: Callable[[], Any], timeout: float = 30.0, interval: float = 0.5
) -> Tuple[Any, float]:
    """
    Wait for a response with polling instead of fixed sleep.

    Args:
        check_func: Function to check if response is ready
        timeout: Maximum wait time in seconds
        interval: Polling interval in seconds

    Returns:
        Tuple of (result, duration)
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        result = check_func()
        if result:
            return result, time.time() - start_time
        await asyncio.sleep(interval)

    raise TimeoutError(f"Response not ready after {timeout} seconds")


def make_api_request_with_retry(
    client: TestClient,
    endpoint: str,
    params: dict,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Any:
    """
    Make API request with retry logic.

    Args:
        client: FastAPI test client
        endpoint: API endpoint
        params: Request parameters
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds

    Returns:
        API response
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.get(endpoint, params=params)
            if response.status_code == 200:
                return response
            last_error = f"HTTP {response.status_code}: {response.text}"
        except Exception as e:
            last_error = str(e)

        if attempt < max_retries - 1:
            time.sleep(retry_delay * (attempt + 1))  # Exponential backoff

    raise Exception(f"Failed after {max_retries} attempts. Last error: {last_error}")


def validate_test_data_format(
    test_data: dict, test_type: str
) -> Tuple[bool, Optional[str]]:
    """
    Validate test data format for a specific test type.

    Args:
        test_data: Test data dictionary
        test_type: Type of test (e2e, tool_agent, etc.)

    Returns:
        Tuple of (is_valid, error_message)
    """

    # Common required fields
    if "question" not in test_data:
        return False, "Missing 'question' field"

    # Type-specific validation
    if test_type == "e2e":
        if "response" not in test_data:
            return False, "Missing 'response' field for e2e test"

    elif test_type == "tool_agent":
        if "response" not in test_data:
            return False, "Missing 'response' field for tool_agent test"

    elif test_type == "ir":
        if "response" not in test_data:
            return False, "Missing 'response' field for ir test"
        # IR responses should be dict with specific fields

    elif test_type == "enhance":
        if "candidates" not in test_data or "response" not in test_data:
            return False, "Missing 'candidates' or 'response' field for enhance test"

    elif test_type in ["pre_check", "post_check"]:
        if "approved" not in test_data:
            return False, f"Missing 'approved' field for {test_type} test"

    # Validate judge criteria if present
    if "judge_criteria" in test_data:
        criteria = test_data["judge_criteria"]
        valid_thresholds = [
            "accuracy_threshold",
            "relevance_threshold",
            "completeness_threshold",
            "hallucination_threshold",
            "format_compliance_threshold",
        ]

        for key, value in criteria.items():
            if key in valid_thresholds:
                if not isinstance(value, (int, float)) or value < 0 or value > 10:
                    return False, f"Invalid threshold value for {key}: {value}"

    return True, None


def generate_test_id(test_type: str, feature: str, index: int) -> str:
    """
    Generate standardized test ID.

    Args:
        test_type: Type of test
        feature: Feature being tested
        index: Test index

    Returns:
        Standardized test ID
    """
    return f"{test_type}_{feature}_{index:03d}"


class ResponseCache:
    """Simple response cache to avoid repeated API calls during development."""

    def __init__(self, ttl: int = 900):  # 15 minutes default
        self.cache = {}
        self.timestamps = {}
        self.ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        """Get cached response if not expired."""
        if key in self.cache:
            if time.time() - self.timestamps[key] < self.ttl:
                return self.cache[key]
            else:
                # Expired, remove from cache
                del self.cache[key]
                del self.timestamps[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Cache a response."""
        self.cache[key] = value
        self.timestamps[key] = time.time()

    def clear(self) -> None:
        """Clear all cached responses."""
        self.cache.clear()
        self.timestamps.clear()
