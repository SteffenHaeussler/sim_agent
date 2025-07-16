"""Pytest configuration and fixtures for evaluation tests."""

import os
import pytest


def pytest_configure(config):
    """Set up environment variables before tests run."""
    # Set testing environment
    os.environ["IS_TESTING"] = "true"

    # You can add other environment variables here
    # os.environ["LOG_LEVEL"] = "WARNING"


def pytest_unconfigure(config):
    """Clean up after tests complete."""
    # Optionally remove the environment variable after tests
    if "IS_TESTING" in os.environ:
        del os.environ["IS_TESTING"]


# You can also add shared fixtures here
@pytest.fixture(scope="session")
def test_environment():
    """Ensure test environment is properly configured."""
    assert os.getenv("IS_TESTING") == "true"
    return True
