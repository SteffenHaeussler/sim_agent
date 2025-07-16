"""Pytest configuration and fixtures for evaluation tests."""

import os

import pytest


def get_database_config():
    db_user = os.getenv("PG_USER", "postgres")
    db_password = os.getenv("PG_PASSWORD", "example")
    db_host = os.getenv("PG_HOST", "localhost")
    db_port = os.getenv("PG_PORT", "5432")
    db_name = os.getenv("PG_EVAL_DB", "evaluation")

    database_connection_string = (
        f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )
    database_type = os.getenv("database_type", "postgres")

    if database_connection_string is None:
        raise ValueError("database_connection_string not set in environment variables")

    return dict(
        connection_string=database_connection_string,
        db_type=database_type,
    )


def pytest_configure(config):
    """Set up environment variables before tests run."""
    # Set testing environment
    os.environ["IS_TESTING"] = "true"

    # Set database connection for evaluation tests
    try:
        db_config = get_database_config()
        os.environ["EVALS_DB_CONNECTION"] = db_config["connection_string"]
    except ValueError as e:
        # If database config is not available, tests will run without database saving
        print(f"Warning: Database configuration not available: {e}")
        print("Tests will save to JSON only.")

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
