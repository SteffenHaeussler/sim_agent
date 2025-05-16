# tests/conftest.py
import os

import pytest
from dotenv import load_dotenv


def pytest_configure(config):
    os.environ["IS_TESTING"] = "true"  # Set the flag as early as possible

    load_dotenv(".env.tests", override=True)


@pytest.fixture(scope="session", autouse=True)
def load_test_environment_fixture():  # Renamed to avoid confusion
    assert os.getenv("IS_TESTING") == "true", "IS_TESTING not true in fixture!"
