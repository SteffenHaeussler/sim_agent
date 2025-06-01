# tests/conftest.py
import os

import pytest
from dotenv import load_dotenv


def pytest_addoption(parser):
    parser.addoption(
        "--envfile",
        action="store",
        default=".env.tests",
        help="Specify the .env file to load for tests",
    )


def pytest_configure(config):
    os.environ["IS_TESTING"] = "true"  # Set early
    env_file = config.getoption("--envfile")
    load_dotenv(env_file, override=True)


@pytest.fixture(scope="session", autouse=True)
def load_test_environment_fixture():
    assert os.getenv("IS_TESTING") == "true", "IS_TESTING not true in fixture!"
