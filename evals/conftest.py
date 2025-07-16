"""Pytest configuration and fixtures for evaluation tests."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOTDIR: str = str(Path(__file__).resolve().parents[1])


def get_agent_config():
    prompts_file = os.getenv("agent_prompts_file")

    if prompts_file is None:
        raise ValueError("prompts_file not set in environment variables")

    sql_prompts_file = os.getenv("sql_prompts_file")

    if sql_prompts_file is None:
        raise ValueError("sql_prompts_file not set in environment variables")

    scenario_prompts_file = os.getenv("scenario_prompts_file")

    if scenario_prompts_file is None:
        raise ValueError("scenario_prompts_file not set in environment variables")

    prompt_path = Path(ROOTDIR, prompts_file)
    sql_prompt_path = Path(ROOTDIR, sql_prompts_file)
    scenario_prompt_path = Path(ROOTDIR, scenario_prompts_file)

    return dict(
        prompt_path=prompt_path,
        sql_prompt_path=sql_prompt_path,
        scenario_prompt_path=scenario_prompt_path,
    )


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


def get_rag_config():
    embedding_api_base = os.getenv("embedding_api_base")
    retrieval_api_base = os.getenv("retrieval_api_base")
    ranking_api_base = os.getenv("ranking_api_base")

    embedding_endpoint = os.getenv("embedding_endpoint")
    ranking_endpoint = os.getenv("ranking_endpoint")
    retrieval_endpoint = os.getenv("retrieval_endpoint")

    n_ranking_candidates = os.getenv("n_ranking_candidates")
    n_retrieval_candidates = os.getenv("n_retrieval_candidates")
    retrieval_table = os.getenv("retrieval_table")
    if embedding_api_base is None or embedding_endpoint is None:
        raise ValueError(
            "embedding_api_base or embedding_endpoint not set in environment variables"
        )

    if retrieval_api_base is None or retrieval_endpoint is None:
        raise ValueError(
            "retrieval_api_base or retrieval_endpoint not set in environment variables"
        )

    if ranking_api_base is None or ranking_endpoint is None:
        raise ValueError(
            "ranking_api_base or ranking_endpoint not set in environment variables"
        )

    if retrieval_table is None:
        raise ValueError("retrieval_table not set in environment variables")

    embedding_url = f"{embedding_api_base}/{embedding_endpoint}"
    ranking_url = f"{ranking_api_base}/{ranking_endpoint}"
    retrieval_url = f"{retrieval_api_base}/{retrieval_endpoint}"

    return dict(
        embedding_url=embedding_url,
        ranking_url=ranking_url,
        retrieval_url=retrieval_url,
        n_ranking_candidates=n_ranking_candidates,
        n_retrieval_candidates=n_retrieval_candidates,
        retrieval_table=retrieval_table,
    )


def get_tools_config():
    llm_model_id = os.getenv("tools_model_id")
    llm_api_base = os.getenv("tools_model_api_base")
    max_steps = os.getenv("tools_max_steps")
    prompts_file = os.getenv("tools_prompts_file")
    tools_api_base = os.getenv("tools_api_base")
    tools_api_limit = os.getenv("tools_api_limit")

    if llm_model_id is None:
        raise ValueError("tools_model_id not set in environment variables")

    if prompts_file is None:
        raise ValueError("tools_prompts_file not set in environment variables")

    if tools_api_base is None:
        raise ValueError("tools_api_base not set in environment variables")

    prompt_path = Path(ROOTDIR, prompts_file)

    return dict(
        llm_model_id=llm_model_id,
        llm_api_base=llm_api_base,
        max_steps=max_steps,
        prompt_path=prompt_path,
        tools_api_base=tools_api_base,
        tools_api_limit=tools_api_limit,
    )


def get_llm_config():
    model_id = os.getenv("llm_model_id")
    temperature = os.getenv("llm_temperature")

    if model_id is None:
        raise ValueError("llm_model_id not set in environment variables")

    return dict(model_id=model_id, temperature=temperature)


def pytest_configure(config):
    """Set up environment variables before tests run."""
    # Load .env file for evaluation tests
    load_dotenv(".env", override=True)

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


@pytest.fixture(scope="session")
def rag_adapter():
    """Provide a RAG adapter instance for tests."""
    from src.agent.adapters import rag

    return rag.BaseRAG(get_rag_config())


@pytest.fixture(scope="session")
def llm_config():
    """Provide LLM configuration for tests."""
    return get_llm_config()


@pytest.fixture(scope="session")
def rag_config():
    """Provide RAG configuration for tests."""
    return get_rag_config()


@pytest.fixture(scope="session")
def agent_config():
    """Provide LLM configuration for tests."""
    return get_agent_config()


@pytest.fixture(scope="session")
def tools_config():
    """Provide LLM configuration for tests."""
    return get_tools_config()


@pytest.fixture(scope="session")
def test_notifications():
    """Provide a CollectingNotifications instance for tests."""
    from evals.utils import CollectingNotifications

    return CollectingNotifications()


@pytest.fixture(scope="session")
def test_app(test_notifications):
    """Provide a test app with CollectingNotifications."""
    import src.agent.entrypoints.app
    from src.agent import bootstrap
    from src.agent.adapters.adapter import RouterAdapter
    from src.agent.entrypoints.app import app

    # Create test messagebus
    test_bus = bootstrap.bootstrap(
        adapter=RouterAdapter(), notifications=[test_notifications]
    )

    # Replace the app's bus
    src.agent.entrypoints.app.bus = test_bus

    return app


@pytest.fixture
def test_client(test_app):
    """Provide a test client with CollectingNotifications."""
    from fastapi.testclient import TestClient

    return TestClient(test_app)
