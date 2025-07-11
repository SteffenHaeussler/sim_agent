from os import getenv
from pathlib import Path

ROOTDIR: str = str(Path(__file__).resolve().parents[2])


def get_agent_config():
    prompts_file = getenv("agent_prompts_file")

    if prompts_file is None:
        raise ValueError("prompts_file not set in environment variables")

    sql_prompts_file = getenv("sql_prompts_file")

    if sql_prompts_file is None:
        raise ValueError("sql_prompts_file not set in environment variables")

    prompt_path = Path(ROOTDIR, prompts_file)
    sql_prompt_path = Path(ROOTDIR, sql_prompts_file)

    return dict(prompt_path=prompt_path, sql_prompt_path=sql_prompt_path)


def get_llm_config():
    model_id = getenv("llm_model_id")
    temperature = getenv("llm_temperature")

    if model_id is None:
        raise ValueError("llm_model_id not set in environment variables")

    return dict(model_id=model_id, temperature=temperature)


def get_guardrails_config():
    model_id = getenv("guardrails_model_id")
    temperature = getenv("guardrails_temperature")
    if model_id is None:
        raise ValueError("guardrails_model_id not set in environment variables")

    return dict(model_id=model_id, temperature=temperature)


def get_rag_config():
    embedding_api_base = getenv("embedding_api_base")
    retrieval_api_base = getenv("retrieval_api_base")
    ranking_api_base = getenv("ranking_api_base")

    embedding_endpoint = getenv("embedding_endpoint")
    ranking_endpoint = getenv("ranking_endpoint")
    retrieval_endpoint = getenv("retrieval_endpoint")

    n_ranking_candidates = getenv("n_ranking_candidates")
    n_retrieval_candidates = getenv("n_retrieval_candidates")
    retrieval_table = getenv("retrieval_table")
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
    llm_model_id = getenv("tools_model_id")
    llm_api_base = getenv("tools_model_api_base")
    max_steps = getenv("tools_max_steps")
    prompts_file = getenv("tools_prompts_file")
    tools_api_base = getenv("tools_api_base")
    tools_api_limit = getenv("tools_api_limit")

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


def get_tracing_config():
    langfuse_public_key = getenv("langfuse_public_key")
    langfuse_secret_key = getenv("langfuse_secret_key")
    langfuse_project_id = getenv("langfuse_project_id")
    langfuse_host = getenv("langfuse_host")
    otel_exporter_otlp_endpoint = "https://cloud.langfuse.com/api/public/otel"
    telemetry_enabled = getenv("telemetry_enabled", "false")

    if langfuse_public_key is None:
        raise ValueError("langfuse_public_key not set in environment variables")

    if langfuse_project_id is None:
        raise ValueError("langfuse_project_id not set in environment variables")

    if langfuse_host is None:
        raise ValueError("langfuse_host not set in environment variables")

    if langfuse_secret_key is None:
        raise ValueError("langfuse_secret_key not set in environment variables")

    return dict(
        langfuse_public_key=langfuse_public_key,
        langfuse_project_id=langfuse_project_id,
        langfuse_host=langfuse_host,
        langfuse_secret_key=langfuse_secret_key,
        otel_exporter_otlp_endpoint=otel_exporter_otlp_endpoint,
        telemetry_enabled=telemetry_enabled,
    )


def get_logging_config():
    logging_level = getenv("logging_level")
    logging_format = getenv("logging_format")

    return dict(logging_level=logging_level, logging_format=logging_format)


def get_email_config():
    smtp_host = getenv("smtp_host")
    smtp_port = getenv("smtp_port")
    receiver_email = getenv("receiver_email")
    sender_email = getenv("sender_email")
    app_password = getenv("app_password")

    return dict(
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        sender_email=sender_email,
        receiver_email=receiver_email,
        app_password=app_password,
    )


def get_slack_config():
    slack_webhook_url = getenv("slack_webhook_url")

    return dict(slack_webhook_url=slack_webhook_url)


def get_database_config():
    db_user = getenv("PG_USER")
    db_password = getenv("PG_PASSWORD")
    db_host = getenv("PG_HOST")
    db_port = getenv("PG_PORT")
    db_name = getenv("PG_NAME")

    database_connection_string = (
        f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )
    database_type = getenv("database_type", "postgres")

    if database_connection_string is None:
        raise ValueError("database_connection_string not set in environment variables")

    return dict(
        connection_string=database_connection_string,
        db_type=database_type,
    )


def get_evaluation_database_config():
    """Get configuration for the evaluation database."""
    db_user = getenv("PG_USER")
    db_password = getenv("PG_PASSWORD")
    db_host = getenv("PG_HOST")
    db_port = getenv("PG_PORT")
    db_name = getenv("PG_EVAL_DB", "evaluation")  # Default to 'evaluation'

    evaluation_connection_string = (
        f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )

    if not all([db_user, db_password, db_host, db_port]):
        raise ValueError(
            "PostgreSQL connection parameters not set in environment variables"
        )

    return dict(
        connection_string=evaluation_connection_string,
        db_type="postgres",
        db_name=db_name,
    )
