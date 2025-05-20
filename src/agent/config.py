from os import getenv
from pathlib import Path

ROOTDIR: str = str(Path(__file__).resolve().parents[1])


def get_agent_config():
    prompts_file = getenv("agent_prompts_file")

    if prompts_file is None:
        raise ValueError("prompts_file not set in environment variables")

    prompt_path = Path(ROOTDIR, "agent", "prompts", prompts_file)

    return dict(prompt_path=prompt_path)


def get_llm_config():
    model_id = getenv("llm_model_id")
    temperature = getenv("llm_temperature")

    if model_id is None:
        raise ValueError("llm_model_id not set in environment variables")

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

    prompt_path = Path(ROOTDIR, "agent", "prompts", prompts_file)

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
