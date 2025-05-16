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
