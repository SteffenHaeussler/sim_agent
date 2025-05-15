from os import getenv
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOTDIR: str = str(Path(__file__).resolve().parents[1])


def get_agent_config():
    prompts_file = getenv("agent_prompts_file")

    if prompts_file is None:
        raise ValueError("prompts_file not set in environment variables")

    prompt_path = Path(ROOTDIR, "agent", "prompts", prompts_file)

    return dict(prompt_path=prompt_path)


def get_llm_config():
    model_id = getenv("llm_model_id", None)
    temperature = getenv("llm_temperature", 0.0)

    if model_id is None:
        raise ValueError("llm_model_id not set in environment variables")

    return dict(model_id=model_id, temperature=temperature)


def get_tools_config():
    llm_model_id = getenv("tools_model_id", None)
    llm_api_base = getenv("tools_model_api_base", None)
    max_steps = getenv("tools_max_steps", None)
    prompts_file = getenv("tools_prompts_file", None)
    tools_api_base = getenv("tools_api_base", None)

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
    )
