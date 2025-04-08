from os import getenv

from dotenv import load_dotenv

load_dotenv()


def get_tools_config():
    model_name = getenv("tools_llm_model_name", None)

    if model_name is None:
        raise ValueError("tools_llm_model_name not set in environment variables")

    return dict(model_name=model_name)


def get_llm_config():
    model_name = getenv("llm_model_name", None)

    if model_name is None:
        raise ValueError("llm_model_name not set in environment variables")

    return dict(model_name=model_name)
