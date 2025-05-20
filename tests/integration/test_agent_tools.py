import os
from pathlib import Path
from unittest.mock import patch

from src.agent.adapters.agent_tools import Tools

ROOTDIR: str = str(Path(__file__).resolve().parents[2])

# tests fail when telemetry is enabled
os.environ["TELEMETRY_ENABLED"] = "false"


class TestTools:
    def test_tools_init(self):
        kwargs = {
            "llm_model_id": "test_model_id",
            "max_steps": 3,
            "prompt_path": Path(
                ROOTDIR, "src", "agent", "prompts", "base_prompts.yaml"
            ),
            "llm_api_base": "http://test_url.com",
            "tools_api_base": "http://test_url.com",
            "tools_api_limit": 100,
        }

        tools_instance = Tools(kwargs)

        assert tools_instance.llm_model_id == kwargs["llm_model_id"]
        assert tools_instance.max_steps == kwargs["max_steps"]
        assert tools_instance.kwargs["llm_api_base"] == kwargs["llm_api_base"]
        assert tools_instance.prompt_templates is not None
        assert tools_instance.agent is not None

    @patch("src.agent.adapters.agent_tools.CodeAgent")
    @patch("src.agent.adapters.agent_tools.LiteLLMModel")
    @patch("src.agent.adapters.tools.CompareData")
    @patch("src.agent.adapters.tools.ConvertIdToName")
    @patch("src.agent.adapters.tools.ConvertNameToId")
    @patch("src.agent.adapters.tools.GetData")
    @patch("src.agent.adapters.tools.GetInformation")
    @patch("src.agent.adapters.tools.GetNeighbors")
    @patch("src.agent.adapters.tools.PlotData")
    @patch("src.agent.adapters.tools.FinalAnswerTool")
    def test_init_agent(
        self,
        mock_FinalAnswerTool,
        mock_PlotData,
        mock_GetNeighbors,
        mock_GetInformation,
        mock_GetData,
        mock_ConvertNameToId,
        mock_ConvertIdToName,
        mock_CompareData,
        mock_LiteLLMModel,
        mock_CodeAgent,
    ):
        kwargs = {
            "llm_model_id": "test_model_id",
            "max_steps": 3,
            "prompt_path": Path(
                ROOTDIR, "src", "agent", "prompts", "base_prompts.yaml"
            ),
            "llm_api_base": "http://test_url.com",
            "tools_api_base": "http://test_url.com",
            "tools_api_limit": 100,
        }

        tools_instance = Tools(kwargs)

        agent = tools_instance.agent

        mock_CodeAgent.assert_called_once()
        mock_LiteLLMModel.assert_called_once_with(
            model_id=kwargs["llm_model_id"], api_base=kwargs["llm_api_base"]
        )
        assert agent == mock_CodeAgent.return_value

    @patch("src.agent.adapters.agent_tools.CodeAgent")
    def test_tools_use(self, mock_CodeAgent):
        kwargs = {
            "llm_model_id": "test_model_id",
            "max_steps": 3,
            "prompt_path": Path(
                ROOTDIR, "src", "agent", "prompts", "base_prompts.yaml"
            ),
            "llm_api_base": "http://test_url.com",
            "tools_api_base": "http://test_url.com",
            "tools_api_limit": 100,
        }
        question = "What is the capital of France?"

        mock_agent_instance = mock_CodeAgent.return_value
        mock_agent_instance.run.return_value = "The capital of France is Paris."

        tools_instance = Tools(kwargs)

        response = tools_instance.use(question)

        mock_agent_instance.run.assert_called_once_with(question)
        assert response == "The capital of France is Paris."
        assert response == "The capital of France is Paris."
