from typing import Dict, List

from loguru import logger
from src.agent.adapters.tools.base import BaseTool


class ConvertIdToName(BaseTool):
    name = "id_to_name"
    description = """Converts asset ids to the asset names."""
    inputs = {"asset_ids": {"type": "list", "description": "list of asset ids"}}
    outputs = {"names": {"type": "list", "description": "list of asset names"}}
    output_type = "dict"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, asset_ids: List[str]) -> Dict[str, List[str]]:
        asset_ids = self.format_input(asset_ids)

        response = []

        for _id in asset_ids:
            api_url = f"{self.base_url}/v1/name_from_id/{_id}"

            out = self.call_api(api_url)

            if out:
                response.append(out)
            else:
                logger.warning(f"No name found for asset id {_id}")

        return {"names": response}
