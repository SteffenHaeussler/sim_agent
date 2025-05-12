from typing import Dict, List

from loguru import logger

from src.agent.adapters.tools.base import BaseTool


class GetInformation(BaseTool):
    name = "asset_information"
    description = """Get information about an asset."""
    inputs = {"asset_ids": {"type": "list", "description": "list of asset ids"}}
    outputs = {"assets": {"type": "list", "description": "list of asset information"}}
    output_type = "dict"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, asset_ids: List[str]) -> Dict[str, List[str]]:
        asset_ids = self.format_input(asset_ids)

        response = []

        for asset_id in asset_ids:
            api_url = f"{self.base_url}/v1/assets/{asset_id}"

            out = self.call_api(api_url)

            if out:
                response.append(out)
            else:
                logger.warning(f"No information found for asset id {asset_id}")

        return {"assets": response}
