from typing import Dict, List

from loguru import logger

from src.agent.adapters.tools.base import BaseTool


class GetNeighbors(BaseTool):
    name = "get_neighbors"
    description = """Get neighbors of an asset."""
    inputs = {"asset_ids": {"type": "list", "description": "list of asset ids"}}
    outputs = {"asset_ids": {"type": "list", "description": "list of neighbor ids"}}
    output_type = "dict"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, asset_ids: List[str]) -> Dict[str, List[str]]:
        asset_ids = self.format_input(asset_ids)

        response = []

        for asset_id in asset_ids:
            api_url = f"{self.base_url}/v1/neighbor/{asset_id}"

            out = self.call_api(api_url)

            if out:
                response.extend(out)
            else:
                logger.warning(f"No neighbors found for asset id {asset_id}")

        return {"asset_ids": response}
