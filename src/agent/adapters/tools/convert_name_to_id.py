from typing import Dict, List

from loguru import logger
from src.agent.adapters.tools.base import BaseTool


class ConvertNameToId(BaseTool):
    name = "name_to_id"
    description = """Converts asset names to ids."""
    inputs = {"names": {"type": "list", "description": "list of asset names"}}
    outputs = {"asset_ids": {"type": "list", "description": "list of asset ids"}}
    output_type = "dict"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, names: List[str]) -> Dict[str, List[str]]:
        names = self.format_input(names)

        response = []

        for name in names:
            api_url = f"{self.base_url}/v1/name_from_id/{name}"

            out = self.call_api(api_url)

            if out:
                response.append(out)
            else:
                logger.warning(f"No name found for asset id {name}")

        return {"asset_ids": response}
