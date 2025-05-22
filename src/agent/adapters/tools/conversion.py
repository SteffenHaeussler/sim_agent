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
        """
        Converts asset ids to the asset names.

        Args:
            asset_ids: List[str]: The ids to convert.
        """
        asset_ids = self.format_input(asset_ids)

        response = []

        for _id in asset_ids:
            api_url = f"{self.base_url}/v1/name_from_id/{_id}"

            out = self.call_api(api_url)

            if out:
                response.extend(out)
            else:
                logger.warning(f"No name found for asset id {_id}")

        return {"names": response}


class ConvertNameToId(BaseTool):
    name = "name_to_id"
    description = """Converts asset names to ids."""
    inputs = {"names": {"type": "list", "description": "list of asset names"}}
    outputs = {"asset_ids": {"type": "list", "description": "list of asset ids"}}
    output_type = "dict"

    def __init__(self, **kwargs):
        """
        Initialize the ConvertNameToId tool.

        Args:
            kwargs: Dict: The kwargs.
        """
        super().__init__(**kwargs)

    def forward(self, names: List[str]) -> Dict[str, List[str]]:
        """
        Converts asset names to ids.

        Args:
            names: List[str]: The names to convert.
        """
        names = self.format_input(names)

        response = []

        for name in names:
            api_url = f"{self.base_url}/v1/id_from_name/{name}"

            out = self.call_api(api_url)

            if out:
                response.extend(out)
            else:
                logger.warning(f"No name found for asset id {name}")

        return {"asset_ids": response}
