import base64
import io
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.agent.adapters.tools.base import BaseTool


class GetData(BaseTool):
    name = "get_data"
    description = """Get data from an asset."""
    inputs = {
        "asset_ids": {"type": "list", "description": "list of asset ids"},
        "start_date": {"type": "string", "description": "start date", "nullable": True},
        "end_date": {"type": "string", "description": "end date", "nullable": True},
        "aggregation": {
            "type": "string",
            "description": "data aggregation",
            "nullable": True,
            "allowed": ["day", "minute", "hour"],
        },
        "last_value": {
            "type": "boolean",
            "description": "last value",
            "nullable": True,
        },
    }
    outputs = {"data": {"type": "dataframe", "description": "sensor data of an asset"}}
    output_type = "dict"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(
        self,
        asset_ids: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        aggregation: Optional[str] = "day",
        last_value: bool = False,
    ) -> Dict[str, List[str]]:
        df = pd.DataFrame()

        asset_ids = self.format_input(asset_ids)

        if not last_value:
            aggregation = self.map_aggregation(aggregation.lower())
            start_date = self.convert_to_iso_format(start_date)
            end_date = self.convert_to_iso_format(end_date)

        body = {
            "last_value": last_value,
            "start_date": start_date,
            "end_date": end_date,
            "aggregation": aggregation,
        }

        for asset_id in asset_ids:
            api_url = f"{self.base_url}/v1/data/{asset_id}"

            out = self.call_api(api_url, body=body)

            if out:
                _df = pd.DataFrame.from_dict(out)
                _df.set_index("timestamp", inplace=True)
                _df.drop(["pk_id", "asset_id"], inplace=True, axis=1)
                _df.columns = [asset_id]

                df = pd.merge(df, _df, left_index=True, right_index=True, how="outer")

        df.replace(np.nan, None, inplace=True)
        df.sort_index(inplace=True)

        return {"data": df}

    def map_aggregation(self, aggregation: str) -> str:
        """
        Map the aggregation to the correct aggregation type.
        """
        if aggregation not in ["day", "hour", "minute", "d", "h", "min"]:
            raise ValueError(
                f"Invalid aggregation: {aggregation} - only day, hour, minute, d, h, min are allowed"
            )

        if aggregation == "day" or aggregation == "d":
            aggregation = "d"
        elif aggregation == "hour" or aggregation == "h":
            aggregation = "h"
        elif aggregation == "minute" or aggregation == "min":
            aggregation = "min"
        else:
            aggregation = "d"

        return aggregation


class CompareData(BaseTool):
    name = "compare_data"
    description = """Compare data from two assets."""
    inputs = {
        "data": {"type": "dataframe", "description": "asset id data"},
    }
    outputs = {"data": {"type": "dataframe", "description": "compared sensor data"}}
    output_type = "dict"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        if isinstance(data, list):
            data = pd.concat(data, axis=1)

        if data.empty:
            comparison = pd.DataFrame()
        else:
            comparison = data.describe()

        return {"data": comparison}


class PlotData(BaseTool):
    name = "plot_data"
    description = """Plot data from data."""
    inputs = {
        "data": {"type": "dataframe", "description": "asset id data"},
    }
    outputs = {"plot": {"type": "str", "description": "encoded plot"}}
    output_type = "dict"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, data: pd.DataFrame) -> Dict[str, str]:
        if isinstance(data, list):
            data = pd.concat(data, axis=1)

        if data.empty:
            return {"plot": None}

        fig, ax = plt.subplots(figsize=(12, 6))

        for column_name in data.columns:
            ax.plot(
                data.index,
                data[column_name],
                label=column_name,
                marker="o",
                linestyle="--",
            )
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            ax.set_title("Time Series Plot (Manual)")
            ax.grid(True)
            ax.legend(title="Series Name")

        buf = io.BytesIO()

        # Save the figure to the buffer in PNG format (or 'jpeg', 'svg', etc.)
        # bbox_inches='tight' helps remove extra whitespace around the plot
        fig.savefig(buf, format="png", bbox_inches="tight")

        buf.seek(0)

        # Read the binary data from the buffer
        image_binary = buf.read()

        base64_bytes = base64.b64encode(image_binary)
        base64_string = base64_bytes.decode("utf-8")

        buf.close()
        plt.close(fig)

        return {"plot": base64_string}
