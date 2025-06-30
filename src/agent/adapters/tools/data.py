import base64
import io
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.agent.adapters.tools.base import BaseTool

matplotlib.use("agg")


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
        """
        Compare data from two assets.

        Args:
            data: pd.DataFrame: The data to compare.

        Returns:
            data: pd.DataFrame: The compared data.
        """
        if isinstance(data, list):
            data = pd.concat(data, axis=1)

        if data.empty:
            comparison = {}
        else:
            comparison = data.describe().to_dict()

        return {"comparison": comparison}


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
        """
        Initialize the GetData tool.

        Args:
            kwargs: Dict: The kwargs.
        """
        super().__init__(**kwargs)

    def forward(
        self,
        asset_ids: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        aggregation: Optional[str] = "day",
        last_value: bool = False,
    ) -> Dict[str, List[str]]:
        """
        Get data from an asset.

        Args:
            asset_ids: List[str]: The asset ids to get data from.
        """
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

        Args:
            aggregation: str: The aggregation to map.

        Returns:
            aggregation: str: The mapped aggregation.
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
        if data.empty:
            return {"plot": None}

        data, freq = self.simplify_time_index(data.copy())

        fig, ax = plt.subplots(figsize=(12, 6))

        for column_name in data.columns:
            ax.plot(
                data.index,
                data[column_name],
                label=column_name,
                marker="o",
                linestyle="--",
            )

        step = max(1, round(len(data.index) / 20))
        xticks = data.index[::step]

        # --- CHANGE 2: Format the labels based on the detected frequency ---
        if freq == "D":
            # For daily data, format as 'Year-Month-Day'
            xtick_labels = xticks.strftime("%Y-%m-%d")
        elif freq == "h":
            # For hourly data, format as 'Year-Month-Day Hour:Minute'
            xtick_labels = xticks.strftime("%Y-%m-%d %H:%M")
        else:
            # A sensible default for other frequencies (e.g., seconds, irregular)
            xtick_labels = xticks.strftime("%Y-%m-%d %H:%M:%S")

        ax.set_xticks(xticks)
        ax.set_xticklabels(
            xtick_labels, rotation=45, ha="right"
        )  # Use formatted labels

        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.set_title("Time Series Plot")
        ax.grid(True)
        ax.legend(title="Series Name")
        fig.tight_layout()

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

    def simplify_time_index(self, data):
        """
        Detects if a DataFrame's index is daily or hourly, simplifies it,
        and returns the modified DataFrame along with the detected frequency.

        Returns:
            tuple: (pd.DataFrame, str or None)
                   The modified DataFrame and the detected frequency string ('D', 'h', etc.).
        """
        data.index = pd.to_datetime(data.index)
        detected_freq = None  # Initialize a variable to store the frequency

        freq = pd.infer_freq(data.index)

        if freq == "D":
            data.index = data.index.normalize()
            detected_freq = "D"
        # Use .startswith() to catch 'H', 'h', '2H', etc.
        elif freq and freq.upper().startswith("H"):
            data.index = data.index.floor("h")
            detected_freq = "h"
        else:
            # Fallback check
            is_daily = (
                (data.index.hour == 0).all()
                and (data.index.minute == 0).all()
                and (data.index.second == 0).all()
            )
            is_hourly = (data.index.minute == 0).all() and (
                data.index.second == 0
            ).all()

            if is_daily:
                data.index = data.index.normalize()
                detected_freq = "D"
            elif is_hourly:
                data.index = data.index.floor("h")
                detected_freq = "h"

        return data, detected_freq
