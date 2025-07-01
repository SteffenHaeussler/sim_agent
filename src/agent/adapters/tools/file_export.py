import io
import os
from typing import Dict
from uuid import uuid4

import nc_py_api
import pandas as pd
from src.agent.adapters.tools.base import BaseTool


class ExportData(BaseTool):
    name = "export_data"
    description = """Stores data in external cloud storage."""
    inputs = {
        "data": {"type": "dataframe", "description": "asset id data"},
    }
    outputs = {"url": {"type": "str", "description": "storage link"}}
    output_type = "dict"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, data: pd.DataFrame) -> Dict[str, str]:
        if data.empty:
            return {"url": None}

        nc = nc_py_api.Nextcloud(
            nextcloud_url=os.getenv("NX_URL"),
            nc_auth_user=os.getenv("NX_USER"),
            nc_auth_pass=os.getenv("NX_PASSWORD"),
        )

        file_name = f"agent/{str(uuid4())}.csv"

        buffer = io.BytesIO()

        # DataFrame als CSV in den BytesIO-Puffer schreiben
        data.to_csv(buffer, index=False)

        buffer.seek(0)
        nc.files.upload_stream(file_name, buffer)

        share = nc.files.sharing.create(
            path=file_name,  # path to the folder in your Nextcloud
            share_type=3,  # 3 = public link
        )
        return {"url": share.url}
