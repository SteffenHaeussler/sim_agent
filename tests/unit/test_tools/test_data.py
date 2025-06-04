import unittest
from unittest.mock import patch

import pandas as pd

from src.agent.adapters.tools import CompareData, GetData
from tests.mock_object import data_mock_response


class TestGetData(unittest.TestCase):
    @patch("httpx.get")
    def test_get_data(self, mock_httpx_get):
        ids = [12, "test", None]
        params = {
            "tools_api_base": "http://mockapi.com",
            "tools_api_limit": 100,
        }

        mock_httpx_get.side_effect = data_mock_response

        data = GetData(**params)

        result = data.forward(
            asset_ids=ids,
            start_date="2025-04-01T00:00:00",
            end_date="2025-04-02T00:00:00",
            aggregation="day",
            last_value=False,
        )

        self.assertCountEqual(result["data"].columns, ["12", "test"])
        self.assertEqual(result["data"].shape, (2, 2))
        self.assertEqual(mock_httpx_get.call_count, 2)

    @patch("httpx.get")
    def test_get_last_data(self, mock_httpx_get):
        ids = [12, "test", None]
        params = {
            "tools_api_base": "http://mockapi.com",
            "tools_api_limit": 100,
        }

        mock_httpx_get.side_effect = data_mock_response

        data = GetData(**params)

        result = data.forward(
            asset_ids=ids,
            last_value=True,
        )

        self.assertCountEqual(result["data"].columns, ["12", "test"])
        self.assertEqual(result["data"].shape, (1, 2))
        self.assertEqual(mock_httpx_get.call_count, 2)

    @patch("httpx.get")
    def test_no_id(self, mock_httpx_get):
        mock_httpx_get.side_effect = data_mock_response
        ids = [None]
        params = {
            "tools_api_base": "mock",
            "tools_api_limit": 100,
        }

        mock_httpx_get.return_value = None

        data = GetData(**params)

        result = data.forward(
            asset_ids=ids,
            last_value=True,
        )

        self.assertCountEqual(result["data"], [])
        self.assertEqual(mock_httpx_get.call_count, 0)

    @patch("httpx.get")
    def test_raises_exception(self, mock_httpx_get):
        mock_httpx_get.side_effect = data_mock_response

        ids = ["raise_error"]
        params = {
            "tools_api_base": "http://mockapi.com",
            "tools_api_limit": 100,
        }
        data = GetData(**params)

        result = data.forward(
            asset_ids=ids,
            last_value=True,
        )

        self.assertCountEqual(result["data"], [])
        self.assertEqual(mock_httpx_get.call_count, 1)


class TestMapAggregation(unittest.TestCase):
    def test_map_aggregation(self):
        params = {
            "tools_api_base": "http://mockapi.com",
            "tools_api_limit": 100,
        }
        data = GetData(**params)
        self.assertEqual(data.map_aggregation("day"), "d")
        self.assertEqual(data.map_aggregation("hour"), "h")
        self.assertEqual(data.map_aggregation("minute"), "min")
        self.assertEqual(data.map_aggregation("d"), "d")
        self.assertEqual(data.map_aggregation("h"), "h")
        self.assertEqual(data.map_aggregation("min"), "min")

    def test_map_aggregation_invalid(self):
        params = {
            "tools_api_base": "http://mockapi.com",
            "tools_api_limit": 100,
        }
        data = GetData(**params)
        with self.assertRaises(ValueError):
            data.map_aggregation("invalid")


class TestCompareData(unittest.TestCase):
    def test_compare_no_data(self):
        params = {
            "tools_api_base": "http://mockapi.com",
            "tools_api_limit": 100,
        }
        compare = CompareData(**params)
        out = compare.forward(pd.DataFrame())
        assert len(out["comparison"]) == 0

    def test_compare_data(self):
        _input = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [1, 2, 3],
            }
        )
        params = {
            "tools_api_base": "http://mockapi.com",
            "tools_api_limit": 100,
        }
        compare = CompareData(**params)
        out = compare.forward(_input)
        assert len(out["comparison"]) == 2
        assert len(out["comparison"]["a"]) == 8
        assert len(out["comparison"]["b"]) == 8
