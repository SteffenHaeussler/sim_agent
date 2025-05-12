import unittest
from unittest.mock import patch

from src.agent.adapters.tools import GetData
from tests.mock_object import data_mock_response


class TestGetNeighbors(unittest.TestCase):
    @patch("httpx.get")
    def test_get_data(self, mock_httpx_get):
        ids = [12, "test", None]
        params = {"base_url": "http://mockapi.com"}

        mock_httpx_get.side_effect = data_mock_response

        data = GetData(**params)

        result = data.forward(asset_ids=ids)

        self.assertCountEqual(result["data"].columns, ["1", "12", "test"])
        self.assertEqual(mock_httpx_get.call_count, 2)

    @patch("httpx.get")
    def test_no_id(self, mock_httpx_get):
        mock_httpx_get.side_effect = data_mock_response
        ids = [None]
        params = {"base_url": "mock"}

        mock_httpx_get.return_value = None

        data = GetData(**params)

        result = data.forward(asset_ids=ids)

        self.assertCountEqual(result["asset_ids"], [])
        self.assertEqual(mock_httpx_get.call_count, 0)

    @patch("httpx.get")
    def test_raises_exception(self, mock_httpx_get):
        mock_httpx_get.side_effect = data_mock_response

        ids = ["raise_error"]
        params = {"base_url": "http://mockapi.com"}
        data = GetData(**params)

        result = data.forward(asset_ids=ids)
        self.assertCountEqual(result["asset_ids"], [])
        self.assertEqual(mock_httpx_get.call_count, 1)
