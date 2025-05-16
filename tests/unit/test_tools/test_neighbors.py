import unittest
from unittest.mock import patch

from src.agent.adapters.tools import GetNeighbors
from tests.mock_object import neighbor_mock_response


class TestGetNeighbors(unittest.TestCase):
    @patch("httpx.get")
    def test_get_neighbors(self, mock_httpx_get):
        ids = [12, "test", None]
        params = {
            "tools_api_base": "http://mockapi.com",
            "tools_api_limit": 100,
        }
        mock_httpx_get.side_effect = neighbor_mock_response

        neighbors = GetNeighbors(**params)

        result = neighbors.forward(asset_ids=ids)

        self.assertCountEqual(result["asset_ids"], ["test_neighbor", "12_neighbor"])
        self.assertEqual(mock_httpx_get.call_count, 2)

    @patch("httpx.get")
    def test_no_id(self, mock_httpx_get):
        mock_httpx_get.side_effect = neighbor_mock_response
        ids = [None]
        params = {
            "tools_api_base": "mock",
            "tools_api_limit": 100,
        }

        mock_httpx_get.return_value = None

        neighbors = GetNeighbors(**params)

        result = neighbors.forward(asset_ids=ids)

        self.assertCountEqual(result["asset_ids"], [])
        self.assertEqual(mock_httpx_get.call_count, 0)

    @patch("httpx.get")
    def test_raises_exception(self, mock_httpx_get):
        mock_httpx_get.side_effect = neighbor_mock_response

        ids = ["raise_error"]
        params = {
            "tools_api_base": "http://mockapi.com",
            "tools_api_limit": 100,
        }
        neighbors = GetNeighbors(**params)

        result = neighbors.forward(asset_ids=ids)
        self.assertCountEqual(result["asset_ids"], [])
        self.assertEqual(mock_httpx_get.call_count, 1)
