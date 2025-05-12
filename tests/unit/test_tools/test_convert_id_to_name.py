import unittest
from unittest.mock import patch

from src.agent.adapters.tools.convert_id_to_name import ConvertIdToName
from tests.mock_object import dynamic_mock_response


class TestConvertIdToName(unittest.TestCase):
    @patch("httpx.get")
    def test_convert_id_to_name(self, mock_httpx_get):
        ids = [12, "test", None]
        params = {"base_url": "http://mockapi.com"}

        mock_httpx_get.side_effect = dynamic_mock_response

        id2name = ConvertIdToName(**params)

        result = id2name.forward(asset_ids=ids)

        self.assertCountEqual(result["names"], ["test", "12"])
        self.assertEqual(mock_httpx_get.call_count, 2)

    @patch("httpx.get")
    def test_no_id(self, mock_httpx_get):
        mock_httpx_get.side_effect = dynamic_mock_response

        ids = [None]
        params = {"base_url": "mock"}

        mock_httpx_get.return_value = None

        id2name = ConvertIdToName(**params)

        result = id2name.forward(asset_ids=ids)

        self.assertCountEqual(result["names"], [])
        self.assertEqual(mock_httpx_get.call_count, 0)

    @patch("httpx.get")
    def test_raises_exception(self, mock_httpx_get):
        mock_httpx_get.side_effect = dynamic_mock_response

        ids = ["raise_error"]
        params = {"base_url": "http://mockapi.com"}
        id2name = ConvertIdToName(**params)

        result = id2name.forward(asset_ids=ids)
        self.assertCountEqual(result["names"], [])
        self.assertEqual(mock_httpx_get.call_count, 1)
