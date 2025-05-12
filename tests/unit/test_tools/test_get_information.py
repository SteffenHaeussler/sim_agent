import unittest
from unittest.mock import patch

from src.agent.adapters.tools import GetInformation
from tests.mock_object import information_mock_response


class TestGetInformation(unittest.TestCase):
    @patch("httpx.get")
    def test_get_information(self, mock_httpx_get):
        ids = [12, "9280dee1-5dbf-45b7-9e29-c805c4555ba6", None]
        params = {"base_url": "http://mockapi.com"}

        mock_httpx_get.side_effect = information_mock_response

        information = GetInformation(**params)

        result = information.forward(asset_ids=ids)
        out_ids = [i["id"] for i in result["assets"]]

        self.assertCountEqual(out_ids, ["9280dee1-5dbf-45b7-9e29-c805c4555ba6", "12"])
        self.assertEqual(mock_httpx_get.call_count, 2)
        self.assertEqual(len(result["assets"]), 2)

    @patch("httpx.get")
    def test_no_id(self, mock_httpx_get):
        mock_httpx_get.side_effect = information_mock_response

        ids = [None]
        params = {"base_url": "mock"}

        mock_httpx_get.return_value = None

        information = GetInformation(**params)

        result = information.forward(asset_ids=ids)

        self.assertCountEqual(result["assets"], [])
        self.assertEqual(mock_httpx_get.call_count, 0)

    @patch("httpx.get")
    def test_raises_exception(self, mock_httpx_get):
        mock_httpx_get.side_effect = information_mock_response

        ids = ["raise_error"]
        params = {"base_url": "http://mockapi.com"}
        information = GetInformation(**params)

        result = information.forward(asset_ids=ids)
        self.assertCountEqual(result["assets"], [])
        self.assertEqual(mock_httpx_get.call_count, 1)
