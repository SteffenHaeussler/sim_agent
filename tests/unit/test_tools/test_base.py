import unittest

from src.agent.adapters.tools import GetData


class TestBaseTool(unittest.TestCase):
    def test_call_api(self):
        data = GetData()
        self.assertEqual(data.call_api("http://mockapi.com"), [])

    def test_convert_to_iso_format(self):
        base = GetData()

        test_dates = [
            "2025-12-31 23:59:00",
            "2025/12/31 23:59:00",
            "31/12/2025 23:59:00",  # Needs dateutil or specific format for dayfirst
            "12/31/2025 11:59:00 PM",
            "Dec 31, 2025 23:59",
            "2025-12-31T23:59:00",
        ]

        for date_str in test_dates:
            converted = base.convert_to_iso_format(date_str)
            self.assertEqual(converted, "2025-12-31T23:59:00")

        test_dates = [
            "2025-12-31",  # Date only, time will be 00:00:00
            "31-Dec-2025",
            "December 31, 2025",
            "2025-12-31T00:00:00.123456",  # With microseconds
        ]

        for date_str in test_dates:
            converted = base.convert_to_iso_format(date_str)
            self.assertEqual(converted, "2025-12-31T00:00:00")

    def test_fail_convert_to_iso_format(self):
        base = GetData()

        test_dates = [
            "invalid",
        ]

        for date_str in test_dates:
            with self.assertRaises(AttributeError):
                base.convert_to_iso_format(date_str)

        test_dates = [
            12345,
            None,
            True,
            False,
        ]

        for date_str in test_dates:
            with self.assertRaises(ValueError):
                base.convert_to_iso_format(date_str)
