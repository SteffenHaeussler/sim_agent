from unittest.mock import Mock

import httpx


def conversion_mock_response(*args, **kwargs):
    mock_resp = Mock()
    url_called = args[0]

    try:
        id_from_url = url_called.split("/")[-1]
    except IndexError:
        id_from_url = "unknown_id_from_url"

    if id_from_url == "raise_error":
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = httpx.RequestError(
            "Simulated network problem"
        )
    else:
        mock_resp.status_code = 200

        mock_resp.json.return_value = str(id_from_url)

    return mock_resp


def information_mock_response(*args, **kwargs):
    mock_resp = Mock()
    url_called = args[0]

    body = {
        "parent_id": "parent_id",
        "id": None,
        "name": "test",
        "tag": "test",
        "asset_type": "instrument",
        "unit": "°C",
        "description": "indicator",
        "type": "indicator",
        "range": ["-100", "100"],
    }

    try:
        id_from_url = url_called.split("/")[-1]
    except IndexError:
        id_from_url = "unknown_name_from_url"

    if id_from_url == "raise_error":
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = httpx.RequestError(
            "Simulated network problem"
        )
    else:
        mock_resp.status_code = 200

        body["id"] = id_from_url

        mock_resp.json.return_value = body

    return mock_resp


def neighbor_mock_response(*args, **kwargs):
    mock_resp = Mock()
    url_called = args[0]

    try:
        id_from_url = url_called.split("/")[-1]
    except IndexError:
        id_from_url = "unknown_id_from_url"

    if id_from_url == "raise_error":
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = httpx.RequestError(
            "Simulated network problem"
        )
    else:
        mock_resp.status_code = 200

        mock_resp.json.return_value = str(id_from_url) + "_neighbor"

    return mock_resp


def data_mock_response(*args, **kwargs):
    test_result = [
        {
            "asset_id": "1",
            "timestamp": "2025-04-01T00:00:00",
            "value": 5.34,
            "pk_id": "2",
        },
        {
            "asset_id": "1",
            "timestamp": "2025-04-01T00:01:00",
            "value": 2.46,
            "pk_id": "2",
        },
    ]

    mock_resp = Mock()
    url_called = args[0]

    try:
        id_from_url = url_called.split("/")[-1]
    except IndexError:
        id_from_url = "unknown_id_from_url"

    if id_from_url == "raise_error":
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = httpx.RequestError(
            "Simulated network problem"
        )
    else:
        mock_resp.status_code = 200

        if kwargs.get("params").get("last_value"):
            mock_resp.json.return_value = [test_result[-1]]
        else:
            mock_resp.json.return_value = test_result

    return mock_resp
