from unittest.mock import Mock

import httpx


def dynamic_mock_response(*args, **kwargs):
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
