import json
import os
from pathlib import Path
from time import sleep

import httpx
import pytest

from tests.utils import get_fixtures

current_path = Path(__file__).parent

fixtures = get_fixtures(current_path, keys=["e2e"])
results = []


class TestEvalE2E:
    @pytest.mark.parametrize(
        "fixture_name, fixture",
        [
            pytest.param(fixture_name, fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_eval_e2e(self, fixture_name, fixture):
        question, expected_response = (
            fixture["e2e"]["question"],
            fixture["e2e"]["response"],
        )

        params = {"question": question, "q_id": fixture_name}

        response = httpx.get(
            os.getenv("agent_api_base_url"),
            params=params,
            timeout=60,
        )

        report = {
            "question": question,
            "response": response.text,
            "expected_response": expected_response,
        }
        results.append(report)

        with open("report.json", "w") as f:
            json.dump(results, f)

        sleep(60)

        assert response
