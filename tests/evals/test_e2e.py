import json
from pathlib import Path
from time import sleep

import pytest
from fastapi.testclient import TestClient

from src.agent.entrypoints.app import app
from tests.utils import get_fixtures

current_path = Path(__file__).parent

fixtures = get_fixtures(current_path, keys=["e2e"])
results = []

# Create test client
client = TestClient(app)


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
        response = client.get("/answer", params=params)

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
