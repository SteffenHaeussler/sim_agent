from pathlib import Path

import pytest

from src.agent import config
from src.agent.adapters import rag
from tests.utils import get_fixtures

current_path = Path(__file__).parent

fixtures = get_fixtures(Path(__file__).parent, keys=["ir"])
results = []


rag = rag.BaseRAG(config.get_rag_config())


class TestIR:
    @pytest.mark.parametrize(
        "fixture_name, fixture",
        [
            pytest.param(fixture_name, fixture, id=fixture_name)
            for fixture_name, fixture in fixtures.items()
        ],
    )
    def test_ir(self, fixture_name, fixture):
        candidates = []

        question, expected_response = (
            fixture["ir"]["question"],
            fixture["ir"]["response"],
        )

        response = rag.embed(question)
        response = rag.retrieve(response["embedding"])

        for candidate in response["results"]:
            temp = rag.rerank(question, candidate["description"])
            candidate["score"] = temp["score"]
            candidates.append(candidate)

        candidates = sorted(candidates, key=lambda x: -x["score"])

        response = candidates[0]
        response.pop("score")
        if "area" not in expected_response:
            response.pop("area", None)
        if "location" not in expected_response:
            response.pop("location", None)

        assert response == expected_response
