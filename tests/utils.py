import json
from collections import defaultdict
from pathlib import Path


def get_fixtures(current_path, fixtures=None, keys=None):
    if not fixtures:
        fixtures = defaultdict(lambda: {"in": None, "out": None})

    if not keys:
        keys = ["in", "out"]

    for key in keys:
        files_path = Path(current_path) / key

        for filename in files_path.iterdir():
            with open(filename, "r") as f:
                fixtures[filename.stem][key] = json.load(f)

    return fixtures
