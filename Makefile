export COMPOSE_DOCKER_CLI_BUILD=1
export DOCKER_BUILDKIT=1

all: down build up test

dev:
	uv run python -m uvicorn src.agent.entrypoints.app:app --host 0.0.0.0 --port 5055 --workers 1 --log-level debug
prod:
	uv run python -m uvicorn src.agent.entrypoints.app:app --host 0.0.0.0 --port 5055 --workers 2 --log-level error
run:
	@export PYTHONPATH=. && uv run src/agent/entrypoints/main.py --q "$(Q)"


build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down --remove-orphans


test:
	uv run python -m pytest tests/ -s -v

coverage:
	uv run python -mpytest tests/ -s -v --cov=src --cov-report=term-missing
