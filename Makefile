export COMPOSE_DOCKER_CLI_BUILD=1
export DOCKER_BUILDKIT=1

all: down build up test

dev:
	uv run python -m uvicorn src.agent.entrypoints.app:app --host 0.0.0.0 --port 5055 --workers 1 --log-level debug
prod:
	uv run python -m uvicorn src.agent.entrypoints.app:app --host 0.0.0.0 --port 5055 --workers 2 --log-level error
DEV: dev
PROD:prod
run:
	@export PYTHONPATH=. && uv run src/agent/entrypoints/main.py --q "$(Q)"


build:
	docker compose build

up:
	docker compose up

down:
	docker compose down --remove-orphans


test:
	uv run python -m pytest tests/ -s -v --envfile=.env.tests

coverage:
	uv run python -m pytest tests/ -s -v --cov=src --cov-report=term-missing

eval_e2e:
	uv run python -m pytest tests/eval/test_e2e.py -s -v --envfile=.env

eval_ir:
	uv run python -m pytest tests/eval/test_ir.py -s -v --envfile=.env

eval: eval_e2e eval_ir
