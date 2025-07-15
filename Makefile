.PHONY: eval eval_e2e eval_ir eval_tool_agent eval_enhance eval_pre_check eval_post_check eval_sql_e2e eval_sql_stages

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
	@export PYTHONPATH=. && uv run src/agent/entrypoints/main.py --q "$(Q)" --m "$(M)"


build:
	docker compose build

up:
	docker compose up

down:
	docker compose down --remove-orphans


test:
	uv run python -m pytest tests/ -s -v

tests: test

coverage:
	uv run python -m pytest tests/ -s -v --cov=src --cov-report=term-missing

eval_e2e:
	uv run python -m pytest evals/tool_agent/test_e2e.py -s -v

eval_enhance:
	uv run python -m pytest evals/tool_agent/test_enhance.py -s -v

eval_pre_check:
	uv run python -m pytest evals/tool_agent/test_pre_check.py -s -v

eval_post_check:
	uv run python -m pytest evals/tool_agent/test_post_check.py -s -v

eval_ir:
	uv run python -m pytest evals/tool_agent/test_ir.py -s -v

eval_tool_agent:
	uv run python -m pytest evals/tool_agent/test_tool_agent.py -s -v

eval_sql_e2e:
	uv run python -m pytest evals/sql_agent/test_sql_e2e.py -s -v

eval_sql_stages:
	uv run python -m pytest evals/sql_agent/test_sql_stages.py -s -v


eval: eval_e2e eval_ir eval_tool_agent eval_enhance eval_pre_check eval_post_check eval_sql_e2e eval_sql_stages
