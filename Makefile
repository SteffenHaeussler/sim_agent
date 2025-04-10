api:
	@export PYTHONPATH=. && uv run src/agent/entrypoints/app.py
run:
	@export PYTHONPATH=. && uv run src/agent/entrypoints/main.py --q "$(Q)"
