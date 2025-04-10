FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y make

# Copy the application into the container.
COPY . /app

# Install the application dependencies.
WORKDIR /app
RUN uv sync --frozen --no-cache

ENTRYPOINT ["uv", "run", "python", "-m", "uvicorn", "src.agent.entrypoints.app:app"]
CMD ["--host", "0.0.0.0", "--port", "5050"]
