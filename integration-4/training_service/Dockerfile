FROM ghcr.io/astral-sh/uv:latest AS uv
FROM python:3.12-slim

WORKDIR /app
COPY --from=uv /uv /uvx /bin/

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY training_service/ ./
CMD ["uv", "run", "python", "train.py"]