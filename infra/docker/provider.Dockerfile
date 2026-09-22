FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir uv && uv sync --frozen --extra serve

EXPOSE 8000
