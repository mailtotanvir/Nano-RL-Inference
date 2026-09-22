.DEFAULT_GOAL := help

.PHONY: help install lint typecheck test smoke verify format

help:
	@printf '%s\n' 'Targets: install lint typecheck test smoke verify format'

install:
	uv sync --extra serve --extra dev

lint:
	uv run ruff check .

typecheck:
	uv run mypy src

test:
	uv run pytest -q

smoke:
	uv run pytest tests/unit -q

verify: lint typecheck test
	uv run detect-secrets scan --all-files src tests configs > /dev/null

format:
	uv run ruff format .
