# Every Python command goes through `uv run`, which uses the project's .venv/ and never a global Python.
# Targets for later phases (data, train, eval, api, ui) are added in the phase that builds them.

.PHONY: setup test lint format typecheck notebook clean

setup:
	uv python install 3.11
	uv sync
	@uv run python -c "import sys; print('virtual environment:', sys.prefix)"

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy

notebook:
	uv run jupyter lab

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -not -path "./.venv/*" -exec rm -rf {} +
