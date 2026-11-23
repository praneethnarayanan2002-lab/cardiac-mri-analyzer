# Every Python command goes through `uv run`, which uses the project's .venv/ and never a global Python.
# Targets for later phases (eval, api, ui) are added in the phase that builds them.

.PHONY: setup data data-lite explore train eval-val test lint format typecheck notebook clean

setup:
	uv python install 3.11
	uv sync
	@uv run python -c "import sys; print('virtual environment:', sys.prefix)"

data:
	uv run python scripts/download_data.py
	uv run python scripts/make_splits.py

# Same as `data` but skips the 4D cine files (~1 GB); enough for training and evaluation.
data-lite:
	uv run python scripts/download_data.py --no-cine
	uv run python scripts/make_splits.py

explore:
	cd notebooks && uv run jupyter nbconvert --to notebook --execute --inplace 01_data_exploration.ipynb

train:
	uv run python -m cardiac.train --config configs/unet_scratch.yaml

# Validation-set evaluation of one run, plus its figures: make eval-val RUN=runs/unet_scratch/<folder>
eval-val:
	uv run python -m cardiac.evaluate --run $(RUN) --split val
	uv run python scripts/plot_run.py --run $(RUN)

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
