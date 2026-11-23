# Every Python command goes through `uv run`, which uses the project's .venv/ and never a global Python.
# Each target has a comment above it saying what it does and what it needs.

.PHONY: setup data data-lite explore train eval eval-val samples api ui ui-setup demo test lint format \
	typecheck notebook clean

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

# Both U-Nets, three seeds each (about 2.5 hours on an M4). One run: uv run python -m cardiac.train
# --config configs/unet_scratch.yaml --seed 0
train:
	for cfg in unet_scratch unet_resnet34; do \
		for seed in 0 1 2; do \
			uv run python -m cardiac.train --config configs/$$cfg.yaml --seed $$seed || exit 1; \
		done; \
	done

# Test-set evaluation of every trained run and of MedSAM, then clinical numbers, uncertainty and
# diagnosis, each with its summary tables and figures.
# Run once, after all training is finished. Segmentation first, then clinical numbers from the same
# frozen models; cardiac.measure checks that its masks match the ones evaluate scored.
eval:
	for run in runs/unet_scratch/*/ runs/unet_resnet34/*/; do \
		uv run python -m cardiac.evaluate --run "$$run" --split test --confirm-test || exit 1; \
	done
	uv run python -m cardiac.evaluate_medsam --split test --confirm-test
	uv run python scripts/summarize_segmentation.py --split test
	uv run python -m cardiac.measure --expert --split train
	uv run python -m cardiac.measure --expert --split val
	uv run python -m cardiac.measure --expert --split test --confirm-test
	for run in runs/unet_scratch/*/ runs/unet_resnet34/*/; do \
		uv run python -m cardiac.measure --run "$$run" --split test --confirm-test || exit 1; \
	done
	uv run python -m cardiac.measure --medsam --split test --confirm-test
	uv run python scripts/summarize_clinical.py --split test
	for run in runs/unet_resnet34/*/; do \
		uv run python -m cardiac.evaluate_uncertainty --run "$$run" --split test --confirm-test \
			|| exit 1; \
	done
	uv run python scripts/summarize_uncertainty.py --split test
	uv run python scripts/plot_doubt_maps.py --run $(firstword $(wildcard runs/unet_resnet34/*-seed0))
	uv run python scripts/run_diagnosis.py

# Validation-set evaluation of one run, plus its figures: make eval-val RUN=runs/unet_scratch/<folder>
eval-val:
	uv run python -m cardiac.evaluate --run $(RUN) --split val
	uv run python scripts/plot_run.py --run $(RUN)

# Normal ranges from the NOR training patients, then the app's 5 precomputed sample patients
# (data/cache/samples/, git-ignored). Needs the trained runs and `make eval`.
samples:
	uv run python scripts/normal_ranges.py
	PYTHONPATH=. CARDIAC_DEVICE=$(DEVICE) uv run python scripts/cache_sample_cases.py

# The backend on http://localhost:8000 (API docs at /docs). CPU by default; `make api DEVICE=mps`.
DEVICE ?= cpu
api:
	CARDIAC_DEVICE=$(DEVICE) uv run uvicorn backend.app.main:app --port 8000 --reload \
		--reload-dir backend --reload-dir src

# The frontend on http://localhost:5173, talking to `make api` (run both). Needs Node 22+.
ui-setup:
	cd frontend && npm ci

ui:
	cd frontend && npm run dev

# The public static demo: export the samples and metrics to frontend/public/demo/, then build the
# static site into frontend/dist/ (what the GitHub Pages workflow deploys). Needs `make samples`.
demo:
	PYTHONPATH=. uv run python scripts/export_demo.py
	cd frontend && npm run build:static

test:
	uv run pytest
	cd frontend && npm test

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	cd frontend && npm run lint

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
