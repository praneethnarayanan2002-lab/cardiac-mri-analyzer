# Cardiac MRI Analyzer

A system that takes a cardiac cine-MRI scan, outlines the heart's chambers with deep learning, turns those
outlines into the numbers cardiologists use (ventricle volumes, ejection fraction, heart-muscle mass, wall
thickness), predicts one of five diagnoses from those numbers with an explainable model, and flags the cases
where the model is unsure. It is built on the public ACDC dataset.

> **Research demo, not a medical device.** Nothing here is validated for clinical use.

## Status

The project is being built in phases (see `docs/`). Phases 0–2 are done: setup, data, and a baseline
U-Net trained from scratch. So far the model has only been evaluated on the 20 **validation** patients
(`results/val/`); the 50 test patients stay untouched until all three segmenters are trained (Phase 3), so
there is no results table yet. Every number that appears here later will come from a real run saved in
`results/`.

## Quick start

You need [uv](https://docs.astral.sh/uv/) installed. Everything runs inside a project-local virtual
environment in `.venv/`; nothing is installed into your system or Anaconda Python. On macOS, XGBoost also
needs the OpenMP runtime, which is a system library (`brew install libomp`).

```bash
make setup    # installs Python 3.11 if needed, creates .venv/ and installs every pinned dependency
make data     # downloads ACDC (~1.1 GB), verifies it, prints a summary, writes the patient splits
make test     # runs the test suite inside .venv/
make lint     # ruff (lint + format check) and mypy
```

To use the environment directly, either prefix commands with `uv run` (for example `uv run python`) or
activate it with `source .venv/bin/activate`.

## Project structure

```
configs/     training and classifier configs (YAML)
data/        downloaded data (git-ignored) and splits.json (patient ids only, committed)
notebooks/   data exploration; saves its figures to figures/
scripts/     data download, splits, demo export
src/cardiac/ the research package: data, models, training, metrics, clinical maths, diagnosis
backend/     FastAPI app (inference only, never trains)
frontend/    React + TypeScript app
tests/       pytest suite
results/     metrics from real runs (small JSON files, committed)
figures/     figures used by this README and the docs
docs/        architecture, execution flow, design decisions, ML basics for beginners
```

## Data and licences

Code is MIT-licensed (see `LICENSE`). The ACDC data is **not** in this repository and is under its own
licence, CC BY-NC-SA 4.0 (non-commercial).

**Which copy of ACDC.** I use the preprocessed Hugging Face mirror
[`mathpluscode/ACDC`](https://huggingface.co/datasets/mathpluscode/ACDC), pinned to commit `4a312b6`,
instead of the official download (which needs a registration). The mirror is not the raw data, and that
affects the results:

- Every scan was **resampled to 1 × 1 × 10 mm** voxels. All volumes, masses and distances in this project
  are computed with that resampled spacing (read from each file's header), not the original scanner
  spacing. 23 of 150 patients originally had 5–6.5 mm slice gaps and were resampled through-plane.
- Every image was **cropped to 192 × 192 around the expert's LV centre**, so the models never have to
  locate the heart. Segmentation scores are somewhat optimistic compared with full-field scans.
- The cine has no real frame timing, so playback in the app uses a fixed speed.

The reasoning and the alternatives are in `docs/decisions.md` (D9).

If you use ACDC, cite:
O. Bernard et al., "Deep Learning Techniques for Automatic MRI Cardiac Multi-Structures Segmentation and
Diagnosis: Is the Problem Solved?", *IEEE Transactions on Medical Imaging*, 37(11), 2018.
