# Design decisions

Every non-obvious choice in the project is recorded here, in the order it was made. Each entry follows
the same shape: **Context → Options considered → Decision → Why → Trade-offs → How I'd revisit it.**

---

## D1. uv and `pyproject.toml` instead of pip or conda

**Context.** The project depends on about twenty libraries (PyTorch, MONAI, XGBoost, FastAPI and
others). Results must be reproducible on another machine months from now, which means the exact same
versions.

**Options considered.**

| Option | Reproducible versions | Speed | Notes |
|---|---|---|---|
| `pip` + `requirements.txt` | Only if I freeze by hand | Slow | No lockfile for sub-dependencies unless I add pip-tools |
| conda / mamba | Yes, with an environment file | Slow to solve | Heavy; mixes conda and pip packages, which breaks in subtle ways |
| **uv** | Yes, `uv.lock` pins every sub-dependency | Very fast | One tool installs Python, creates the venv and installs packages |

**Decision.** uv, with every direct dependency pinned to an exact version (`==`) in `pyproject.toml`
and every indirect dependency pinned in `uv.lock`.

**Why.** `make setup` on a fresh clone produces the same environment I used, in under a minute. A test
(`tests/test_environment.py`) fails if anyone adds a dependency without an exact pin.

**Trade-offs.** Exact pins mean I don't get bug fixes automatically; I have to bump versions on
purpose. uv is younger than pip and conda, so some readers may not have it installed yet.

**How I'd revisit it.** If a library I need only ships through conda (some GPU tooling does), I would
reconsider. So far nothing does.

---

## D2. A project-local virtual environment, enforced

**Context.** This Mac also has Anaconda, whose Python 3.10 is first on the `PATH`. Typing `python`
would pick it up and silently use whatever packages happen to be installed there.

**Options considered.** Trust everyone to activate the right environment; or make the tools refuse to
run anywhere else.

**Decision.** All Python runs inside `.venv/` in the project folder, created by uv. The Makefile runs
everything through `uv run`, and `tests/conftest.py` stops the test suite with a clear message if it is
started outside a virtual environment.

**Why.** A test run against the wrong packages proves nothing. Failing loudly is better than passing
for the wrong reason.

**Trade-offs.** One extra rule to learn: use `uv run …` or `source .venv/bin/activate`.

**How I'd revisit it.** I wouldn't. The only thing that might change is the folder name if the project
moves into Docker.

---

## D3. Python 3.11

**Context.** PyTorch, MONAI, XGBoost and SHAP each support a range of Python versions, and the ranges
don't always line up on the newest release.

**Decision.** Python 3.11, installed by uv (`.python-version` pins it).

**Why.** Every library in the stack ships prebuilt Apple Silicon wheels for 3.11, so nothing has to
compile from source. It also has `tomllib` built in, which the environment test uses.

**Trade-offs.** I miss small speed and typing improvements in newer versions.

**How I'd revisit it.** Move up once MONAI and SHAP both officially support a newer version.

---

## D4. XGBoost needs a system OpenMP library on macOS

**Context.** XGBoost's macOS wheel links against `libomp`, the OpenMP runtime used for parallel tree
building. It is a C library, not a Python package, so it cannot live inside `.venv/`. Without it,
`import xgboost` fails.

**Decision.** Install it once with Homebrew: `brew install libomp`. The README's setup section says so.

**Trade-offs.** This is the one dependency outside the virtual environment. On Linux it is usually
already present (`libgomp`), so only Mac users hit it.

---

## D5. No patient data and no model weights in git

**Context.** ACDC is licensed CC BY-NC-SA 4.0 and requires registration to download. Model weights are
tens to hundreds of megabytes each.

**Options considered.** Commit them (simple, but breaks the licence terms and bloats the repository
forever); Git LFS (still redistributes the data); download on demand.

**Decision.** `.gitignore` blocks `data/`, `runs/`, NIfTI files and weight files. Data is fetched by
`scripts/download_data.py` (Phase 1). Trained weights go to the Hugging Face Hub and are downloaded on
first run (Phase 6).

**Why.** It respects the dataset's licence and keeps `git clone` fast.

**Trade-offs.** A fresh clone can't do anything useful until the download step runs.

---

## D6. MIT licence for the code

**Context.** The code should be easy for others to read and reuse; the data has its own non-commercial
licence that I can't change.

**Decision.** MIT for the code. The licence file and README both say the ACDC data is not covered by it.

**Trade-offs.** Anyone can reuse the code commercially. That's fine for a research project; the data
licence still limits what they can do with ACDC itself.
