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

---

## D7. 2D slices instead of 3D or 2.5D

**Context.** Each ACDC frame is a stack of about 10 short-axis slices, each 192 × 192 pixels in our data.
Pixels are 1 mm wide, but the gap between slices is 10 mm. The volume is a pile of thin, widely spaced
pancakes, not a cube.

**Options considered.**

| Option | Input to the network | Pros | Cons |
|---|---|---|---|
| **2D** | one slice | Simple; ~1,900 training slices instead of 160 volumes; fits easily in memory | Ignores the slices above and below |
| 2.5D | a slice plus its two neighbours as channels | Some through-plane context, almost free | Neighbours 10 mm away look quite different; edge slices need padding |
| 3D | the whole stack | Uses all context | Only ~10 voxels deep, so 3D kernels barely help; far fewer training samples; much more memory |

**Decision.** A 2D U-Net, applied slice by slice. Predictions are stacked back into 3D for every metric
and clinical number.

**Why.** With a 10 mm gap, a 3×3×3 kernel mixes tissue 20 mm apart, which is more than the thickness of
the heart wall. Most strong ACDC results are 2D or 2.5D. 2D is also the easiest to explain and debug.

**Trade-offs.** The model can't use the slice above to resolve an ambiguous slice, which hurts most at the
base (near the valves) and the apex, where the heart changes shape fastest.

**How I'd revisit it.** If the base and apex slices turn out to be the main source of error, try 2.5D.
That's a small change: three input channels instead of one.

---

## D8. Splitting by patient, stratified by diagnosis group

**Context.** ACDC has 100 official training patients and 50 official test patients, 20 and 10 per
group. I need a validation set for early stopping and tuning that is separate from the test set.

**Options considered.** Split by slice (wrong: two neighbouring slices from one heart are nearly
identical, so the model would be validated on hearts it has trained on); split by patient at random;
split by patient, stratified by group.

**Decision.** `scripts/make_splits.py` writes `data/splits.json`: 80 train and 20 validation patients
drawn from the official training set, with exactly 4 per group in validation, and the official 50 test
patients unchanged. Seed 0. The file holds patient ids only, so it is committed to git even though the
rest of `data/` is not.

**Why.** Stratifying means validation has every disease in the same proportion as the test set; with only
20 validation patients, a random split could easily end up with 1 HCM patient and 7 DCM ones. Keeping
the official test set lets the results be compared with published ACDC numbers.

**Trade-offs.** 20 validation patients is small, so validation Dice will be noisy (one bad patient moves
the mean). Cross-validation over all 100 would be steadier but costs 5× the training time; I use it only
for the small diagnosis classifier in Phase 5.

**How I'd revisit it.** Nothing about the split changes later. `tests/test_splits.py` checks that no
patient id appears in two splits, on synthetic data and on the committed file.

---

## D9. The Hugging Face mirror of ACDC instead of the official download

**Context.** The official ACDC download needs a registration on the challenge website. A preprocessed
copy, `mathpluscode/ACDC`, is on the Hugging Face Hub under the same licence (CC BY-NC-SA 4.0) and
downloads with one function call.

**Options considered.**

| | Official ACDC | HF mirror `mathpluscode/ACDC` |
|---|---|---|
| Access | Registration, manual download | `snapshot_download`, pinned to a commit |
| Spacing | Original, varies per scanner (0.7–1.9 mm in-plane, 5–10 mm between slices) | Resampled to 1 × 1 × 10 mm for everyone |
| Image size | Varies (e.g. 216 × 256) | 192 × 192, cropped around the **expert LV centre** |
| Metadata | `Info.cfg` per patient | `train.csv` / `test.csv` with the same fields plus the original spacing |
| Cine | 4D, with real frame timing | 4D, but the time spacing is a placeholder (1.0) |

**Decision.** Use the mirror, pinned to commit `4a312b6`. `scripts/download_data.py` downloads and
verifies it.

**Why.** It removes the registration step, so anyone can reproduce the project from a fresh clone, and
the download is pinned to an exact version.

**What this changes, and what I gave up.** These all go into the README's limitations section:

1. **Volumes use the mirror's spacing.** Every volume is computed with 1 × 1 × 10 mm voxels
   (0.01 mL each), read from the NIfTI header, never assumed. The mirror's resampling already happened,
   so our numbers can differ slightly from values computed on the original scans.
2. **Through-plane resampling.** 23 of the 150 patients (14 train, 9 test) originally had slices 5–6.5 mm
   apart. The mirror resampled them to 10 mm, which removes real slices and interpolates the rest. I
   would not have done this myself (it invents anatomy between slices), but it is baked into the data.
3. **The crop uses the expert labels.** Every image is centred on the expert's LV centre. The model
   never has to find the heart, only outline it. Our Dice scores are therefore a little optimistic
   compared with a full-field-of-view scan, and the app needs its own centring for uploaded scans.
4. **No real frame rate for cine playback.** The app plays the cine at a fixed speed instead of the
   scan's real timing.
5. **The CSV's `edv` / `esv` / `ef` columns are not used.** Its `edv` is 4–9 times the LV volume I get
   from the expert masks, with a ratio that varies by patient, so it is not LV volume in mL. All
   "expert" reference values in this project are computed from the expert masks by our own code.

**Effect on preprocessing (Phase 2).** The plan was to resample in-plane to 1.25 mm and crop to
224 × 224. The mirror is already at a uniform 1 mm and 192 × 192, so resampling again would only blur
it. Phase 2 keeps 1 mm and 192 × 192. Both U-Nets need sizes divisible by 16 or 32, and 192 is.

**How I'd revisit it.** Download the official data, point `--root` at it, and add an
`Info.cfg` reader next to the CSV reader. The rest of the pipeline only sees `PatientMeta` and `Frame`,
so it wouldn't change.
