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

---

## D10. Preprocessing: 1 mm, 192 × 192, clip and z-score per volume

**Context.** The plan said resample in-plane to 1.25 mm and crop to 224 × 224. That was written for the
official data, where spacing varies. The mirror is already 1 mm and 192 × 192 (D9).

**Options considered.** Resample the mirror to 1.25 mm (blurs it for no gain and shrinks the heart to
~154 px, then pads to 224 with empty space); keep 1 mm and 192 × 192.

**Decision.** Keep 1 mm and 192 × 192. Intensities are clipped to each volume's 1st–99th percentile
and then z-scored (mean 0, standard deviation 1) per volume.

**Why.** No resampling means no interpolation blur. 192 is divisible by 16 (MONAI U-Net, four stride-2
steps) and by 32 (the ResNet-34 encoder in Phase 3). MRI intensities have no physical unit (unlike CT's
Hounsfield units), so the only meaningful scale is relative to the same scan; z-scoring per volume
captures that, and clipping first stops a few very bright fat pixels from compressing everything else.

**Trade-offs.** The resampling and crop code still exists and is tested (`prepare` / `restore`) because
uploaded scans in the app will have other sizes. On the mirror it is simply a no-op.

---

## D11. MONAI instead of plain PyTorch

**Context.** A U-Net, a Dice loss, medical augmentations and Hausdorff distances can all be written by
hand in PyTorch.

**Decision.** Use MONAI for the network (`UNet`), the loss (`DiceCELoss`), the augmentations and the
`DataLoader`. Write the evaluation metrics myself (D16).

**Why.** MONAI's implementations are tested, widely used in published work, and citable. Writing my own
U-Net would teach a lot but add risk without improving results. The augmentations handle image/label
pairs correctly (nearest-neighbour for labels) out of the box. MONAI's `DataLoader` also gives every
worker process its own random seed, which plain PyTorch does not do for MONAI's transforms.

**Trade-offs.** One more large dependency, and MONAI's U-Net differs slightly from the original paper
(residual units, downsampling in the first block). `ml_basics.md` §7 draws the real one.

---

## D12. Dice + cross-entropy loss

**Options considered.**

| Loss | Handles small structures | Stable early in training | Notes |
|---|---|---|---|
| Cross-entropy only | Poorly: background dominates (88% of voxels) | Yes | Could add class weights, but they need tuning |
| Dice only | Yes | Gradients can be erratic when overlap is ~0 | |
| Focal | Yes (down-weights easy pixels) | Yes | One more hyperparameter (γ) |
| **Dice + CE** | Yes | Yes | The default in nnU-Net and most strong ACDC results |

**Decision.** `DiceCELoss` with softmax, equal weights, background included in the Dice term.

**Why.** It is the most common choice for exactly this problem, with no extra knobs to tune on 20
validation patients.

**How I'd revisit it.** If the RV (the hardest structure) lags far behind, try weighting its Dice term.

---

## D13. Instance norm instead of batch norm

**Context.** Normalisation layers rescale activations inside the network to keep training stable.
Batch norm uses statistics across the 16 slices in a batch; instance norm uses statistics within each
slice.

**Decision.** Instance norm.

**Why.** With batch 16 drawn from different patients and scanners, batch statistics are noisy, and at
inference (often one patient at a time) they are replaced by running averages that may not match. Instance
norm behaves identically in training and inference and is the nnU-Net default for MRI.

**Trade-offs.** It discards each slice's overall contrast level, which carries a little information. For
MRI, where absolute intensity has no meaning (D10), that is mostly a benefit.

---

## D14. AdamW, warm-up, cosine decay, early stopping on 3D validation Dice

**Decision.** AdamW (lr 1e-3, weight decay 1e-4), 5 warm-up epochs, then cosine decay over a maximum of
150 epochs. After every epoch, compute the mean 3D Dice over the 20 validation patients (both phases,
three structures); keep the best checkpoint; stop after 20 epochs without improvement.

**Why each piece.** AdamW adapts the step size per weight and works well with little tuning. Warm-up
avoids big, destabilising updates while the weights are still random. Cosine decay makes the late
epochs small refinements. Selecting on **3D per-patient** Dice matches exactly how results are reported;
per-slice Dice would over-weight the many easy mid-ventricular slices.

**Trade-offs.** Early stopping means the cosine schedule may not reach zero. Validation runs without
post-processing, to stay fast, so the selected checkpoint is the best *raw* model.

**What the first run showed.** The seed-0 run peaked at epoch 50 and early-stopped at epoch 70, when the
learning rate was still 58% of its peak: the cosine curve was laid out over 150 epochs and the slow,
small-step end of it never happened. From epoch 35 onwards validation Dice stayed between 0.884 and 0.896,
so the model wasn't still learning much, but a schedule that actually anneals might squeeze
out a little more. If I change it, it will be by setting `max_epochs` closer to where runs actually stop
(around 80), decided before looking at any test result, and applied to every model in the comparison.

---

## D15. Post-processing: remove small components, fill cavity holes

**Context.** A pixel-wise model sometimes predicts small isolated blobs far from the heart, or leaves
pinholes inside a cavity.

**Decision.** After prediction: for each structure, find its 3D connected components (26-connectivity,
so slices 10 mm apart that overlap diagonally still connect) and delete any component smaller than 10% of
that structure's largest one. Then fill background holes inside the LV and RV slice by slice. The
myocardium is never hole-filled, because its hole *is* the LV.

**Why.** Each structure is one connected organ, so anything else is certainly an error. Both operations
are cheap and easy to explain.

**What I tried first, and why it changed.** The first version kept only the *single largest*
component of each structure. On the validation set it made the RV worse: mean RV Dice fell from 0.887
(raw) to 0.883 and RV HD95 rose from 5.13 to 5.82 mm. The worst case was patient081 at end-systole,
where RV Dice dropped from 0.818 to 0.709. Looking at that case, the model's RV came out in two real
pieces (slices 0–4: 8,394 voxels, and slices 5–7: 2,591 voxels, 84–87% of each inside the expert RV) plus
one stray voxel. The thin RV crescent moves between slices 10 mm apart, so two parts of a real RV need
not touch. "Largest only" deleted the genuine 2,591-voxel piece.

The aim of post-processing is to remove *small stray* blobs, so the rule became "drop components
smaller than 10% of the largest". The 10% was chosen as a round number, not tuned. With it, on
validation (seed 0, 20 patients × ED/ES):

| | RV Dice | MYO Dice | LV Dice | RV HD95 | MYO HD95 | LV HD95 |
|---|---|---|---|---|---|---|
| raw | 0.887 | 0.867 | 0.936 | 5.13 mm | 3.07 mm | 3.18 mm |
| keep largest only (first try) | 0.883 | 0.867 | 0.937 | 5.82 mm | 3.06 mm | 3.13 mm |
| **drop < 10% of largest** | **0.888** | **0.867** | **0.937** | **5.08 mm** | **3.06 mm** | **3.13 mm** |

The effect is small, which is itself a result: this U-Net rarely produces stray blobs. The choice was made
on the validation set, so the test set (Phase 3) is the honest check of it.

**Trade-offs.** A real piece smaller than 10% of the main one would still be deleted, and a stray blob
bigger than 10% would survive. Evaluation always reports raw and post-processed numbers side by side.

---

## D16. HD95 computed by our own code, pooled over both directions

**Context.** Hausdorff distance measures the worst boundary error; HD95 takes the 95th percentile so a
single stray voxel doesn't dominate. There are two common definitions: pool the surface distances from
both directions and take one percentile (as `medpy`, used by most ACDC papers, does), or take the
percentile of each direction and report the larger (as MONAI does).

**Decision.** Pooled, implemented in `metrics.hd95()` with `scipy.ndimage.distance_transform_edt` and
the real voxel spacing (`sampling=spacing_mm`). If either mask is empty, HD95 is undefined: it is
recorded as NaN and the count of missing values is reported alongside the mean.

**Why.** It is about 15 lines, easy to test (`tests/test_metrics.py` checks that a one-slice shift is
1 mm with 1 mm slices and 10 mm with 10 mm slices), and comparable with the ACDC literature.

**Trade-offs.** The two definitions give slightly different numbers; mixing them across papers would be
misleading, so this one is stated wherever HD95 is reported.

---

## D17. Augmentation includes flips

**Context.** Flipping a short-axis slice produces a mirror-image heart (RV on the wrong side), which
never occurs in a real scan.

**Decision.** Keep the flip (50%), as in the plan.

**Why.** The uncertainty method in Phase 5 averages predictions over flipped and rotated copies of each
slice (test-time augmentation). If the network had never seen a flipped heart, those copies would be
out of distribution and the averaged prediction would be worse for a reason unrelated to real
uncertainty. Training with flips makes the network handle both orientations.

**Trade-offs.** The network can no longer use "the RV is on the left" as a cue, and must tell RV from LV
by shape (the LV is round with a thick wall; the RV is a thin crescent). If the RV score turns out to be
weak, this is the first thing I'd test, by training once without flips.

---

## D18. ResNet-34 as the pretrained encoder, trained under the same rules as the scratch U-Net

**Context.** RQ2 asks whether starting from ImageNet weights helps on a small medical dataset. To answer
it, the pretrained model must differ from the scratch model in that one respect as far as possible.

**Options considered.** ResNet-18 (smaller, weaker features), **ResNet-34**, ResNet-50 and larger
(bottleneck blocks, 2–4× the encoder size), EfficientNet or ConvNeXt encoders, a transformer encoder.

**Decision.** `smp.Unet("resnet34", encoder_weights="imagenet", in_channels=1, classes=4)`. Same data,
split, augmentation, loss, batch size, schedule shape, early-stopping rule and three seeds as the scratch
U-Net. The only training difference is the learning rate (3e-4 instead of 1e-3): large early updates
would overwrite the pretrained features before the decoder has learned to use them.

**Why.** ResNet-34 is the most common pretrained encoder in medical segmentation papers. It is
well understood, and at 24 M parameters it trains on the M4 in about the same time as the scratch
model. Bigger encoders mostly add capacity, which 80 training patients can't use and would make the
comparison about size instead of pretraining.

**Trade-offs, stated honestly.** The two models are *not* identical apart from the weights. The
pretrained one has 3.8× the parameters, one more downsampling level, and batch norm instead of instance
norm (that's how ResNet is built). So a difference in results is "ImageNet-pretrained ResNet-34 U-Net vs.
scratch MONAI U-Net", not "pretraining alone". A cleaner test would also train the same ResNet-34 U-Net
from random weights. I left it out to keep to the plan's three segmenters, and list it as next work.

**Result (test set, 3 seeds each).** Mean Dice 0.903 ± 0.002 pretrained vs. 0.902 ± 0.003 scratch. Per
structure the differences are at most 0.004 Dice, inside the seed-to-seed spread, and the paired
Wilcoxon tests are not significant (p = 0.44 RV, 0.27 MYO, 0.24 LV; pretrained better for 28–30 of 50
patients). On this dataset, with this much augmentation and training, ImageNet pretraining did not
measurably improve overlap. HD95 is lower with the pretrained encoder (LV 2.98 vs 3.67 mm), but for the
LV only 26 of 50 patients improve: the mean moves because a few patients with large boundary errors get
better, not because typical patients do (see D20 for why that test is labelled exploratory).

The plan said smp *averages* the RGB weights of the first convolution for 1-channel input. Reading the
library code (`patch_first_conv`) shows it **sums** them. Summing keeps the response to a grey image the
same as to an RGB photo of equal brightness, which is the more sensible choice anyway.

---

## D19. MedSAM with oracle boxes is an upper bound, not a fair comparison

**Context.** MedSAM needs a prompt for every structure on every slice. In real use, a person or another
model would have to draw those boxes.

**Decision.** Follow the plan: build each box from the **expert** label (tight bounding box + 5 px),
then measure sensitivity by moving each box edge by up to 0, 5, 10 and 20 px (1 px = 1 mm here), with a
fixed random seed. Paint masks in the order RV → MYO → LV so the LV wins where masks overlap.

**Why this is an upper bound.** The oracle box gives MedSAM three things the U-Nets have to work out
themselves: which slices contain each structure (no box means "not here"), roughly where each structure
is, and how big it is. A U-Net that is shown nothing still has to find all of that. So MedSAM's score
here is the best it could do with a perfect assistant, and the jitter curve shows how quickly that
falls as the assistant gets sloppier.

**What happened with the myocardium.** On the first validation patient, a box around the myocardium made
MedSAM outline the bright **LV blood pool inside it**: 3,927 of its 4,012 mask pixels were LV cavity, 85
were myocardium. The LV mask then paints over it, and almost no myocardium is left. This isn't a bug: the
same code gives sensible RV and LV masks. A box can't say "the ring, not the thing in the middle", and
SAM-style models tend to pick the most obvious object inside a box. I kept the plan's protocol and report
the result as it is. Better prompting (a positive point on the muscle and a negative point in the
cavity, or "epicardium box minus LV mask") might fix it. That would be a different method, so it goes
under next steps rather than quietly into this comparison.

**On the test set** the myocardium failure is general but uneven. Myocardium Dice with exact boxes is
0.471 overall, **0.309 at end-diastole and 0.633 at end-systole**, and by group from 0.312 (DCM) to
0.707 (HCM). The pattern fits the explanation: when the wall is thick compared with the cavity (a
contracted heart, or hypertrophic cardiomyopathy), the ring is the most obvious object in the box; when
the cavity dominates (a relaxed or dilated heart), MedSAM outlines the cavity. For RV and LV, exact boxes
give 0.877 and 0.896 Dice, below both trained U-Nets, which are given no prompt at all.

**Jitter.** Moving each box edge by up to 5 px barely matters (RV 0.877 → 0.869, LV 0.896 → 0.892);
up to 10 px costs a few points; up to 20 px drops RV to 0.681 and LV to 0.743. A quickly drawn human box
is probably within 5–10 px, so in that range MedSAM's RV and LV would stay usable, but not better than
a U-Net trained on 80 patients.

**Trade-offs.** Results here say nothing about MedSAM with automatic prompts, and nothing about MedSAM
fine-tuned on ACDC; both would likely change the picture.

---

## D20. Three seeds, the same stopping rule for every model, and when the test set is opened

**Context.** One training run can be lucky or unlucky: random initial weights, random augmentation and
random batch order all change the result. Two models whose scores differ by less than this
run-to-run noise are not really different.

**Decision.** Train each U-Net with seeds 0, 1 and 2, and report mean ± standard deviation over the three
seeds. All six runs use the same rule: up to 150 epochs, stop after 20 without improvement. The test set
is evaluated once per trained model, after all six runs exist and nothing more will be changed.

**About the learning-rate schedule (D14).** After the first run I noted that early stopping ends runs
before the cosine schedule gets small, and considered setting `max_epochs` to about 80. I decided not
to. Validation Dice had already been flat for 35 epochs, so the expected gain is small. Changing it
would also mean retraining seed 0, and both models would be affected the same way, so it doesn't bias
the comparison. It stays a candidate for future work.

**What the six runs looked like.** Best validation epochs were 50, 80 and 88 (scratch) and 96, 68
and 34 (pretrained); runs lasted 55–117 epochs. So the stopping point varies a lot from seed to seed,
which is one more reason not to trust a single run. Test-set mean Dice per seed ranged 0.899–0.904
(scratch) and 0.901–0.905 (pretrained).

**Statistics.** Seed-level mean ± std shows training noise. To compare scratch and pretrained on the same
patients, I average each patient's Dice over seeds and over ED/ES (one number per patient, 50 pairs) and
use a paired Wilcoxon signed-rank test, which doesn't assume the differences are normally distributed.

Dice was the planned comparison. After it came out not significant, I also ran the same test on HD95,
because the HD95 means looked different. That second test was chosen after seeing data, so
`results/segmentation.json` marks it `"planned": false` and I treat it as exploratory: RV p = 0.018,
MYO p = 0.20, LV p = 0.07. With three structures tested, a Bonferroni-corrected threshold is
0.05 / 3 ≈ 0.017, and none of them pass it.

---

## D21. The reference for clinical numbers is our own formulas applied to the expert masks

**Context.** RQ4 asks whether the clinical numbers from predicted masks are usable. That needs a
reference value per patient. Three candidates exist: values reported by the hospital, the `edv`, `esv`
and `ef` columns in the mirror's CSV files, or the same formulas run on the expert masks.

**Options considered.**

| Reference | Problem |
|---|---|
| Hospital-reported values | not released with ACDC |
| The mirror's CSV columns | `edv` is 4–9 times the LV volume of the expert masks (D9); nothing says how they were made |
| Our formulas on the expert masks | measures only segmentation error, which is what RQ4 is about |

**Decision.** The reference is `measure_patient()` applied to the expert masks, with the same spacing,
the same code and the same conventions (mass at ED, Mosteller BSA, thickness per slice). The
difference between model and expert then comes from the masks alone, not from two different ways of
measuring.

**Trade-offs.** Our "expert" numbers are not what a cardiologist would have written in the report.
They inherit the mirror's resampling: 23 patients had their slices resampled through-plane (D9), which
changes their volumes slightly compared with the original scans. Model and reference are on the same
grid, so the comparison is fair, but these absolute values should not be compared with numbers
published on the original ACDC files.

**How I'd revisit it.** With the official ACDC download (D9), run the same code on the original
spacing and compare both sets of expert numbers; the difference would show what the mirror's
resampling costs.

---

## D22. Wall thickness: outer border to nearest cavity pixel, per slice, mean and max at ED

**Context.** The plan asks for wall thickness "per slice, distance from inner to outer myocardial
border (distance transform)", reported as mean and maximum at end-diastole.

**Options considered.**

| Method | How | Trade-off |
|---|---|---|
| Rays from the cavity centre | cast lines outwards, measure the muscle crossed | breaks for non-round cavities (the RV insertion, basal slices) |
| Skeleton × 2 | thin the ring to its centre line, double the distance to the edge there | sensitive to small bumps, which add skeleton branches |
| **Outer border → nearest cavity pixel** | distance transform of "not cavity", read at each outer-border pixel | simple, uses real spacing, no centre needed; overestimates a little where the wall is cut obliquely |
| AHA 16/17-segment model | thickness per named wall segment | needs the RV insertion points to orient the segments: a separate piece of work |

**Decision.** Use the outer-border method on every ED slice that contains both LV cavity and
myocardium, with the in-plane spacing of the scan. The outer border is every myocardium pixel that
touches something other than myocardium or cavity, so the septum is measured against the RV. Report
the mean and the maximum of all samples in the heart.

**Why.** It follows the plan's wording directly, needs no landmarks, and is checked by a unit test on
a synthetic ring of known thickness (`tests/test_clinical.py`).

**Trade-offs.** The maximum is one pixel's measurement and is noisy. On basal slices, where the ring is
open towards the outflow tract, or where the slice cuts the wall at an angle, it reads thicker than the
wall really is. The mean, over hundreds to thousands of samples per heart, is much steadier. Segment-level
thickness would be closer to clinical practice; it is listed as future work.

**What the test set showed.** For the pretrained U-Net, mean thickness agrees closely with the expert
(bias +0.45 mm, limits of agreement −0.42 to +1.33 mm, r = 0.975); maximum thickness much less so
(limits −5.3 to +5.7 mm, r = 0.874), as expected from a one-pixel statistic.

**How I'd revisit it.** Report a high percentile (say the 95th) instead of the maximum, or move to the
16-segment model once RV insertion points are detected.

---

## D23. Clinical agreement on the test set, from the same frozen models

**Context.** Phase 3 already evaluated every model once on the test set. RQ4 needs the predicted masks
again to measure them, but the masks weren't saved.

**Decision.** `cardiac.measure` recomputes them with the same checkpoint, the same deterministic
inference and the same post-processing, then measures them. Nothing about the models or the
measurement code was chosen after looking at test results: the measurement code was finished and run
on the validation set first (`results/clinical_val.json`), and the test run used it unchanged.

To prove the recomputed masks are the ones Phase 3 scored, `cardiac.measure` recomputes every Dice and
compares it with the Phase 3 CSV. The largest difference for every run is stored in its
`clinical_<run>.json` as `max_dice_difference_vs_evaluation`; it is below 0.0001 in every case, which is
the rounding of the 4-decimal CSV, not a different mask.

**Statistics.** Bland-Altman for each U-Net is computed per seed; `results/clinical.json` reports the
mean ± std of bias, limits of agreement, MAE and Pearson r over the three seeds. Plots show seed 0, so
each dot is one real patient rather than an average. MedSAM is reported at each jitter level separately,
like in Phase 3; the plots show exact boxes (jitter 0), its best case.

**How I'd revisit it.** Save the predicted masks once, at evaluation time, so later phases read them
instead of recomputing. I didn't, because Phase 3 was already finished and frozen; the Dice check makes
recomputing safe.

---

## D24. XGBoost and logistic regression for 100 patients, not a random forest or a neural network

**Context.** The diagnosis model sees 18 numbers per patient and has 100 training patients, 20 per
group.

**Options considered.**

| Model | For | Against |
|---|---|---|
| Neural network | flexible | far too little data; nothing to gain on 18 hand-made features |
| Random forest | robust, few settings | usually a little behind boosting on tabular data; SHAP works too |
| **XGBoost, depth 3–4** | strong on small tabular data; handles missing values; exact SHAP | more settings to tune; can overfit 100 patients |
| **Logistic regression** | simplest model that can learn; weights readable | one straight boundary per class |

**Decision.** XGBoost is the main model (the plan's choice), with logistic regression and the rule set
(D25) as baselines. Both learned models are tuned by stratified 5-fold cross-validation on the 100
training patients over the small grids in `configs/classifier.yaml`; the test patients are not used.

**What happened.** Cross-validation chose depth 3, 300 trees, learning rate 0.1 for XGBoost (CV
accuracy 0.91) and C = 100 for logistic regression (0.97). C first landed on the largest value of the
original grid (10), so I added 100 before touching the test set. It landed on the edge again. By then
the test set had been opened, so I left the grid as it was rather than tune after seeing test results.

On the 50 test patients with expert features, logistic regression scored 0.88 and XGBoost 0.86, one
patient apart. With features from the pretrained U-Net, 0.91 ± 0.02 and 0.88 ± 0.02. So on this data
the simple linear model is as good as boosting. That is a common result for small, clean tabular
data, where the classes are separated by a few thresholds that a weighted sum can also draw.

The exception is MedSAM's features, where XGBoost (0.80) holds up far better than logistic regression
(0.64). MedSAM's myocardial mass is about 85 g too low, far outside anything seen in training. A
linear model multiplies that by its weight and extrapolates without limit; a tree only asks "is it
below this threshold?" and gives the same answer for any value past it.

**Trade-offs.** XGBoost stays the model the app explains with SHAP, because it is robust to
out-of-range inputs, which is exactly what an uploaded scan can produce.

**How I'd revisit it.** With more patients, nested cross-validation (tuning inside each fold) would
give an honest estimate of the tuned model's accuracy without a separate test set.

---

## D25. Train on expert features; a hand-written rule set as a baseline

**Context.** RQ5 asks whether the five conditions can be diagnosed from derived measurements. The
classifier could be trained on numbers from the expert masks or from a segmenter's masks.

**Decision.** Train on features from the **expert** masks of the 100 training patients. Test on the
50 test patients twice: with expert features, and with features from each segmenter's predicted masks.

**Why.** Training on expert features gives a model of "what the measurements say about the disease",
independent of any one segmenter. Testing on predicted features then measures how much segmentation
error costs at diagnosis time, which is how the app uses it: an uploaded scan only has predicted
masks.

**The rule set.** `classify.rule_based()` writes the ACDC group definitions as if-then rules with
their published thresholds, untuned: low LV EF with a dilated LV → DCM, low LV EF otherwise → MINF,
large or weak RV → RV, heavy and thick LV → HCM, else NOR. It scores 0.70 on the test set with expert
features. It fails mostly on HCM (F1 0.18): the definition needs a mass index above 110 g/m², and by
our measurement most HCM patients are below it (median 93.9 g/m² on the training patients). That is
a difference in how mass is measured (these masks, the mirror's 10 mm slices), not a
wrong definition, and it is a good argument for learning thresholds from data measured the same way
as the test data.

**Trade-offs.** A model trained on expert features has never seen segmentation errors, so it is not
taught to be wary of, say, a missed basal slice. Training on predicted features of the training
patients would need cross-validated predictions (a segmenter's predictions on its own training
patients are too good), which is more work than this project's scope.

---

## D26. TTA views: small rotations and the training flip, not 90° rotations

**Context.** The plan specified 8 test-time views: rotations by 0, 90, 180 and 270 degrees, each with
and without a flip, chosen because they can be undone exactly.

**What happened.** On validation patients, single views rotated by 180° scored 0.51–0.61 mean Dice
instead of about 0.90, and 90° or 270° failed for some patients. The flipped view without rotation
scored like the original. The U-Net was trained with rotations of at most 15° and a flip along one
axis (D17); a 180° rotation puts the RV on the opposite side of the LV from where it was in every
training image, and the model labels it as something else. Averaging those views lowered Dice from
0.92 to 0.67 on one validation patient, and the "doubt" they produced measured how unusual the
rotation was, not how hard the patient was.

**Decision.** Use views the model has seen the like of: rotations by −10°, −5°, 0°, +5° and +10°,
each with and without the training flip: 10 views. The flip and the 0° view are undone exactly;
the rotations are undone with linear interpolation, which is close to exact away from the corners
(checked in `tests/test_uncertainty.py`).

**Trade-offs.** The views are less different from each other than 90° turns, so the doubt they
measure is smaller in absolute terms; what matters is whether it ranks patients correctly (RQ6).
Interpolation smooths the probability maps very slightly.

**What it gave.** On the test set (3 seeds): Spearman correlation between doubt and error
0.71 ± 0.01, area under the risk–coverage curve 0.077 against 0.093 for a random order and 0.072 for a
perfect one, and mean Dice 0.907 with TTA against 0.903 from one view. The settings were fixed on
validation (correlation 0.70 ± 0.08) before the test run.

---

## D27. TTA instead of MC dropout or a deep ensemble

**Context.** RQ6 needs a per-patient uncertainty score.

| Method | How | Cost | Why not here |
|---|---|---|---|
| **TTA** | same model, several transformed inputs | ×10 inference, no retraining | (chosen) |
| MC dropout | keep dropout on at test time, run several times | ×N inference | the pretrained U-Net was trained without dropout (`configs/unet_resnet34.yaml`) |
| Deep ensemble | train several models, compare their answers | ×N training and inference | usually the best estimate; we have 3 seeds, which is a small ensemble, but the app would have to run 3 models |

**Decision.** TTA on the pretrained U-Net, as the plan says. It needs no retraining and works on any
model.

**How I'd revisit it.** The three seeds of each U-Net already exist, so a 3-model ensemble is a cheap
comparison to add.

---

## D28. torch and xgboost in one process: start OpenMP single-threaded

**Context.** On macOS, the torch wheel ships its own OpenMP library (`libomp`) and xgboost links the
Homebrew one (D4). When the test suite loaded both, a torch convolution crashed with a segmentation
fault. Small experiments showed it crashes in either import order and even with xgboost on one thread,
but not when the environment variable `OMP_NUM_THREADS=1` is set **before** either library loads.
Torch can then raise its own thread count with `torch.set_num_threads()`; 40 rounds of alternating
convolutions and XGBoost fits ran without a crash.

**Decision.**
- `tests/conftest.py` sets `OMP_NUM_THREADS=1` before anything else is imported.
- `repro.py` imports torch only inside `set_seed()`, so `scripts/run_diagnosis.py` never loads torch.
- The backend (Phase 6) sets the variable at the top of its package, before any import, then sets
  torch's thread count explicitly.

**Trade-offs.** A setting that must come before imports is easy to break by reordering code, so each
place has a comment pointing here. A cleaner fix would be one shared `libomp` for both libraries,
which needs building one of them from source.
