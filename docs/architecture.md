# Architecture

What the pieces are and how they connect. `flow.md` follows execution step by step; this file is the
map. Sections are added in the phase that builds each part: the research package and the three
segmenters, clinical numbers, diagnosis and uncertainty (Phases 3–5), the backend in Phase 6, the frontend in Phase 7, deployment in Phase 8.

---

## 1. System context

```mermaid
flowchart LR
    ACDC[("ACDC mirror<br/>Hugging Face Hub<br/>mathpluscode/ACDC")] -->|"scripts/download_data.py"| PKG
    W[("Pretrained weights<br/>HF Hub: smp-hub/resnet34.imagenet,<br/>wanglab/medsam-vit-base")] --> PKG
    subgraph Repo["This repository"]
        PKG["src/cardiac/<br/>research package"] --> RES[("results/*.json<br/>figures/*.png")]
    end
    RES --> DOCS["README and docs/"]
    PKG -.->|"Phase 6"| APP["FastAPI + React app"]
```

Everything downloaded is pinned to an exact revision (dataset commit `4a312b6`, MedSAM commit
`de8488b`), so the inputs are the same on every machine.

## 2. The research package, `src/cardiac/`

Which module calls which. Arrows point from the caller to what it uses.

```mermaid
flowchart TD
    subgraph CLI["Entry points (python -m ...)"]
        TR["train.py"]
        EV["evaluate.py"]
        EM["evaluate_medsam.py"]
        ME["measure.py"]
        EU["evaluate_uncertainty.py"]
    end
    subgraph Data["data/"]
        IO["io.py<br/>PatientMeta, Frame, load_frame"]
        SP["splits.py<br/>load_splits"]
        PP["preprocess.py<br/>prepare / restore"]
        DS["dataset.py<br/>SliceDataset"]
        AU["augment.py"]
    end
    subgraph Models["models/"]
        UN["unet.py<br/>scratch U-Net (MONAI)"]
        UP["unet_pretrained.py<br/>ResNet-34 U-Net (smp)"]
        MS["medsam.py<br/>box prompts"]
    end
    INF["infer.py<br/>predict_volume, postprocess"]
    MET["metrics.py<br/>dice, hd95"]
    CL["clinical.py<br/>volumes, EF, mass,<br/>thickness, Bland-Altman"]
    CLS["classify.py<br/>features, rules,<br/>logistic, XGBoost"]
    UNC["uncertainty.py<br/>TTA, doubt maps,<br/>risk–coverage"]
    CFG["config.py"]
    REP["repro.py<br/>seeds, git commit"]
    TR --> CFG & DS & AU & SP & INF & MET & REP
    DS --> IO & PP
    EV --> INF & MET & SP & IO & REP
    EM --> MS & INF & MET & SP & IO & EV
    ME --> CL & INF & MS & MET & SP & IO & REP
    EU --> UNC & INF & MET & ME & REP
    UNC --> PP & INF
    CLS --> IO
    INF --> PP & UN & UP
```

`clinical.py` is pure arithmetic on label maps: it never sees an image or a model, so the same code
measures expert masks, U-Net masks, MedSAM masks and, later, masks from an upload in the app.
`classify.py` likewise only sees a table of measurements: it imports no deep-learning code, so the
diagnosis script runs without torch (`decisions.md` D28).

`infer.py` is the one place that turns an image into a label map. Training (for validation), evaluation
and, later, the web backend all call it, so they cannot drift apart.

## 3. The three segmenters

All three take a 192 × 192 greyscale slice and return a label per pixel (0 background, 1 RV, 2 MYO,
3 LV). They differ in what they know before they see our data.

| | Scratch U-Net | Pretrained U-Net | MedSAM |
|---|---|---|---|
| Library | MONAI `UNet` | `segmentation_models_pytorch` `Unet` | `transformers` `SamModel` |
| Parameters | 6.5 M | 24.4 M (21.3 M encoder + 3.2 M decoder) | 93.7 M |
| Starts from | random weights | ResNet-34 trained on ImageNet photos (encoder only) | MedSAM, trained on ~1.5 M medical image–mask pairs |
| Trained on our 80 patients | yes, all weights | yes, all weights (lr 3e-4) | no, never |
| Input at inference | slice | slice | slice **plus a box around each structure** |
| Normalisation layers | instance norm | batch norm | layer norm (ViT) |

### 3a. Scratch U-Net

Drawn with measured shapes in `ml_basics.md` §7: five levels, 32 → 512 channels, 192 → 12 pixels at
the bottom.

### 3b. Pretrained U-Net (ResNet-34 encoder)

Shapes measured from the real model for one 192 × 192 slice (channels × height × width):

```mermaid
flowchart TD
    IN["input 1 × 192 × 192"] --> E1["ResNet stem: 7×7 conv, stride 2<br/>64 × 96 × 96"]
    E1 --> E2["max-pool + layer1 (3 residual blocks)<br/>64 × 48 × 48"]
    E2 --> E3["layer2 (4 blocks)<br/>128 × 24 × 24"]
    E3 --> E4["layer3 (6 blocks)<br/>256 × 12 × 12"]
    E4 --> E5["layer4 (3 blocks)<br/>512 × 6 × 6"]
    E5 --> D1["decoder 1: upsample + concat skip<br/>→ 256 × 12 × 12"]
    E4 -. skip .-> D1
    D1 --> D2["decoder 2 → 128 × 24 × 24"]
    E3 -. skip .-> D2
    D2 --> D3["decoder 3 → 64 × 48 × 48"]
    E2 -. skip .-> D3
    D3 --> D4["decoder 4 → 32 × 96 × 96"]
    E1 -. skip .-> D4
    D4 --> D5["decoder 5 → 16 × 192 × 192"]
    D5 --> H["3×3 conv head → 4 × 192 × 192"]
    classDef pre fill:#E0F2F1,stroke:#0E7C86
    class E1,E2,E3,E4,E5 pre
```

The shaded encoder blocks start with ImageNet weights; the decoder and head start random. Compared
with the scratch U-Net, it goes one level deeper (6 × 6 at the bottom) and has 3.8× as many weights,
almost all in the encoder.

The first convolution was built for 3 colour channels. `smp` turns it into a 1-channel convolution by
**summing** the red, green and blue weights, so a grey MRI slice excites it the way a grey photo with
the same brightness would.

### 3c. MedSAM (zero-shot, oracle boxes)

```mermaid
flowchart LR
    S["slice (192 × 192)<br/>→ RGB 0–255<br/>→ resized to 1024 × 1024"] --> IE["image encoder<br/>ViT-B, run once per slice<br/>→ 256 × 64 × 64 embedding"]
    L["expert label"] -->|"box around each structure,<br/>+5 px, optionally jittered"| PE["prompt encoder"]
    IE --> MD["mask decoder<br/>one mask per box"]
    PE --> MD
    MD --> M["3 binary masks<br/>(RV, MYO, LV)"]
    M -->|"paint RV, then MYO, then LV"| OUT["label map"]
```

Two things set it apart. First, it needs a prompt: here, boxes cut from the **expert's** answer, which
also tell it which slices contain each structure. Second, the expensive image encoder runs once per
slice and the cheap decoder once per box, so trying four jitter levels costs little more than one.
`decisions.md` D19 explains why these oracle boxes make this a best case for MedSAM, not a fair
real-world comparison.

## 4. The backend, `backend/app/` (Phase 6)

Three layers. Routes only translate HTTP to Python calls; services do the work; the research package
does the science. Nothing in `backend/` trains a model or computes a statistic that is not in
`src/cardiac/`.

```mermaid
flowchart TD
    subgraph Routes["routes/ (HTTP)"]
        RI["info.py<br/>GET /api/health<br/>GET /api/samples<br/>GET /api/metrics/summary"]
        RC["cases.py<br/>POST /api/analyze<br/>GET /api/cases/{id}<br/>GET .../slice/{phase}/{k}?layer=<br/>GET .../cine/{k}?frame=<br/>GET .../report.pdf"]
    end
    subgraph Services["services/"]
        MO["models.py<br/>load once at startup"]
        AN["analysis.py<br/>segment, measure,<br/>diagnose, explain"]
        UP["uploads.py<br/>read + check NIfTI"]
        IM["images.py<br/>PNG bytes"]
        ST["store.py<br/>samples (disk),<br/>uploads (memory, 1 h)"]
        ME["metrics.py<br/>reshape results/"]
    end
    RP["report.py<br/>ReportLab PDF"]
    subgraph Pkg["src/cardiac/ (research package)"]
        INF["infer.predict_volume"]
        UNC["uncertainty.predict_with_doubt"]
        CL["clinical.measure_patient"]
        CLS["classify.FEATURES"]
    end
    RC --> UP & AN & ST & IM & RP
    RI --> ST & ME
    AN --> MO & INF & UNC & CL & CLS
    MO --> F1[("runs/unet_*/…-seed0/best.pt<br/>runs/classifier/xgboost.json")]
    MO --> F2[("results/normal_ranges.json<br/>results/val/uncertainty_…seed0.csv")]
    ST --> F3[("data/cache/samples/*.npz + *.json")]
    ME --> F4[("results/*.json, results/test/*.csv")]
```

| File | Job |
|---|---|
| `__init__.py` | sets `OMP_NUM_THREADS=1` before torch or xgboost load (`decisions.md` D28) |
| `config.py` | paths and limits, each overridable by an environment variable |
| `main.py` | `create_app()`: CORS for the Vite dev server, routers, model loading at startup |
| `state.py` | the per-app state handed to every route: settings, models, case store |
| `schemas.py` | response shapes (pydantic); `frontend/src/data/types.ts` mirrors them |
| `services/models.py` | both U-Nets (seed 0), the XGBoost model and its SHAP explainer, normal ranges, doubt threshold |
| `services/analysis.py` | re-centring, both U-Nets, TTA doubt, `measure_patient`, diagnosis and plain-word reasons |
| `services/uploads.py` | NIfTI from bytes in a temporary folder, with checks that explain what's wrong |
| `services/images.py` | windowing and PNG encoding; one orientation function for every layer |
| `services/store.py` | sample cases from `data/cache/samples/`, uploads in memory for an hour |
| `services/metrics.py` | the comparison page's data, selected from `results/` |
| `report.py` | the one-page PDF |

**Data model on the wire.** A case is one JSON summary plus PNGs fetched slice by slice:

| What | Shape / type | Encoding |
|---|---|---|
| image slice | (X, Y) uint8, windowed 1–99th percentile over the whole volume | 8-bit greyscale PNG |
| mask slice (`unet`, `unet_pretrained`, `medsam`, `expert`) | (X, Y) uint8, values 0–3 | 8-bit greyscale PNG; the browser maps 0–3 to colours |
| doubt slice | (X, Y) uint8, normalised entropy × 255 | 8-bit greyscale PNG |
| cine slice (samples only) | (X, Y) uint8 per frame | 8-bit greyscale PNG |
| summary | measurements with normal ranges and status, probabilities, top-5 SHAP reasons, doubt score and flag, model versions | JSON (`CaseSummary`) |

Every 2D plane goes through `images.to_display()` (transpose, then flip rows), so image, masks and
doubt map always line up and match the orientation of the figures in `figures/`.

**What the app shows is what was evaluated.** Masks, measurements and diagnosis come from the
single-view pretrained U-Net, seed 0: for the 5 sample patients, every measurement equals
`results/test/clinical_unet_resnet34_seed0.csv` to the second decimal, and every diagnosis equals
`results/test/diagnosis_predictions.csv`. TTA is used for the doubt map and score only, as in
`cardiac.evaluate_uncertainty` (`decisions.md` D31).

## 5. The frontend, `frontend/` (Phase 7)

React + TypeScript, built by Vite, styled with Tailwind on the design tokens in
`src/styles/globals.css`. Four pages, one data-source interface, one small store.

```mermaid
flowchart TD
    App["app/App.tsx<br/>hash router, theme, layout"] --> Home["HomePage<br/>samples, upload"]
    App --> WS["WorkspacePage"]
    App --> MC["ComparisonPage"]
    App --> HOW["HowItWorksPage"]
    Home --> UF["UploadForm"]
    WS --> V["ScanViewer ×1 or ×2 (compare)<br/>canvas: image, mask, doubt"]
    WS --> VC["ViewerControls<br/>slice, ED/ES, cine, model,<br/>opacity, doubt, compare, keys"]
    WS --> LG["Legend"]
    WS --> CB["ConfidenceBanner"]
    WS --> MP["MeasurementsCard → RangeBar, StatusChip"]
    WS --> DX["DiagnosisCard"]
    WS --> WHY["WhyCard (SHAP)"]
    MC --> CH["charts: DiceBars, BlandAltman,<br/>ConfusionMatrix, RiskCoverage, JitterCurve"]
    V & VC --> ST[("stores/viewerStore.ts<br/>Zustand: phase, slice, layer,<br/>opacity, doubt, compare, cine, zoom")]
    V --> IC["lib/imageCache.ts<br/>decode PNG → coloured canvas"]
    IC --> CM["lib/colormap.ts<br/>labels 0–3 → RGBA, doubt → purple ramp"]
    Home & WS & MC --> DS{{"data/dataSource.ts"}}
    DS -->|"VITE_DATA_MODE=api"| API["apiSource.ts → FastAPI /api"]
    DS -->|"VITE_DATA_MODE=static"| STA["staticSource.ts → public/demo/*"]
```

```
frontend/src/
├── app/App.tsx           router (hash), theme toggle, header, footer, demo banner
├── pages/                HomePage, WorkspacePage, ComparisonPage, HowItWorksPage
├── components/
│   ├── viewer/           ScanViewer (canvas), ViewerControls (+ keyboard, cine), Legend
│   ├── panels/           MeasurementsCard, RangeBar, StatusChip, DiagnosisCard, WhyCard, ConfidenceBanner
│   ├── charts/           hand-built SVG charts, scale helpers, model names
│   ├── ui/controls.tsx   Button, Card, Segmented, Toggle, Skeleton, ErrorState, Icon
│   └── UploadForm.tsx
├── data/                 types.ts (mirrors backend/app/schemas.py), dataSource, apiSource, staticSource, http
├── lib/                  colormap, normalRanges, format, imageCache, useLoad, useCountUp (+ helpers.test.ts)
├── stores/viewerStore.ts
└── styles/globals.css    tokens, fonts, motion, reduced-motion rules
```

**How a slice is drawn.** For the current case, phase and slice, `ScanViewer` asks the data source for
three URLs: the greyscale image, the chosen model's label PNG and (if on) the doubt PNG.
`imageCache.getLayer()` loads each once, draws it onto an off-screen canvas, and for labels and doubt
rewrites the pixels through `colormap.ts`. The viewer then draws the image (smoothed), the mask
(nearest-neighbour, at the chosen opacity, cross-fading over 200 ms when the model changes) and the
doubt map on one canvas, with the current zoom and pan. Every slice of both phases is preloaded in the
background, so scrolling never waits.

## 6. Deployment view (Phase 8)

```mermaid
flowchart LR
    subgraph Local["Local full app (make api + make ui)"]
        UI1["React app<br/>apiSource"] -->|"/api (Vite proxy)"| API["FastAPI<br/>backend/app"]
        API --> PKG["src/cardiac<br/>U-Nets, XGBoost, TTA"]
        API --> W[("runs/ weights<br/>data/cache/samples")]
    end
    subgraph Export["make demo (local, once)"]
        EX["scripts/export_demo.py"] --> FILES[("frontend/public/demo/<br/>5 cases, PNG slices,<br/>cine, PDFs, metrics.json")]
    end
    subgraph Pages["GitHub Pages (public)"]
        GA["GitHub Actions<br/>deploy-demo.yml<br/>npm ci → build:static"] --> SITE["static site<br/>staticSource"]
    end
    W --> EX
    FILES -->|"git push"| GA
```

| | Local full app | Public static demo |
|---|---|---|
| Sample patients, all 4 outlines, cine, doubt map | yes | yes |
| PDF report | generated on request | pre-built for the 5 samples |
| Model comparison page | read from `results/` by the API | the same data, exported to `metrics.json` |
| Upload your own scan | yes | no (says how to run locally) |
| Needs | Python env, weights, Node | a browser |


## 7. Data shapes at every stage

One frame of one patient, from file to browser (the mirror's sizes; X and Y are in-plane, S is the
number of short-axis slices, typically 6–10):

```mermaid
flowchart LR
    A["NIfTI on disk<br/>(X, Y, S) float<br/>+ spacing mm"] -->|"load_frame()"| B["image (X, Y, S) float32<br/>label (X, Y, S) uint8 0–3"]
    B -->|"prepare()"| C["(S, 192, 192) float32<br/>z-scored"]
    C -->|"batched, + channel"| D["(B, 1, 192, 192)<br/>network input"]
    D -->|"U-Net, softmax"| E["(S, 4, 192, 192) float32<br/>probabilities"]
    E -->|"argmax, restore(),<br/>postprocess()"| F["(X, Y, S) uint8<br/>label map"]
    E -->|"10 TTA views,<br/>entropy × 255"| G["(X, Y, S) uint8<br/>doubt map"]
    F -->|"measure_patient()"| H["15 numbers<br/>mL, %, g, mm, /m²"]
    H -->|"feature_matrix()"| I["(1, 18) float<br/>features"]
    I -->|"XGBoost"| J["(1, 5) probabilities<br/>+ (18, 5) SHAP"]
    F & G -->|"to_display(), PNG"| K["(Y, X) uint8 PNG<br/>per slice"]
    K -->|"colormap.ts"| L["RGBA canvas"]
```

| Stage | Shape | dtype | Where |
|---|---|---|---|
| file | (X, Y, S) | float (as stored) | `data/raw/acdc/…/*.nii.gz` |
| frame | (X, Y, S) image, (X, Y, S) label | float32, uint8 | `cardiac.data.io.Frame` |
| network input | (S, 192, 192) → batches of (B, 1, 192, 192) | float32 | `cardiac.data.preprocess.prepare` |
| network output | (S, 4, 192, 192) | float32 | `cardiac.infer.predict_probabilities` |
| label map | (X, Y, S) | uint8, 0–3 | `cardiac.infer.predict_volume` |
| doubt map | (X, Y, S) | uint8, 0–255 | `cardiac.uncertainty.predict_with_doubt` |
| measurements | 15 named values | float | `cardiac.clinical.measure_patient` |
| features | (patients, 18) | float | `cardiac.classify.feature_matrix` |
| PNG slice | (Y, X) after `to_display` | uint8 | `backend/app/services/images.py` |

## 8. Folder map

```
.github/workflows/deploy-demo.yml   builds and publishes the static demo
backend/app/                        FastAPI app (§4)
configs/                            unet_scratch.yaml, unet_resnet34.yaml, classifier.yaml
data/splits.json                    patient ids per split (the only committed file in data/)
docs/                               ml_basics, flow, architecture, decisions; images/ for the README
figures/                            every figure, written by scripts and the notebook
frontend/                           React app (§5); public/demo/ holds the exported demo files
notebooks/01_data_exploration.ipynb data exploration
results/                            every number, written by evaluation scripts
scripts/
  download_data.py                  fetch and verify the ACDC mirror
  make_splits.py                    write data/splits.json
  plot_run.py, plot_augmentations.py, plot_wall_thickness.py, plot_doubt_maps.py   figures
  summarize_segmentation.py         results/segmentation.json + figures
  summarize_clinical.py             results/clinical.json + Bland-Altman plots
  summarize_uncertainty.py          results/uncertainty.json + risk–coverage plot
  run_diagnosis.py                  diagnosis models, results/diagnosis.json, SHAP
  normal_ranges.py                  results/normal_ranges.json
  cache_sample_cases.py             the app's 5 samples → data/cache/samples/
  export_demo.py                    the static demo → frontend/public/demo/
  make_readme_gif.py                docs/images/slices.gif
src/cardiac/
  data/io.py, splits.py, preprocess.py, augment.py, dataset.py
  models/unet.py, unet_pretrained.py, medsam.py
  config.py, repro.py               configs; seeds and code version
  train.py                          training loop with early stopping
  infer.py                          image → label map, post-processing
  metrics.py                        Dice, HD95
  evaluate.py, evaluate_medsam.py   segmentation scores per patient
  clinical.py, measure.py           clinical numbers
  classify.py                       diagnosis features and models
  uncertainty.py, evaluate_uncertainty.py   TTA doubt
  plots.py                          shared figure helpers
tests/                              pytest suite
```
