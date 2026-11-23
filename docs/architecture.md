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
