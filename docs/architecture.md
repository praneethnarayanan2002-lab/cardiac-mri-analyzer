# Architecture

What the pieces are and how they connect. `flow.md` follows execution step by step; this file is the
map. Sections are added in the phase that builds each part: the research package and the three
segmenters (Phases 3–4), the backend in Phase 6, the frontend in Phase 7, deployment in Phase 8.

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
    CFG["config.py"]
    REP["repro.py<br/>seeds, git commit"]
    TR --> CFG & DS & AU & SP & INF & MET & REP
    DS --> IO & PP
    EV --> INF & MET & SP & IO & REP
    EM --> MS & INF & MET & SP & IO & EV
    ME --> CL & INF & MS & MET & SP & IO & REP
    INF --> PP & UN & UP
```

`clinical.py` is pure arithmetic on label maps: it never sees an image or a model, so the same code
measures expert masks, U-Net masks, MedSAM masks and, later, masks from an upload in the app.

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
