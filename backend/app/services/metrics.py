"""The test-set results the "Model comparison" page shows, assembled from results/ only.

Nothing here computes a new statistic: it selects and reshapes what the research scripts wrote, plus
per-patient points for the charts (Bland-Altman dots, risk-coverage curve)."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from cardiac.clinical import MEASUREMENTS
from cardiac.uncertainty import risk_coverage

MODELS = {
    "unet_scratch": "U-Net (scratch)",
    "unet_resnet34": "U-Net (ResNet-34)",
    "medsam_jitter0": "MedSAM (oracle boxes)",
}
STRUCTURES = ("RV", "MYO", "LV")


def _load(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text()))


def segmentation(results: Path) -> dict[str, Any]:
    seg = _load(results / "segmentation.json")
    dice = {m: {s: seg[m]["overall"][f"dice_{s}"] for s in STRUCTURES} for m in MODELS if m in seg}
    hd95 = {m: {s: seg[m]["overall"][f"hd95_{s}"] for s in STRUCTURES} for m in MODELS if m in seg}
    jitters = sorted(
        int(k.removeprefix("medsam_jitter")) for k in seg if k.startswith("medsam_jitter")
    )
    jitter = [
        {
            "px": j,
            **{s: seg[f"medsam_jitter{j}"]["overall"][f"dice_{s}"]["mean"] for s in STRUCTURES},
        }
        for j in jitters
    ]
    return {"dice": dice, "hd95": hd95, "jitter": jitter, "n_patients": 50}


def clinical(results: Path) -> dict[str, Any]:
    stats = _load(results / "clinical.json")
    expert = pd.read_csv(results / "test" / "clinical_expert.csv").set_index("pid")
    points: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for model in MODELS:
        stem = f"{model}_seed0" if model.startswith("unet") else model
        path = results / "test" / f"clinical_{stem}.csv"
        if not path.exists():
            continue
        pred = pd.read_csv(path).set_index("pid").reindex(expert.index)
        points[model] = {
            key: [
                {
                    "pid": pid,
                    "group": expert.group[pid],
                    "model": float(pred[key][pid]),
                    "expert": float(expert[key][pid]),
                }
                for pid in expert.index
                if pd.notna(pred[key][pid]) and pd.notna(expert[key][pid])
            ]
            for key in MEASUREMENTS
        }
    summary = {
        model: {
            key: {
                k: stats[model][key][k] for k in ("bias", "loa_low", "loa_high", "mae", "pearson_r")
            }
            for key in MEASUREMENTS
        }
        for model in MODELS
        if model in stats
    }
    units = dict(MEASUREMENTS)
    return {"stats": summary, "points": points, "units": units, "points_seed": 0}


def diagnosis(results: Path) -> dict[str, Any]:
    diag = _load(results / "diagnosis.json")
    out: dict[str, Any] = {"train": diag["train"], "test": {}}
    for source, models in diag["test"].items():
        out["test"][source] = {
            name: {
                "accuracy": block["accuracy"],
                "macro_f1": block["macro_f1"],
                "confusion": block["per_run"][0]["confusion"],
            }
            for name, block in models.items()
        }
    return out


def uncertainty(results: Path) -> dict[str, Any]:
    unc = _load(results / "uncertainty.json")
    curve = []
    path = results / "test" / "uncertainty_unet_resnet34_seed0.csv"
    if path.exists():
        table = pd.read_csv(path)
        table["error"] = 1.0 - table[[f"dice_{s}" for s in STRUCTURES]].mean(axis=1)
        patients = table.groupby("pid")[["doubt_score", "error"]].mean()
        coverage, risk = risk_coverage(patients.doubt_score.to_numpy(), patients.error.to_numpy())
        _, oracle = risk_coverage(patients.error.to_numpy(), patients.error.to_numpy())
        curve = [
            {"coverage": float(c), "risk": float(r), "oracle": float(o)}
            for c, r, o in zip(coverage, risk, oracle, strict=True)
        ]
        random_risk = float(patients.error.mean())
    else:
        random_risk = float("nan")
    block = unc.get("unet_resnet34", {})
    return {
        "spearman_rho": block.get("spearman_rho"),
        "aurc": block.get("aurc"),
        "curve_seed0": curve,
        "random_risk_seed0": random_risk,
    }


def metrics_summary(results: Path) -> dict[str, Any]:
    return {
        "segmentation": segmentation(results),
        "clinical": clinical(results),
        "diagnosis": diagnosis(results),
        "uncertainty": uncertainty(results),
    }
