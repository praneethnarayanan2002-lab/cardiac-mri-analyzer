"""From two image volumes (ED, ES) to everything the app shows: masks, doubt maps, measurements,
diagnosis, SHAP reasons and the doubt flag. Used for uploads and to build the sample cache."""

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage

from backend.app.services.models import ModelBundle
from cardiac.classify import FEATURES, add_derived
from cardiac.clinical import INDEXED, MEASUREMENTS, measure_patient
from cardiac.data.io import ACDC_GROUPS
from cardiac.infer import predict_volume
from cardiac.uncertainty import doubt_score, predict_with_doubt

PHASES = ("ed", "es")
# Re-centre the crop on the heart only when the first pass finds it this far (mm) from the image
# centre. The mirror's scans are already centred, so their masks stay exactly the evaluated ones.
RECENTRE_MM = 16.0

LABELS = {
    "lv_edv_ml": "LV end-diastolic volume",
    "lv_esv_ml": "LV end-systolic volume",
    "lv_ef_pct": "LV ejection fraction",
    "rv_edv_ml": "RV end-diastolic volume",
    "rv_esv_ml": "RV end-systolic volume",
    "rv_ef_pct": "RV ejection fraction",
    "myo_mass_g": "Myocardial mass",
    "wall_thickness_mean_mm": "Mean wall thickness",
    "wall_thickness_max_mm": "Max wall thickness",
    "lv_edvi_ml_m2": "LV EDV per body area",
    "lv_esvi_ml_m2": "LV ESV per body area",
    "rv_edvi_ml_m2": "RV EDV per body area",
    "rv_esvi_ml_m2": "RV ESV per body area",
    "myo_mass_index_g_m2": "Myocardial mass per body area",
    "rv_lv_edv_ratio": "RV/LV volume ratio",
    "mass_volume_ratio_g_ml": "Mass/volume ratio",
    "height_cm": "Height",
    "weight_kg": "Weight",
}
UNITS = {
    **MEASUREMENTS,
    "lv_edvi_ml_m2": "mL/m²",
    "lv_esvi_ml_m2": "mL/m²",
    "rv_edvi_ml_m2": "mL/m²",
    "rv_esvi_ml_m2": "mL/m²",
    "myo_mass_index_g_m2": "g/m²",
    "rv_lv_edv_ratio": "",
    "mass_volume_ratio_g_ml": "g/mL",
    "height_cm": "cm",
    "weight_kg": "kg",
}
GROUP_NAMES = {
    "NOR": "normal",
    "MINF": "previous heart attack (MINF)",
    "DCM": "dilated cardiomyopathy (DCM)",
    "HCM": "hypertrophic cardiomyopathy (HCM)",
    "RV": "abnormal right ventricle (RV)",
}
SHOWN_MEASUREMENTS = (*MEASUREMENTS, *INDEXED.values())


@dataclass
class Analysis:
    """Arrays are (X, Y, Z) on the scan's own grid; doubt maps are uint8 0-255."""

    layers: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)  # phase -> layer -> map
    summary: dict[str, object] = field(default_factory=dict)


def heart_centre(
    label: np.ndarray, spacing_mm: tuple[float, float, float]
) -> tuple[float, float] | None:
    """In-plane centre of mass (voxel units) of all heart voxels, or None if the first pass found
    nothing or the heart is already near the image centre."""
    if not (label > 0).any():
        return None
    cx, cy, _ = ndimage.center_of_mass(label > 0)
    mid_x, mid_y = (label.shape[0] - 1) / 2, (label.shape[1] - 1) / 2
    offset_mm = math.hypot((cx - mid_x) * spacing_mm[0], (cy - mid_y) * spacing_mm[1])
    return (float(cx), float(cy)) if offset_mm > RECENTRE_MM else None


def segment(
    models: ModelBundle,
    images: dict[str, np.ndarray],
    spacing_mm: tuple[float, float, float],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, float]]:
    """Both U-Nets on both phases, the pretrained one with TTA. Returns layers and doubt scores."""
    pre, scratch = models.unet_pretrained, models.unet
    data = pre.config.data
    first = predict_volume(
        pre.model, images["ed"], spacing_mm, models.device, data.target_spacing_mm, data.size
    )
    centre = heart_centre(first, spacing_mm)
    layers: dict[str, dict[str, np.ndarray]] = {}
    scores: dict[str, float] = {}
    for phase in PHASES:
        label, doubt = predict_with_doubt(
            pre.model,
            images[phase],
            spacing_mm,
            models.device,
            data.target_spacing_mm,
            data.size,
            centre_xy=centre,
        )
        second = predict_volume(
            scratch.model,
            images[phase],
            spacing_mm,
            models.device,
            scratch.config.data.target_spacing_mm,
            scratch.config.data.size,
            centre_xy=centre,
        )
        layers[phase] = {"unet_pretrained": label, "unet": second, "doubt": doubt}
        scores[phase] = doubt_score(doubt, label)
    return layers, scores


def range_status(value: float, rng: dict[str, float] | None) -> str:
    """normal inside mean ± 2 SD of the normal training patients, borderline up to 3 SD, else
    abnormal. Unknown when there is no value or no range."""
    if rng is None or value is None or math.isnan(value):
        return "unknown"
    z = abs(value - rng["mean"]) / rng["sd"] if rng["sd"] > 0 else 0.0
    return "normal" if z <= 2 else "borderline" if z <= 3 else "abnormal"


def explain(
    models: ModelBundle, row: dict[str, float], top: int = 5
) -> tuple[dict[str, float], str, list[dict[str, object]]]:
    """Probabilities, predicted group and the top SHAP reasons for it, in plain words."""
    table = add_derived(pd.DataFrame([row]))
    x = table[list(FEATURES)].to_numpy(dtype=float)
    probs = models.classifier.predict_proba(x)[0]
    k = int(np.argmax(probs))
    group = ACDC_GROUPS[k]
    values = np.asarray(models.explainer.shap_values(x))
    if values.ndim == 3 and values.shape[0] == len(ACDC_GROUPS):  # older shap: classes first
        values = np.moveaxis(values, 0, 2)
    contributions = values[0][:, k]  # (features,) log-odds towards the predicted group
    reasons = []
    for i in np.argsort(-np.abs(contributions))[:top]:
        name = FEATURES[i]
        value = float(table[name].iloc[0])
        rng = models.normal_ranges.get(name)
        level = ""
        if rng is not None and not math.isnan(value):
            level = "High " if value > rng["mean"] else "Low "
        verb = "pushed toward" if contributions[i] > 0 else "pushed away from"
        label = LABELS[name]
        text = (
            f"{level}{label[0].lower() + label[1:] if level else label} {verb} {GROUP_NAMES[group]}"
        )
        reasons.append(
            {
                "feature": name,
                "label": label,
                "value": value,
                "unit": UNITS[name],
                "shap": float(contributions[i]),
                "text": text,
            }
        )
    return {g: float(p) for g, p in zip(ACDC_GROUPS, probs, strict=True)}, group, reasons


def analyse(
    models: ModelBundle,
    images: dict[str, np.ndarray],
    spacing_mm: tuple[float, float, float],
    height_cm: float,
    weight_kg: float,
) -> Analysis:
    layers, scores = segment(models, images, spacing_mm)
    labels = {p: layers[p]["unet_pretrained"] for p in PHASES}
    row = measure_patient(labels["ed"], labels["es"], spacing_mm, height_cm, weight_kg)
    row["height_cm"], row["weight_kg"] = height_cm, weight_kg
    probabilities, predicted, reasons = explain(models, row)
    measurements = []
    for key in SHOWN_MEASUREMENTS:
        rng = models.normal_ranges.get(key)
        measurements.append(
            {
                "key": key,
                "label": LABELS[key],
                "unit": UNITS[key],
                "value": None if math.isnan(row[key]) else round(float(row[key]), 2),
                "range": rng,
                "status": range_status(row[key], rng),
            }
        )
    score = float(np.nanmean([scores[p] for p in PHASES]))
    summary: dict[str, object] = {
        "spacing_mm": [float(s) for s in spacing_mm],
        "shape": [int(n) for n in images["ed"].shape],
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "bsa_m2": round(row["bsa_m2"], 3),
        "measurements": measurements,
        "diagnosis": {"predicted": predicted, "probabilities": probabilities},
        "why": reasons,
        "uncertainty": {
            "score": score,
            "per_phase": scores,
            "threshold": models.doubt_threshold,
            "flagged": bool(score > models.doubt_threshold),
        },
        "models": {
            "measurements_from": f"unet_resnet34 ({models.unet_pretrained.run}) with TTA",
            "unet": models.unet.run,
            "unet_pretrained": models.unet_pretrained.run,
            "classifier": "xgboost, trained on expert features of 100 training patients",
            "code_version": models.code_version,
        },
    }
    return Analysis(layers=layers, summary=summary)
