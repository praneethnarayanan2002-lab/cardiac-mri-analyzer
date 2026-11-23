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
# Re-centre the crop on the LV only when the first pass finds it this far (mm) from the centre.
# The mirror's scans are already centred, so their masks stay exactly the evaluated ones. Passes
# stop once the centre moves less than SETTLED_MM.
RECENTRE_MM = 16.0
SETTLED_MM = 2.0
MAX_RECENTRE_PASSES = 3
LV = 3

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


def lv_centre(label: np.ndarray) -> tuple[float, float] | None:
    """In-plane centre of mass (voxel units) of the predicted LV cavity, or None if there is none.

    The LV, not the whole heart: the mirror's training images were cropped around the expert LV
    centre, so this reproduces the framing the model learned on."""
    if not (label == LV).any():
        return None
    cx, cy, _ = ndimage.center_of_mass(label == LV)
    return float(cx), float(cy)


def find_crop_centre(
    models: ModelBundle, image: np.ndarray, spacing_mm: tuple[float, float, float]
) -> tuple[float, float] | None:
    """Where to centre the crop: None (the image centre) if the LV is already within RECENTRE_MM
    of it, otherwise the LV centre, refined over a few passes because the first pass may only see
    part of a heart near the edge of the window."""
    pre = models.unet_pretrained
    data = pre.config.data
    mid = ((image.shape[0] - 1) / 2, (image.shape[1] - 1) / 2)
    centre: tuple[float, float] | None = None
    for _ in range(MAX_RECENTRE_PASSES):
        label = predict_volume(
            pre.model,
            image,
            spacing_mm,
            models.device,
            data.target_spacing_mm,
            data.size,
            centre_xy=centre,
        )
        found = lv_centre(label)
        if found is None:
            return centre
        here = centre or mid
        moved_mm = math.hypot(
            (found[0] - here[0]) * spacing_mm[0], (found[1] - here[1]) * spacing_mm[1]
        )
        if centre is None and moved_mm <= RECENTRE_MM:
            return None  # already centred, like every scan in the mirror
        if centre is not None and moved_mm <= SETTLED_MM:
            return centre
        centre = found
    return centre


def segment(
    models: ModelBundle,
    images: dict[str, np.ndarray],
    spacing_mm: tuple[float, float, float],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, float]]:
    """Both U-Nets on both phases, plus TTA on the pretrained one. Returns layers and doubt scores.

    The masks shown and measured are the single-view predictions, the ones every number in
    results/ was computed from. TTA only supplies the doubt map and score, computed exactly as in
    cardiac.evaluate_uncertainty (docs/decisions.md D31)."""
    pre, scratch = models.unet_pretrained, models.unet
    centre = find_crop_centre(models, images["ed"], spacing_mm)
    layers: dict[str, dict[str, np.ndarray]] = {}
    scores: dict[str, float] = {}
    for phase in PHASES:
        masks = {}
        for name, unet in (("unet_pretrained", pre), ("unet", scratch)):
            data = unet.config.data
            masks[name] = predict_volume(
                unet.model,
                images[phase],
                spacing_mm,
                models.device,
                data.target_spacing_mm,
                data.size,
                centre_xy=centre,
            )
        tta_label, doubt = predict_with_doubt(
            pre.model,
            images[phase],
            spacing_mm,
            models.device,
            pre.config.data.target_spacing_mm,
            pre.config.data.size,
            centre_xy=centre,
        )
        layers[phase] = {**masks, "doubt": doubt}
        scores[phase] = doubt_score(doubt, tta_label)
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
        # "High LV ejection fraction", but "Low myocardial mass": keep acronyms in capitals.
        shown = label if not level or label[:2].isupper() else label[0].lower() + label[1:]
        text = f"{level}{shown} {verb} {GROUP_NAMES[group]}"
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
            "measurements_from": f"unet_resnet34 ({models.unet_pretrained.run})",
            "doubt_from": "10-view TTA of the same model",
            "unet": models.unet.run,
            "unet_pretrained": models.unet_pretrained.run,
            "classifier": "xgboost, trained on expert features of 100 training patients",
            "code_version": models.code_version,
        },
    }
    return Analysis(layers=layers, summary=summary)
