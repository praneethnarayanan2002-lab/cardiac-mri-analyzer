"""Clinical numbers from label maps: volumes, ejection fraction, mass, wall thickness, BSA indexing.

Every function takes label maps shaped (X, Y, Z) with values 0 background, 1 RV, 2 MYO, 3 LV, and
the voxel spacing in millimetres in the same (X, Y, Z) order. Volumes are in mL, mass in g,
thickness in mm, body surface area in m².
"""

import math

import numpy as np
from scipy import ndimage, stats

RV, MYO, LV = 1, 2, 3
MYOCARDIAL_DENSITY_G_PER_ML = 1.05

# Column name -> unit, in the order they appear in tables and plots.
MEASUREMENTS = {
    "lv_edv_ml": "mL",
    "lv_esv_ml": "mL",
    "lv_ef_pct": "%",
    "rv_edv_ml": "mL",
    "rv_esv_ml": "mL",
    "rv_ef_pct": "%",
    "myo_mass_g": "g",
    "wall_thickness_mean_mm": "mm",
    "wall_thickness_max_mm": "mm",
}
INDEXED = {
    "lv_edv_ml": "lv_edvi_ml_m2",
    "lv_esv_ml": "lv_esvi_ml_m2",
    "rv_edv_ml": "rv_edvi_ml_m2",
    "rv_esv_ml": "rv_esvi_ml_m2",
    "myo_mass_g": "myo_mass_index_g_m2",
}


def voxel_volume_ml(spacing_mm: tuple[float, float, float]) -> float:
    """1 mL = 1 cm³ = 1000 mm³."""
    return spacing_mm[0] * spacing_mm[1] * spacing_mm[2] / 1000.0


def structure_volume_ml(
    label: np.ndarray, structure: int, spacing_mm: tuple[float, float, float]
) -> float:
    return int(np.count_nonzero(label == structure)) * voxel_volume_ml(spacing_mm)


def ejection_fraction(edv_ml: float, esv_ml: float) -> float:
    """Percentage of the filled volume pumped out per beat; NaN when there is no ED volume."""
    if edv_ml <= 0:
        return math.nan
    return (edv_ml - esv_ml) / edv_ml * 100.0


def myocardial_mass_g(ed_label: np.ndarray, spacing_mm: tuple[float, float, float]) -> float:
    """Myocardium is measured at end-diastole, by convention."""
    return structure_volume_ml(ed_label, MYO, spacing_mm) * MYOCARDIAL_DENSITY_G_PER_ML


def wall_thickness_samples(label: np.ndarray, spacing_mm: tuple[float, float, float]) -> np.ndarray:
    """Distances (mm) from each outer myocardial border pixel to the nearest LV cavity pixel.

    Works slice by slice, because slices are 10 mm apart and the wall is a few mm thick: measuring
    across slices would mix in the slice gap. Slices without both cavity and myocardium (the apex
    cap, slices above the valve) are skipped. The outer border is every myocardium pixel that
    touches something other than myocardium or cavity, so the septum is measured against the RV.
    """
    samples = []
    cross = ndimage.generate_binary_structure(2, 1)
    for z in range(label.shape[2]):
        sl = label[:, :, z]
        cavity, myo = sl == LV, sl == MYO
        if not cavity.any() or not myo.any():
            continue
        outside = ~(cavity | myo)
        outer_border = myo & ndimage.binary_dilation(outside, structure=cross)
        # Distance from every pixel to the nearest cavity pixel, in mm, using real in-plane spacing.
        to_cavity = ndimage.distance_transform_edt(~cavity, sampling=spacing_mm[:2])
        samples.append(np.asarray(to_cavity)[outer_border])
    return np.concatenate(samples) if samples else np.empty(0)


def wall_thickness_mm(
    label: np.ndarray, spacing_mm: tuple[float, float, float]
) -> tuple[float, float]:
    """(mean, max) wall thickness over all border samples in the volume; NaN if there are none."""
    samples = wall_thickness_samples(label, spacing_mm)
    if samples.size == 0:
        return math.nan, math.nan
    return float(samples.mean()), float(samples.max())


def body_surface_area_m2(height_cm: float, weight_kg: float) -> float:
    """Mosteller formula."""
    return math.sqrt(height_cm * weight_kg / 3600.0)


def measure_patient(
    ed_label: np.ndarray,
    es_label: np.ndarray,
    spacing_mm: tuple[float, float, float],
    height_cm: float,
    weight_kg: float,
) -> dict[str, float]:
    """Every measurement in MEASUREMENTS, the indexed ones in INDEXED, and the BSA."""
    out: dict[str, float] = {}
    for name, structure in (("lv", LV), ("rv", RV)):
        edv = structure_volume_ml(ed_label, structure, spacing_mm)
        esv = structure_volume_ml(es_label, structure, spacing_mm)
        out[f"{name}_edv_ml"] = edv
        out[f"{name}_esv_ml"] = esv
        out[f"{name}_ef_pct"] = ejection_fraction(edv, esv)
    out["myo_mass_g"] = myocardial_mass_g(ed_label, spacing_mm)
    mean_t, max_t = wall_thickness_mm(ed_label, spacing_mm)
    out["wall_thickness_mean_mm"] = mean_t
    out["wall_thickness_max_mm"] = max_t
    bsa = body_surface_area_m2(height_cm, weight_kg)
    out["bsa_m2"] = bsa
    for raw, indexed in INDEXED.items():
        out[indexed] = out[raw] / bsa
    return out


def bland_altman(predicted: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Agreement of predicted with reference values, one pair per patient; NaN pairs are dropped.

    bias = mean(predicted - reference); limits of agreement = bias ± 1.96 SD of the differences,
    the range expected to hold 95% of future differences if they are roughly normal."""
    predicted, reference = np.asarray(predicted, float), np.asarray(reference, float)
    keep = ~(np.isnan(predicted) | np.isnan(reference))
    p, r = predicted[keep], reference[keep]
    diff = p - r
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1))
    return {
        "n": int(keep.sum()),
        "n_dropped": int((~keep).sum()),
        "bias": bias,
        "sd_diff": sd,
        "loa_low": bias - 1.96 * sd,
        "loa_high": bias + 1.96 * sd,
        "mae": float(np.abs(diff).mean()),
        "pearson_r": float(stats.pearsonr(p, r).statistic),
        "reference_mean": float(r.mean()),
    }
