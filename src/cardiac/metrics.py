"""Segmentation metrics on 3D label volumes, in physical units."""

import numpy as np
from scipy import ndimage

STRUCTURES = {1: "RV", 2: "MYO", 3: "LV"}


def dice(pred: np.ndarray, truth: np.ndarray) -> float:
    """Dice overlap of two binary masks: 2|A∩B| / (|A| + |B|). 1 = identical, 0 = no overlap.

    Two empty masks count as a perfect match (1.0): the model correctly found nothing."""
    pred, truth = pred.astype(bool), truth.astype(bool)
    total = pred.sum() + truth.sum()
    if total == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, truth).sum() / total)


def _surface(mask: np.ndarray) -> np.ndarray:
    # A voxel is on the surface if it is in the mask but at least one neighbour is not.
    return np.asarray(mask & ~ndimage.binary_erosion(mask, border_value=0))


def hd95(pred: np.ndarray, truth: np.ndarray, spacing_mm: tuple[float, ...]) -> float:
    """95th-percentile symmetric surface distance in mm.

    For every surface voxel of each mask, find the distance to the nearest surface voxel of the
    other mask (using the real voxel size), pool both directions, take the 95th percentile.
    Returns NaN if either mask is empty, because a distance to nothing is undefined."""
    pred, truth = pred.astype(bool), truth.astype(bool)
    if not pred.any() or not truth.any():
        return float("nan")
    pred_surface, truth_surface = _surface(pred), _surface(truth)
    # distance_transform_edt measures distance to the nearest zero, so invert the surfaces.
    to_truth = ndimage.distance_transform_edt(~truth_surface, sampling=spacing_mm)
    to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing_mm)
    distances = np.concatenate([to_truth[pred_surface], to_pred[truth_surface]])
    return float(np.percentile(distances, 95))


def evaluate_volume(
    pred: np.ndarray, truth: np.ndarray, spacing_mm: tuple[float, ...]
) -> dict[str, float]:
    """Dice and HD95 for each structure, e.g. {"dice_LV": 0.95, "hd95_LV": 3.2, ...}."""
    scores = {}
    for value, name in STRUCTURES.items():
        scores[f"dice_{name}"] = dice(pred == value, truth == value)
        scores[f"hd95_{name}"] = hd95(pred == value, truth == value, spacing_mm)
    return scores
