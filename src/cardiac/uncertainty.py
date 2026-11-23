"""Test-time augmentation (TTA) uncertainty: a doubt map per voxel and a doubt score per patient.

Each slice is shown to the model 10 times: rotated by -10, -5, 0, +5 and +10 degrees, each with and
without a left-right flip. Each prediction is turned back to the original orientation, the 10 are
averaged, and the disagreement between them becomes the doubt.

These are the kinds of change the model saw during training (rotations up to 15 degrees, the same
flip axis). The plan's 90-degree rotations were tried first and dropped: they put the RV on a side
the model never saw, and single views lost up to 0.4 Dice (docs/decisions.md D26).
"""

import math

import numpy as np
import torch
from scipy import ndimage
from torch import nn

from cardiac.data.preprocess import prepare, restore
from cardiac.infer import postprocess, predict_probabilities

ANGLES_DEG = (-10.0, -5.0, 0.0, 5.0, 10.0)
TRANSFORMS = tuple((angle, flip) for angle in ANGLES_DEG for flip in (False, True))  # 10 views


def _rotate(x: np.ndarray, angle: float) -> np.ndarray:
    """Rotates the last two axes about the centre; linear interpolation, edges repeated."""
    if angle == 0:
        return x
    return np.asarray(
        ndimage.rotate(x, angle, axes=(-2, -1), reshape=False, order=1, mode="nearest")
    )


def apply_transform(x: np.ndarray, angle: float, flip: bool) -> np.ndarray:
    """Rotates the last two axes by `angle` degrees, then optionally flips the last axis (the
    same axis the training augmentation flips)."""
    out = _rotate(x, angle)
    return np.flip(out, axis=-1) if flip else out


def invert_transform(x: np.ndarray, angle: float, flip: bool) -> np.ndarray:
    """Inverse of apply_transform: undo the flip (exact), then rotate back (exact for 0 degrees,
    otherwise up to interpolation)."""
    out = np.flip(x, axis=-1) if flip else x
    return _rotate(out, -angle)


def tta_probabilities(model: nn.Module, slices: np.ndarray, device: torch.device) -> np.ndarray:
    """(S, H, W) normalised slices -> (S, 4, H, W) softmax probabilities averaged over all views."""
    total = np.zeros((slices.shape[0], 4, *slices.shape[1:]), dtype=np.float64)
    for angle, flip in TRANSFORMS:
        view = np.ascontiguousarray(apply_transform(slices, angle, flip), dtype=np.float32)
        probs = predict_probabilities(model, view, device)
        total += invert_transform(probs, angle, flip)
    return (total / len(TRANSFORMS)).astype(np.float32)


def normalised_entropy(probs: np.ndarray, axis: int = 1) -> np.ndarray:
    """Entropy of the class probabilities divided by its maximum, log(classes): 0 = certain,
    1 = all classes equally likely."""
    p = np.clip(probs, 1e-8, 1.0)
    return np.asarray(-(p * np.log(p)).sum(axis=axis) / math.log(probs.shape[axis]))


def predict_with_doubt(
    model: nn.Module,
    image: np.ndarray,
    spacing_mm: tuple[float, float, float],
    device: torch.device,
    target_mm: float = 1.0,
    size: int = 192,
) -> tuple[np.ndarray, np.ndarray]:
    """(X, Y, Z) image -> (TTA label map uint8, doubt map uint8 0-255), both on the image's grid.

    The doubt map is normalised entropy x 255, stored as uint8 so it can go through the same
    restore() as a label map and be sent to the browser as a PNG."""
    prepared = prepare(image, spacing_mm, None, target_mm, size)
    probs = tta_probabilities(model, prepared.image, device)
    label = postprocess(restore(probs.argmax(axis=1).astype(np.uint8), prepared.geometry))
    doubt = np.round(normalised_entropy(probs) * 255).astype(np.uint8)
    return label, restore(doubt, prepared.geometry)


def doubt_score(doubt: np.ndarray, label: np.ndarray) -> float:
    """Mean doubt (0-1) over the voxels predicted as heart; NaN if nothing was predicted."""
    heart = label > 0
    return float(doubt[heart].mean() / 255.0) if heart.any() else math.nan


def risk_coverage(scores: np.ndarray, errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the most confident fraction of patients; report the mean error of those kept.

    Returns (coverage, risk) with coverage = k / n for k = 1..n, patients kept in order of
    increasing doubt score."""
    order = np.argsort(scores, kind="stable")
    kept_errors = np.asarray(errors, float)[order]
    k = np.arange(1, len(order) + 1)
    return k / len(order), np.cumsum(kept_errors) / k
