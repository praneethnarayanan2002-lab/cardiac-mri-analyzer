"""Turning a 3D frame into the 2D slices the network sees, and putting predictions back.

Forward:  (X, Y, Z) volume -> resample in-plane -> crop/pad -> clip -> z-score -> (S, H, W) slices
Inverse:  (S, H, W) labels -> un-crop -> resample back (nearest) -> (X, Y, Z) on the original grid

On the ACDC mirror the resampling and cropping are no-ops (it is already 1 mm, 192 x 192), but the
app accepts uploads with any spacing and size, so both directions are implemented and tested.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class Geometry:
    """Everything needed to map slices back onto the original grid."""

    original_shape: tuple[int, int, int]
    original_spacing_mm: tuple[float, float, float]
    resampled_shape: tuple[int, int]  # in-plane (X, Y) after resampling, before crop/pad
    crop_start: tuple[int, int]  # where the crop window starts in the resampled plane (may be < 0)
    size: int


@dataclass(frozen=True)
class Prepared:
    image: np.ndarray  # (S, size, size) float32, z-scored
    label: np.ndarray | None  # (S, size, size) uint8
    geometry: Geometry


def resample_inplane(
    volume: np.ndarray, spacing_mm: tuple[float, float, float], target_mm: float, order: int
) -> np.ndarray:
    """Resamples X and Y to `target_mm`; Z is left alone because slices are 5-10 mm apart and
    interpolating between them would invent anatomy. Use order=1 for images, 0 for labels."""
    factors = (spacing_mm[0] / target_mm, spacing_mm[1] / target_mm, 1.0)
    if np.allclose(factors, 1.0):
        return volume
    return np.asarray(ndimage.zoom(volume, factors, order=order, grid_mode=True, mode="nearest"))


def crop_or_pad(volume: np.ndarray, start: tuple[int, int], size: int) -> np.ndarray:
    """Cuts a size x size window starting at `start` from the (X, Y) plane, padding with zeros
    wherever the window falls outside the volume."""
    out = np.zeros((size, size, volume.shape[2]), dtype=volume.dtype)
    src_x0, src_y0 = max(start[0], 0), max(start[1], 0)
    src_x1 = min(start[0] + size, volume.shape[0])
    src_y1 = min(start[1] + size, volume.shape[1])
    dst_x0, dst_y0 = src_x0 - start[0], src_y0 - start[1]
    out[dst_x0 : dst_x0 + src_x1 - src_x0, dst_y0 : dst_y0 + src_y1 - src_y0] = volume[
        src_x0:src_x1, src_y0:src_y1
    ]
    return out


def normalise(volume: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Clips to the 1st-99th percentile, then z-scores the whole volume.

    MRI intensities have no fixed unit: the same tissue can be 80 on one scanner and 800 on another.
    Per-volume z-scoring puts every scan on the same footing; clipping first stops a few very bright
    pixels (fat, artefacts) from squashing everything else."""
    lo, hi = np.percentile(volume, [low, high])
    clipped = np.clip(volume, lo, hi)
    std = clipped.std()
    return np.asarray((clipped - clipped.mean()) / (std if std > 0 else 1.0), dtype=np.float32)


def prepare(
    image: np.ndarray,
    spacing_mm: tuple[float, float, float],
    label: np.ndarray | None = None,
    target_mm: float = 1.0,
    size: int = 192,
    centre_xy: tuple[float, float] | None = None,
) -> Prepared:
    """Converts one (X, Y, Z) frame into network-ready (S, size, size) slices.

    The crop is centred on the image centre, or on `centre_xy` (voxel coordinates in the original
    grid) when given: uploaded scans are not centred on the heart the way the mirror's are."""
    resampled = resample_inplane(image.astype(np.float32), spacing_mm, target_mm, order=1)
    rx, ry = resampled.shape[:2]
    if centre_xy is None:
        start = ((rx - size) // 2, (ry - size) // 2)
    else:
        cx = centre_xy[0] * spacing_mm[0] / target_mm
        cy = centre_xy[1] * spacing_mm[1] / target_mm
        start = (round(cx - size / 2), round(cy - size / 2))
    geometry = Geometry(
        original_shape=(image.shape[0], image.shape[1], image.shape[2]),
        original_spacing_mm=spacing_mm,
        resampled_shape=(rx, ry),
        crop_start=start,
        size=size,
    )
    slices = np.moveaxis(normalise(crop_or_pad(resampled, start, size)), 2, 0)
    label_slices = None
    if label is not None:
        label_rs = resample_inplane(label, spacing_mm, target_mm, order=0)
        label_slices = np.moveaxis(crop_or_pad(label_rs, start, size), 2, 0).astype(np.uint8)
    return Prepared(image=np.ascontiguousarray(slices), label=label_slices, geometry=geometry)


def restore(label_slices: np.ndarray, geometry: Geometry) -> np.ndarray:
    """Inverse of `prepare` for a predicted label map: (S, size, size) -> original (X, Y, Z)."""
    volume = np.moveaxis(label_slices, 0, 2)
    rx, ry = geometry.resampled_shape
    # Undo the crop: place the window back into a plane of the resampled size.
    back = crop_or_pad(volume, (-geometry.crop_start[0], -geometry.crop_start[1]), max(rx, ry))
    back = back[:rx, :ry]
    target = geometry.original_shape
    if back.shape[:2] == target[:2]:
        return back.astype(np.uint8)
    factors = (target[0] / rx, target[1] / ry, 1.0)
    out = ndimage.zoom(back, factors, order=0, grid_mode=True, mode="nearest")
    return crop_or_pad(out, (0, 0), max(target[:2]))[: target[0], : target[1]].astype(np.uint8)
