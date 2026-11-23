"""Slices as PNG bytes. Images are windowed greyscale; masks are label PNGs whose pixel values are
the labels 0-3 (coloured in the browser); doubt maps are 0-255 greyscale.

Every 2D plane goes through `to_display()`, so images, masks and doubt maps always line up, in the
same orientation as the figures (`cardiac.plots.show_slice`): anterior at the top.
"""

import io

import numpy as np
from PIL import Image


def window_volume(volume: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Whole volume to uint8 between two percentiles, so every slice uses the same grey scale."""
    lo, hi = np.percentile(volume, [low, high])
    scaled = np.clip((volume - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    return np.asarray(np.round(scaled * 255), dtype=np.uint8)


def to_display(plane: np.ndarray) -> np.ndarray:
    """(X, Y) array -> (rows, cols) image: transpose, then put row 0 at the top as imshow's
    origin="lower" would."""
    return np.ascontiguousarray(np.flipud(plane.T))


def png_bytes(plane: np.ndarray) -> bytes:
    """One uint8 (X, Y) plane as an 8-bit greyscale PNG."""
    if plane.dtype != np.uint8:
        raise ValueError(f"expected uint8, got {plane.dtype}")
    buffer = io.BytesIO()
    Image.fromarray(to_display(plane), mode="L").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
