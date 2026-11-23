"""Reading and checking uploaded NIfTI files. Every rejection says what was wrong in plain words."""

import tempfile
from pathlib import Path

import numpy as np

from cardiac.data.io import load_nifti

MIN_INPLANE = 64  # pixels; smaller than this cannot hold a heart at any normal spacing
MAX_INPLANE = 1024
MAX_SLICES = 40
SPACING_RANGE_MM = (0.3, 25.0)


class UploadError(ValueError):
    """An upload the app can't analyse; the message is shown to the user."""


def read_volume(data: bytes, filename: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Bytes of a .nii or .nii.gz file -> float32 (X, Y, Z) array and spacing in mm.

    The file only exists on disk inside a temporary folder that is deleted on return, because
    nibabel reads from paths."""
    name = filename.lower()
    if not (name.endswith(".nii") or name.endswith(".nii.gz")):
        raise UploadError(f"{filename}: expected a .nii or .nii.gz file")
    with tempfile.TemporaryDirectory() as folder:
        # Trust the content, not the name: gzip files start with the bytes 1f 8b.
        gzipped = data[:2] == b"\x1f\x8b"
        path = Path(folder) / ("upload.nii.gz" if gzipped else "upload.nii")
        path.write_bytes(data)
        try:
            array, spacing = load_nifti(path)
        except ValueError as error:
            message = str(error).replace(str(path), filename)
            raise UploadError(message) from None
        except Exception:
            raise UploadError(f"{filename}: could not be read as NIfTI") from None
    array = np.squeeze(np.asarray(array, dtype=np.float32))
    if array.ndim != 3:
        raise UploadError(
            f"{filename}: expected one 3D frame (X, Y, slices), got shape {tuple(array.shape)}"
        )
    x, y, z = array.shape
    if not (MIN_INPLANE <= x <= MAX_INPLANE and MIN_INPLANE <= y <= MAX_INPLANE):
        raise UploadError(
            f"{filename}: in-plane size {x} x {y} is outside {MIN_INPLANE}-{MAX_INPLANE}"
        )
    if not 1 <= z <= MAX_SLICES:
        raise UploadError(f"{filename}: {z} slices; expected 1-{MAX_SLICES} short-axis slices")
    low, high = SPACING_RANGE_MM
    if not all(low <= s <= high for s in spacing):
        raise UploadError(f"{filename}: voxel spacing {spacing} mm looks wrong for a cardiac MRI")
    if not np.isfinite(array).all():
        raise UploadError(f"{filename}: contains NaN or infinite values")
    return array, spacing


def check_pair(
    ed: tuple[np.ndarray, tuple[float, float, float]],
    es: tuple[np.ndarray, tuple[float, float, float]],
) -> None:
    if ed[0].shape != es[0].shape:
        raise UploadError(f"ED shape {ed[0].shape} and ES shape {es[0].shape} differ")
    if not np.allclose(ed[1], es[1], atol=1e-3):
        raise UploadError(f"ED spacing {ed[1]} and ES spacing {es[1]} differ")
