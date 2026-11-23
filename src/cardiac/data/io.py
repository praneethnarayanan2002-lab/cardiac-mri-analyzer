"""Reading the ACDC mirror: patient metadata, NIfTI volumes and their voxel spacing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
import pandas as pd

DEFAULT_DATA_ROOT = Path("data/raw/acdc")
ACDC_GROUPS = ("NOR", "MINF", "DCM", "HCM", "RV")
LABELS = {0: "background", 1: "RV", 2: "MYO", 3: "LV"}

Split = Literal["train", "test"]
Phase = Literal["ed", "es"]


@dataclass(frozen=True)
class PatientMeta:
    """One row of train.csv / test.csv, converted to the units the rest of the code uses."""

    pid: str
    split: Split
    group: str
    height_cm: float
    weight_kg: float
    n_frames: int
    ed_frame: int
    es_frame: int
    original_spacing_mm: tuple[float, float, float]
    n_slices: int


@dataclass(frozen=True)
class Frame:
    """A 3D short-axis frame with its expert label, both shaped (X, Y, Z)."""

    image: np.ndarray  # float32
    label: np.ndarray  # uint8, values in LABELS
    spacing_mm: tuple[float, float, float]


def load_metadata(root: Path, split: Split) -> list[PatientMeta]:
    table = pd.read_csv(root / f"{split}.csv")
    return [_row_to_meta(row, split) for row in table.to_dict("records")]


def _row_to_meta(row: dict[str, object], split: Split) -> PatientMeta:
    group = str(row["pathology"])
    if group not in ACDC_GROUPS:
        raise ValueError(f"{row['pid']}: unknown group {group!r}")
    return PatientMeta(
        pid=str(row["pid"]),
        split=split,
        group=group,
        # The mirror stores height in metres; ACDC's Info.cfg and the BSA formula use centimetres.
        height_cm=float(str(row["height"])) * 100.0,
        weight_kg=float(str(row["weight"])),
        n_frames=int(str(row["n_frames"])),
        ed_frame=int(str(row["ed_frame"])),
        es_frame=int(str(row["es_frame"])),
        original_spacing_mm=(
            float(str(row["original_sax_spacing_x"])),
            float(str(row["original_sax_spacing_y"])),
            float(str(row["original_sax_spacing_z"])),
        ),
        n_slices=int(str(row["n_slices"])),
    )


def patient_files(root: Path, meta: PatientMeta) -> dict[str, Path]:
    folder = root / meta.split / meta.pid
    stem = f"{meta.pid}_sax"
    return {
        "ed_image": folder / f"{stem}_ed.nii.gz",
        "ed_label": folder / f"{stem}_ed_gt.nii.gz",
        "es_image": folder / f"{stem}_es.nii.gz",
        "es_label": folder / f"{stem}_es_gt.nii.gz",
        "cine": folder / f"{stem}_t.nii.gz",
    }


def load_nifti(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Returns the voxel array and the spacing in mm read from the NIfTI header."""
    img = nib.load(path)
    if not isinstance(img, nib.Nifti1Image | nib.Nifti2Image):
        raise ValueError(f"{path}: not a NIfTI image")
    zooms = img.header.get_zooms()
    if len(zooms) < 3 or any(z <= 0 for z in zooms[:3]):
        raise ValueError(f"{path}: header has no valid voxel spacing ({zooms})")
    spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    return np.asanyarray(img.dataobj), spacing


def load_frame(root: Path, meta: PatientMeta, phase: Phase) -> Frame:
    files = patient_files(root, meta)
    image, spacing = load_nifti(files[f"{phase}_image"])
    label, label_spacing = load_nifti(files[f"{phase}_label"])
    if image.shape != label.shape:
        raise ValueError(f"{meta.pid} {phase}: image {image.shape} != label {label.shape}")
    if not np.allclose(spacing, label_spacing):
        raise ValueError(f"{meta.pid} {phase}: image and label spacing differ")
    label = np.rint(label).astype(np.uint8)
    unexpected = set(np.unique(label).tolist()) - set(LABELS)
    if unexpected:
        raise ValueError(f"{meta.pid} {phase}: unexpected label values {sorted(unexpected)}")
    return Frame(image=image.astype(np.float32), label=label, spacing_mm=spacing)
