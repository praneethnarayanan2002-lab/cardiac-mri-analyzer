from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from cardiac.data.io import load_frame, load_metadata, load_nifti, patient_files

CSV_HEADER = (
    "pid,pathology,height,weight,bmi,n_frames,ed_frame,es_frame,original_sax_spacing_x,"
    "original_sax_spacing_y,original_sax_spacing_z,n_slices,edv,esv,ef\n"
)


def _save(array: np.ndarray, spacing: tuple[float, float, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(array, affine=np.diag([*spacing, 1.0]))
    img.header.set_zooms(spacing)
    nib.save(img, path)


@pytest.fixture
def tiny_root(tmp_path: Path) -> Path:
    row = "patient001,DCM,1.84,95.0,28.0,30,1,12,1.5625,1.5625,10.0,4,0,0,0\n"
    (tmp_path / "train.csv").write_text(CSV_HEADER + row)
    meta = load_metadata(tmp_path, "train")[0]
    label = np.zeros((8, 8, 4), dtype=np.uint8)
    label[2:6, 2:6, 1:3] = 3
    for phase in ("ed", "es"):
        files = patient_files(tmp_path, meta)
        _save(np.random.rand(8, 8, 4).astype(np.float32), (1.0, 1.0, 10.0), files[f"{phase}_image"])
        _save(label, (1.0, 1.0, 10.0), files[f"{phase}_label"])
    return tmp_path


def test_metadata_units_are_converted(tiny_root: Path) -> None:
    meta = load_metadata(tiny_root, "train")[0]
    assert meta.group == "DCM"
    assert meta.height_cm == pytest.approx(184.0)
    assert meta.original_spacing_mm == (1.5625, 1.5625, 10.0)


def test_spacing_comes_from_header(tiny_root: Path) -> None:
    meta = load_metadata(tiny_root, "train")[0]
    _, spacing = load_nifti(patient_files(tiny_root, meta)["ed_image"])
    assert spacing == (1.0, 1.0, 10.0)


def test_load_frame_types_and_labels(tiny_root: Path) -> None:
    frame = load_frame(tiny_root, load_metadata(tiny_root, "train")[0], "ed")
    assert frame.image.dtype == np.float32
    assert frame.label.dtype == np.uint8
    assert set(np.unique(frame.label)) == {0, 3}
    assert int((frame.label == 3).sum()) == 4 * 4 * 2


def test_unknown_label_value_is_rejected(tiny_root: Path) -> None:
    meta = load_metadata(tiny_root, "train")[0]
    bad = np.full((8, 8, 4), 7, dtype=np.uint8)
    _save(bad, (1.0, 1.0, 10.0), patient_files(tiny_root, meta)["ed_label"])
    with pytest.raises(ValueError, match="unexpected label"):
        load_frame(tiny_root, meta, "ed")


def test_unknown_group_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "train.csv").write_text(
        CSV_HEADER + "patient001,XYZ,1.8,80,25,30,1,12,1.5,1.5,10,10,0,0,0\n"
    )
    with pytest.raises(ValueError, match="unknown group"):
        load_metadata(tmp_path, "train")
