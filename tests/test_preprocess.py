import numpy as np
import pytest

from cardiac.data.preprocess import crop_or_pad, normalise, prepare, restore


def _label_volume(shape: tuple[int, int, int]) -> np.ndarray:
    label = np.zeros(shape, dtype=np.uint8)
    cx, cy = shape[0] // 2, shape[1] // 2
    label[cx - 20 : cx + 20, cy - 20 : cy + 20] = 2
    label[cx - 12 : cx + 12, cy - 12 : cy + 12] = 3
    label[cx - 20 : cx + 20, cy - 45 : cy - 22] = 1
    return label


def test_prepare_output_shapes() -> None:
    image = np.random.rand(216, 256, 9).astype(np.float32)
    out = prepare(image, (1.0, 1.0, 10.0), _label_volume(image.shape), size=192)
    assert out.image.shape == (9, 192, 192)
    assert out.label is not None and out.label.shape == (9, 192, 192)
    assert out.image.dtype == np.float32


def test_normalise_gives_zero_mean_unit_std() -> None:
    volume = np.random.default_rng(0).normal(300, 80, size=(32, 32, 5))
    out = normalise(volume)
    assert out.mean() == pytest.approx(0.0, abs=1e-5)
    assert out.std() == pytest.approx(1.0, abs=1e-3)


def test_label_resampling_keeps_whole_labels() -> None:
    image = np.random.rand(150, 170, 6).astype(np.float32)
    out = prepare(image, (1.6, 1.6, 10.0), _label_volume(image.shape), target_mm=1.0)
    assert out.label is not None
    assert set(np.unique(out.label)) <= {0, 1, 2, 3}


@pytest.mark.parametrize(
    ("shape", "spacing"),
    [
        ((192, 192, 10), (1.0, 1.0, 10.0)),  # the mirror: nothing to do
        ((216, 256, 9), (1.0, 1.0, 10.0)),  # crop only
        ((150, 140, 8), (1.0, 1.0, 10.0)),  # pad only
        ((180, 200, 7), (1.5625, 1.5625, 10.0)),  # resample + crop/pad, like the original ACDC
    ],
)
def test_restore_inverts_prepare(
    shape: tuple[int, int, int], spacing: tuple[float, float, float]
) -> None:
    label = _label_volume(shape)
    prepared = prepare(np.random.rand(*shape).astype(np.float32), spacing, label, size=192)
    assert prepared.label is not None
    back = restore(prepared.label, prepared.geometry)
    assert back.shape == shape
    # Exact when no resampling happens; with resampling only a thin rim of edge voxels may differ.
    agreement = (back == label).mean()
    assert agreement == 1.0 if spacing[0] == 1.0 else agreement > 0.98


def test_crop_or_pad_handles_negative_start() -> None:
    volume = np.ones((4, 4, 1))
    out = crop_or_pad(volume, (-2, -2), 8)
    assert out.shape == (8, 8, 1)
    assert out.sum() == 16
    assert out[2:6, 2:6].all()
