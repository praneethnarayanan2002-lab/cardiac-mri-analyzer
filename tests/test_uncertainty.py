import numpy as np
import pytest
import torch
from torch import nn

from cardiac.uncertainty import (
    TRANSFORMS,
    apply_transform,
    doubt_score,
    invert_transform,
    normalised_entropy,
    risk_coverage,
    tta_probabilities,
)


def test_ten_distinct_views_including_identity() -> None:
    x = np.random.default_rng(0).random((16, 16))
    views = {apply_transform(x, a, f).round(6).tobytes() for a, f in TRANSFORMS}
    assert len(TRANSFORMS) == 10 and len(views) == 10
    assert (0.0, False) in TRANSFORMS


@pytest.mark.parametrize("flip", [False, True])
def test_flip_and_zero_rotation_invert_exactly(flip: bool) -> None:
    x = np.random.default_rng(0).random((3, 4, 6, 6))  # (S, C, H, W): only the last two move
    assert np.array_equal(invert_transform(apply_transform(x, 0.0, flip), 0.0, flip), x)


@pytest.mark.parametrize(("angle", "flip"), TRANSFORMS)
def test_inverse_restores_orientation(angle: float, flip: bool) -> None:
    """Rotations are undone by interpolation, so check a smooth, asymmetric blob away from the
    corners: it must come back where it was, not mirrored or turned."""
    y, x = np.mgrid[:64, :64]
    blob = np.exp(-((x - 40) ** 2 + (y - 26) ** 2) / 60.0)  # off-centre, so a turn would show
    back = invert_transform(apply_transform(blob, angle, flip), angle, flip)
    assert np.abs(back - blob)[8:-8, 8:-8].max() < 0.05
    assert np.unravel_index(back.argmax(), back.shape) == (26, 40)


class _ConstantModel(nn.Module):
    """Same logits everywhere, whatever it sees: every view agrees, so TTA must change nothing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((x.shape[0], 4, *x.shape[2:]))
        logits[:, 2] = 2.0
        return logits


def test_tta_of_agreeing_views_is_that_view() -> None:
    slices = np.random.default_rng(0).random((2, 16, 16)).astype(np.float32)
    probs = tta_probabilities(_ConstantModel(), slices, torch.device("cpu"))
    expected = torch.softmax(torch.tensor([0.0, 0.0, 2.0, 0.0]), dim=0).numpy()
    assert probs.shape == (2, 4, 16, 16)
    assert np.allclose(probs[0, :, 5, 7], expected, atol=1e-5)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_entropy_bounds() -> None:
    certain = np.zeros((1, 4, 2, 2))
    certain[:, 1] = 1.0
    uniform = np.full((1, 4, 2, 2), 0.25)
    assert np.allclose(normalised_entropy(certain), 0.0, atol=1e-6)
    assert np.allclose(normalised_entropy(uniform), 1.0)


def test_doubt_score_only_inside_heart() -> None:
    doubt = np.array([[255, 0], [0, 0]], dtype=np.uint8)
    label = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    assert doubt_score(doubt, label) == 1.0
    assert np.isnan(doubt_score(doubt, np.zeros_like(label)))


def test_risk_coverage() -> None:
    errors = np.array([0.4, 0.1, 0.3, 0.2])
    coverage, risk = risk_coverage(errors, errors)  # perfect ordering
    assert np.allclose(coverage, [0.25, 0.5, 0.75, 1.0])
    assert np.allclose(risk, [0.1, 0.15, 0.2, 0.25])
    assert np.all(np.diff(risk) >= 0)
