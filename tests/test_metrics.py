import math

import numpy as np
import pytest

from cardiac.metrics import dice, evaluate_volume, hd95


def _cube(shape: tuple[int, int, int], lo: int, hi: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[lo:hi, lo:hi, lo:hi] = True
    return mask


def test_dice_identical_is_one() -> None:
    mask = _cube((10, 10, 10), 2, 6)
    assert dice(mask, mask) == 1.0


def test_dice_disjoint_is_zero() -> None:
    assert dice(_cube((10, 10, 10), 0, 3), _cube((10, 10, 10), 5, 8)) == 0.0


def test_dice_hand_computed() -> None:
    a = np.zeros((4, 4), dtype=bool)
    b = np.zeros((4, 4), dtype=bool)
    a[:2, :] = True  # 8 pixels
    b[1:3, :] = True  # 8 pixels, 4 shared
    assert dice(a, b) == pytest.approx(2 * 4 / (8 + 8))


def test_dice_both_empty_is_one() -> None:
    empty = np.zeros((5, 5, 5), dtype=bool)
    assert dice(empty, empty) == 1.0


def test_hd95_identical_is_zero() -> None:
    mask = _cube((12, 12, 12), 3, 9)
    assert hd95(mask, mask, (1.0, 1.0, 1.0)) == 0.0


def test_hd95_uses_spacing() -> None:
    # Shift a slab by one voxel along Z. With 10 mm slices, the surfaces move 10 mm apart, not 1.
    truth = np.zeros((8, 8, 8), dtype=bool)
    truth[2:6, 2:6, 2:5] = True
    shifted = np.roll(truth, 1, axis=2)
    assert hd95(shifted, truth, (1.0, 1.0, 1.0)) == pytest.approx(1.0)
    assert hd95(shifted, truth, (1.0, 1.0, 10.0)) == pytest.approx(10.0)


def test_hd95_empty_is_nan() -> None:
    mask = _cube((6, 6, 6), 1, 4)
    assert math.isnan(hd95(np.zeros_like(mask), mask, (1.0, 1.0, 1.0)))


def test_evaluate_volume_keys() -> None:
    label = np.zeros((8, 8, 4), dtype=np.uint8)
    label[1:3, 1:3] = 1
    label[3:5, 3:5] = 2
    label[5:7, 5:7] = 3
    scores = evaluate_volume(label, label, (1.0, 1.0, 10.0))
    assert set(scores) == {f"{m}_{s}" for m in ("dice", "hd95") for s in ("RV", "MYO", "LV")}
    assert all(scores[f"dice_{s}"] == 1.0 for s in ("RV", "MYO", "LV"))
