import math

import numpy as np
import pytest

from cardiac.clinical import (
    LV,
    MEASUREMENTS,
    MYO,
    RV,
    bland_altman,
    body_surface_area_m2,
    ejection_fraction,
    measure_patient,
    myocardial_mass_g,
    structure_volume_ml,
    voxel_volume_ml,
    wall_thickness_mm,
)

SPACING = (1.5, 1.5, 10.0)


def _radius_grid(size: int) -> np.ndarray:
    """Distance (in pixels) of each pixel centre from the middle of a size x size slice."""
    centre = (size - 1) / 2
    x, y = np.mgrid[:size, :size]
    return np.hypot(x - centre, y - centre)


def _heart(inner: float, outer: float, n_slices: int = 6, size: int = 96) -> np.ndarray:
    """LV cavity (disc of radius `inner` px) inside a myocardium ring out to `outer` px."""
    r = _radius_grid(size)
    sl = np.zeros((size, size), dtype=np.uint8)
    sl[r < outer] = MYO
    sl[r < inner] = LV
    return np.repeat(sl[:, :, None], n_slices, axis=2)


def test_voxel_volume_units() -> None:
    # 1.5 x 1.5 x 10 mm = 22.5 mm³ = 0.0225 mL
    assert voxel_volume_ml(SPACING) == pytest.approx(0.0225)


def test_cylinder_volume() -> None:
    radius, n_slices = 20.0, 8
    label = _heart(radius, radius, n_slices)  # outer == inner: cavity only
    expected = math.pi * (radius * 1.5) ** 2 * (n_slices * 10.0) / 1000.0  # π r² h, in mL
    volume = structure_volume_ml(label, LV, SPACING)
    assert volume == pytest.approx(expected, rel=0.02)  # pixel staircase vs a true circle
    assert volume == np.count_nonzero(label == LV) * 0.0225


def test_ejection_fraction() -> None:
    assert ejection_fraction(100.0, 40.0) == pytest.approx(60.0)
    assert ejection_fraction(120.0, 120.0) == 0.0
    assert math.isnan(ejection_fraction(0.0, 0.0))


def test_identical_ed_es_gives_zero_ef() -> None:
    label = _heart(15, 22)
    out = measure_patient(label, label, SPACING, 180, 80)
    assert out["lv_ef_pct"] == 0.0


def test_mass_uses_density() -> None:
    label = np.zeros((10, 10, 2), dtype=np.uint8)
    label[:5, :, :] = MYO  # 100 voxels
    assert myocardial_mass_g(label, (1.0, 1.0, 10.0)) == pytest.approx(100 * 0.01 * 1.05)


def test_ring_thickness() -> None:
    mean_t, max_t = wall_thickness_mm(_heart(15, 25), (1.0, 1.0, 10.0))
    assert mean_t == pytest.approx(10.0, abs=1.0)
    assert 10.0 <= max_t <= 11.5


def test_thickness_scales_with_spacing() -> None:
    label = _heart(15, 25)
    at_1mm = wall_thickness_mm(label, (1.0, 1.0, 10.0))
    at_2mm = wall_thickness_mm(label, (2.0, 2.0, 10.0))
    assert at_2mm == pytest.approx((2 * at_1mm[0], 2 * at_1mm[1]))


def test_thickness_skips_slices_without_cavity() -> None:
    label = _heart(15, 25, n_slices=3)
    label[:, :, 0][label[:, :, 0] == LV] = MYO  # apex cap: solid myocardium, no cavity
    assert wall_thickness_mm(label, (1.0, 1.0, 10.0)) == wall_thickness_mm(
        label[:, :, 1:], (1.0, 1.0, 10.0)
    )


def test_thickness_without_myocardium_is_nan() -> None:
    assert all(math.isnan(v) for v in wall_thickness_mm(_heart(15, 15), (1.0, 1.0, 10.0)))


def test_septum_measured_against_rv() -> None:
    label = _heart(15, 25)
    r = _radius_grid(96)
    rv = (r >= 25) & (r < 35)
    rv[:, 48:] = False  # RV crescent hugging one side of the ring
    label[rv] = RV
    assert np.count_nonzero(label == RV) > 0
    mean_t, _ = wall_thickness_mm(label, (1.0, 1.0, 10.0))
    assert mean_t == pytest.approx(10.0, abs=1.0)


def test_bsa_mosteller() -> None:
    assert body_surface_area_m2(180, 80) == pytest.approx(2.0)  # sqrt(180 x 80 / 3600)


def test_measure_patient_keys_and_indexing() -> None:
    ed, es = _heart(18, 25), _heart(12, 22)
    out = measure_patient(ed, es, SPACING, 180, 80)
    assert set(MEASUREMENTS) <= set(out)
    assert out["lv_edv_ml"] > out["lv_esv_ml"] > 0
    assert out["lv_edvi_ml_m2"] == pytest.approx(out["lv_edv_ml"] / 2.0)
    assert out["myo_mass_index_g_m2"] == pytest.approx(out["myo_mass_g"] / 2.0)
    assert out["rv_edv_ml"] == 0.0 and math.isnan(out["rv_ef_pct"])


def test_bland_altman_hand_computed() -> None:
    reference = np.array([10.0, 20.0, 30.0, 40.0])
    predicted = np.array([12.0, 21.0, 33.0, 42.0])  # differences 2, 1, 3, 2
    ba = bland_altman(predicted, reference)
    assert ba["bias"] == pytest.approx(2.0)
    sd = np.std([2, 1, 3, 2], ddof=1)
    assert ba["loa_low"] == pytest.approx(2.0 - 1.96 * sd)
    assert ba["loa_high"] == pytest.approx(2.0 + 1.96 * sd)
    assert ba["mae"] == pytest.approx(2.0)
    assert ba["n"] == 4


def test_bland_altman_drops_nan_pairs() -> None:
    ba = bland_altman(np.array([1.0, np.nan, 3.0, 4.0]), np.array([1.0, 2.0, 2.0, 5.0]))
    assert ba["n"] == 3 and ba["n_dropped"] == 1
    assert ba["bias"] == pytest.approx(0.0)
