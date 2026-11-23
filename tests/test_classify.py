import numpy as np
import pandas as pd

from cardiac.classify import FEATURES, add_derived, encode, feature_matrix, rule_based
from cardiac.data.io import ACDC_GROUPS


def _patient(**overrides: float) -> dict[str, float]:
    """A normal heart, in the units cardiac.measure writes; overrides make it abnormal."""
    row = {
        "lv_edv_ml": 140.0,
        "lv_esv_ml": 56.0,
        "lv_ef_pct": 60.0,
        "rv_edv_ml": 150.0,
        "rv_esv_ml": 70.0,
        "rv_ef_pct": 53.0,
        "myo_mass_g": 110.0,
        "wall_thickness_mean_mm": 7.0,
        "wall_thickness_max_mm": 11.0,
        "bsa_m2": 2.0,
        "lv_edvi_ml_m2": 70.0,
        "lv_esvi_ml_m2": 28.0,
        "rv_edvi_ml_m2": 75.0,
        "rv_esvi_ml_m2": 35.0,
        "myo_mass_index_g_m2": 55.0,
        "height_cm": 175.0,
        "weight_kg": 80.0,
    }
    return {**row, **overrides}


def test_rules_follow_acdc_definitions() -> None:
    table = pd.DataFrame(
        [
            _patient(),
            _patient(lv_ef_pct=25, lv_edvi_ml_m2=150),  # dilated and weak
            _patient(lv_ef_pct=35, lv_edvi_ml_m2=90),  # weak, not dilated
            _patient(rv_edvi_ml_m2=130),  # big RV
            _patient(rv_ef_pct=30),  # weak RV
            _patient(myo_mass_index_g_m2=120, wall_thickness_max_mm=18),  # thick and heavy
            _patient(myo_mass_index_g_m2=120),  # heavy but not thick: not HCM by the rules
        ]
    )
    names = [ACDC_GROUPS[i] for i in rule_based(table)]
    assert names == ["NOR", "DCM", "MINF", "RV", "RV", "HCM", "NOR"]


def test_derived_ratios() -> None:
    out = add_derived(pd.DataFrame([_patient()]))
    assert out.rv_lv_edv_ratio[0] == 150.0 / 140.0
    assert out.mass_volume_ratio_g_ml[0] == 110.0 / 140.0


def test_feature_matrix_order_and_shape() -> None:
    table = pd.DataFrame([_patient(), _patient(lv_ef_pct=np.nan)])
    x = feature_matrix(table)
    assert x.shape == (2, len(FEATURES))
    assert x[0, FEATURES.index("lv_ef_pct")] == 60.0
    assert np.isnan(x[1, FEATURES.index("lv_ef_pct")])


def test_encode_uses_group_order() -> None:
    assert encode(pd.Series(["NOR", "RV", "DCM"])).tolist() == [0, 4, 2]
