"""Diagnosis from clinical measurements: features, a rule baseline, logistic regression, XGBoost.

The five ACDC groups are predicted from one row of numbers per patient (cardiac.clinical). Models
are trained on features from the expert masks of the 100 training patients only.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from cardiac.data.io import ACDC_GROUPS

# Model inputs, in a fixed order. Units: mL, %, g, mm, mL/m², g/m², cm, kg; the two ratios are
# unitless (RV/LV) and g/mL (mass/volume).
FEATURES = (
    "lv_edv_ml",
    "lv_esv_ml",
    "lv_ef_pct",
    "rv_edv_ml",
    "rv_esv_ml",
    "rv_ef_pct",
    "myo_mass_g",
    "wall_thickness_mean_mm",
    "wall_thickness_max_mm",
    "lv_edvi_ml_m2",
    "lv_esvi_ml_m2",
    "rv_edvi_ml_m2",
    "rv_esvi_ml_m2",
    "myo_mass_index_g_m2",
    "rv_lv_edv_ratio",
    "mass_volume_ratio_g_ml",
    "height_cm",
    "weight_kg",
)


@dataclass(frozen=True)
class ClassifierConfig:
    cv_folds: int
    seed: int
    xgb_grid: dict[str, list[float]]
    logreg_c: list[float]


def load_classifier_config(path: Path = Path("configs/classifier.yaml")) -> ClassifierConfig:
    raw = yaml.safe_load(path.read_text())
    return ClassifierConfig(
        cv_folds=int(raw["cv_folds"]),
        seed=int(raw["seed"]),
        xgb_grid={k: list(v) for k, v in raw["xgb_grid"].items()},
        logreg_c=[float(c) for c in raw["logreg_c"]],
    )


def add_derived(table: pd.DataFrame) -> pd.DataFrame:
    """Adds the two ratio features to a cardiac.measure table (one row per patient)."""
    out = table.copy()
    out["rv_lv_edv_ratio"] = out.rv_edv_ml / out.lv_edv_ml
    out["mass_volume_ratio_g_ml"] = out.myo_mass_g / out.lv_edv_ml
    return out


def feature_matrix(table: pd.DataFrame) -> np.ndarray:
    """(patients, len(FEATURES)) float array; NaN stays NaN (XGBoost handles missing values)."""
    return np.asarray(add_derived(table)[list(FEATURES)].to_numpy(dtype=float))


def encode(groups: pd.Series) -> np.ndarray:
    return np.asarray(groups.map({g: i for i, g in enumerate(ACDC_GROUPS)}).to_numpy())


def rule_based(table: pd.DataFrame) -> np.ndarray:
    """The ACDC group definitions written as if-then rules, with the published thresholds, untuned.

    DCM: LV EF < 40% and LV EDV > 100 mL/m². MINF: LV EF < 40% (with abnormal wall motion, which we
    can't measure, so every other low-EF heart). RV: RV EDV > 110 mL/m² or RV EF < 40%. HCM: mass
    > 110 g/m² and wall > 15 mm with normal EF. Everything else: NOR. Returns group indices."""
    t = add_derived(table)
    names = []
    for row in t.itertuples():
        if row.lv_ef_pct < 40:
            names.append("DCM" if row.lv_edvi_ml_m2 > 100 else "MINF")
        elif row.rv_edvi_ml_m2 > 110 or row.rv_ef_pct < 40:
            names.append("RV")
        elif row.myo_mass_index_g_m2 > 110 and row.wall_thickness_max_mm > 15:
            names.append("HCM")
        else:
            names.append("NOR")
    return encode(pd.Series(names))


def _cv(config: ClassifierConfig) -> StratifiedKFold:
    return StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)


def fit_xgboost(x: np.ndarray, y: np.ndarray, config: ClassifierConfig) -> GridSearchCV:
    """Grid search with stratified k-fold CV; the best setting is refitted on all patients."""
    model = XGBClassifier(
        objective="multi:softprob",
        tree_method="hist",
        random_state=config.seed,
        n_jobs=1,  # one thread: identical trees on every run
    )
    search = GridSearchCV(model, config.xgb_grid, cv=_cv(config), scoring="accuracy")
    return search.fit(x, y)


def fit_logistic(x: np.ndarray, y: np.ndarray, config: ClassifierConfig) -> GridSearchCV:
    """Standardised features + multinomial logistic regression; NaNs are not expected here."""
    pipe: Pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10_000))
    grid = {"logisticregression__C": config.logreg_c}
    search = GridSearchCV(pipe, grid, cv=_cv(config), scoring="accuracy")
    return search.fit(x, y)
