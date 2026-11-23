"""Does the doubt score know when the model is wrong? results/uncertainty.json + risk-coverage plot.

Usage:
    uv run python scripts/summarize_uncertainty.py --split test

Reads results/<split>/uncertainty_<model>_seed<k>.csv (cardiac.evaluate_uncertainty) and the plain,
single-view results results/<split>/<model>_seed<k>.csv (cardiac.evaluate). One point per patient:
doubt score and error (1 - Dice, averaged over RV, MYO, LV and over ED/ES).
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from cardiac.clinical import MYOCARDIAL_DENSITY_G_PER_ML
from cardiac.plots import save
from cardiac.uncertainty import risk_coverage

DICE = ["dice_RV", "dice_MYO", "dice_LV"]
COVERAGES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def per_patient(table: pd.DataFrame) -> pd.DataFrame:
    """pid -> doubt score and error, each averaged over ED and ES."""
    t = table.assign(error=1.0 - table[DICE].mean(axis=1))
    return t.groupby("pid")[["doubt_score", "error"]].mean()


def risk_at(coverage: np.ndarray, risk: np.ndarray, c: float) -> float:
    """Risk when keeping the most confident fraction c of patients (rounded down, at least one)."""
    k = max(int(np.floor(c * len(coverage) + 1e-9)), 1)
    return float(risk[k - 1])


def size_adjusted_rho(patients: pd.DataFrame, heart_ml: pd.Series) -> float:
    """Spearman correlation of doubt and error after removing what heart size explains of each.

    Exploratory, added after seeing the results: a small heart has more border per voxel, and
    border pixels are where doubt and errors both live, so size could drive both. Ranks are
    regressed on the rank of heart size and the residuals correlated (a partial Spearman)."""
    ranks = patients.join(heart_ml.rename("size")).rank()

    def residual(column: str) -> np.ndarray:
        fit = np.polyfit(ranks["size"], ranks[column], 1)
        return np.asarray(ranks[column] - np.polyval(fit, ranks["size"]))

    return float(stats.pearsonr(residual("doubt_score"), residual("error")).statistic)


def analyse(
    table: pd.DataFrame, plain: pd.DataFrame | None, heart_ml: pd.Series | None
) -> dict[str, object]:
    patients = per_patient(table)
    rho = stats.spearmanr(patients.doubt_score, patients.error)
    coverage, risk = risk_coverage(patients.doubt_score.to_numpy(), patients.error.to_numpy())
    _, oracle = risk_coverage(patients.error.to_numpy(), patients.error.to_numpy())
    out: dict[str, object] = {
        "n_patients": len(patients),
        "spearman_rho": float(rho.statistic),
        "spearman_p": float(rho.pvalue),
        "risk_at_coverage": {f"{c:.1f}": risk_at(coverage, risk, c) for c in COVERAGES},
        "oracle_risk_at_coverage": {f"{c:.1f}": risk_at(coverage, oracle, c) for c in COVERAGES},
        # Area under the risk-coverage curve: lower is better; random ordering gives the mean error.
        "aurc": float(risk.mean()),
        "aurc_oracle": float(oracle.mean()),
        "aurc_random": float(patients.error.mean()),
        "tta_mean_dice": float(table[DICE].to_numpy().mean()),
    }
    if plain is not None:
        post = plain[plain.postprocess == "post"]
        out["single_view_mean_dice"] = float(post[DICE].to_numpy().mean())
    if heart_ml is not None:
        out["spearman_rho_size_adjusted"] = size_adjusted_rho(patients, heart_ml)
        out["size_adjusted_is_exploratory"] = True
    return out


def heart_volumes(folder: Path) -> pd.Series | None:
    """Expert end-diastolic heart volume per patient (LV + RV + myocardium), in mL."""
    path = folder / "clinical_expert.csv"
    if not path.exists():
        return None
    expert = pd.read_csv(path).set_index("pid")
    return expert.lv_edv_ml + expert.rv_edv_ml + expert.myo_mass_g / MYOCARDIAL_DENSITY_G_PER_ML


def plot(curves: dict[int, pd.DataFrame], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    shades = ["#0d366b", "#2a78d6", "#86b6ef"]
    for (seed, patients), colour in zip(sorted(curves.items()), shades, strict=False):
        coverage, risk = risk_coverage(patients.doubt_score.to_numpy(), patients.error.to_numpy())
        ax.plot(coverage * 100, risk, color=colour, linewidth=2, label=f"doubt score, seed {seed}")
    patients = curves[min(curves)]
    coverage, oracle = risk_coverage(patients.error.to_numpy(), patients.error.to_numpy())
    ax.plot(coverage * 100, oracle, color="#3d3d3a", linewidth=1.2, linestyle=":",
            label=f"perfect ordering (seed {min(curves)})")  # fmt: skip
    ax.axhline(patients.error.mean(), color="#c3c2b7", linewidth=1.2, linestyle="--",
               label=f"random ordering (seed {min(curves)})")  # fmt: skip
    ax.set(xlabel="coverage: % of patients the model keeps (most confident first)",
           ylabel="mean error of kept patients (1 - Dice)", xlim=(0, 100))  # fmt: skip
    ax.legend(fontsize=8, frameon=False)
    ax.grid(color="#ecebe6", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Risk-coverage: U-Net (ResNet-34) with TTA, test set", fontsize=10)
    save(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    folder = Path("results") / args.split
    models: dict[str, dict[int, dict[str, object]]] = {}
    curves: dict[str, dict[int, pd.DataFrame]] = {}
    heart_ml = heart_volumes(folder)
    for path in sorted(folder.glob("uncertainty_*_seed*.csv")):
        match = re.fullmatch(r"uncertainty_(.+)_seed(\d+)", path.stem)
        assert match is not None
        model, seed = match[1], int(match[2])
        table = pd.read_csv(path)
        plain_path = folder / f"{model}_seed{seed}.csv"
        plain = pd.read_csv(plain_path) if plain_path.exists() else None
        models.setdefault(model, {})[seed] = analyse(table, plain, heart_ml)
        curves.setdefault(model, {})[seed] = per_patient(table)

    payload: dict[str, object] = {"split": args.split}
    for model, per_seed in models.items():
        summary: dict[str, object] = {"seeds": sorted(per_seed)}
        for key in ("spearman_rho", "aurc", "aurc_oracle", "aurc_random", "tta_mean_dice",
                    "single_view_mean_dice", "spearman_rho_size_adjusted"):  # fmt: skip
            values = [float(s[key]) for s in per_seed.values() if key in s]  # type: ignore[arg-type]
            if values:
                std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                summary[key] = {"mean": float(np.mean(values)), "std": std}
        summary["per_seed"] = {str(k): v for k, v in per_seed.items()}
        payload[model] = summary
        print(f"{model}: {json.dumps({k: v for k, v in summary.items() if k != 'per_seed'})}")
    name = "uncertainty.json" if args.split == "test" else "uncertainty_val.json"
    (Path("results") / name).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote results/{name}")
    if args.split == "test" and "unet_resnet34" in curves:
        plot(curves["unet_resnet34"], Path("figures/risk_coverage.png"))


if __name__ == "__main__":
    main()
