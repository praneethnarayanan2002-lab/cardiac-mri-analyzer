"""Agreement of model-derived clinical numbers with expert-derived ones, plus Bland-Altman plots.

Usage:
    uv run python scripts/summarize_clinical.py --split test

Reads results/<split>/clinical_expert.csv and every clinical_<model>_seed<k>.csv and
clinical_medsam_jitter<m>.csv (written by cardiac.measure). For each model and measurement it
computes Bland-Altman statistics (predicted - expert) per seed; the summary is mean ± std of each
statistic over seeds. Writes results/clinical.json. Plots show seed 0 for the U-Nets.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from cardiac.clinical import MEASUREMENTS, bland_altman
from cardiac.data.io import ACDC_GROUPS
from cardiac.plots import save

LABELS = {
    "unet_scratch": "U-Net (scratch)",
    "unet_resnet34": "U-Net (ResNet-34, ImageNet)",
    "medsam_jitter0": "MedSAM (oracle boxes)",
}
NAMES = {
    "lv_edv_ml": "LV end-diastolic volume",
    "lv_esv_ml": "LV end-systolic volume",
    "lv_ef_pct": "LV ejection fraction",
    "rv_edv_ml": "RV end-diastolic volume",
    "rv_esv_ml": "RV end-systolic volume",
    "rv_ef_pct": "RV ejection fraction",
    "myo_mass_g": "Myocardial mass",
    "wall_thickness_mean_mm": "Mean wall thickness (ED)",
    "wall_thickness_max_mm": "Max wall thickness (ED)",
}
# Categorical slots 1-5 of the default validated palette, plus a marker per group so the group is
# never told by colour alone (three of the five are below 3:1 contrast on white).
GROUP_STYLE = {
    "NOR": ("#2a78d6", "o"),
    "MINF": ("#eb6834", "s"),
    "DCM": ("#1baf7a", "^"),
    "HCM": ("#eda100", "D"),
    "RV": ("#e87ba4", "v"),
}
STATS = ("bias", "sd_diff", "loa_low", "loa_high", "mae", "pearson_r")


def load_models(folder: Path) -> dict[str, dict[int, pd.DataFrame]]:
    """{model: {seed: per-patient measurements}}; each MedSAM jitter level is its own model."""
    models: dict[str, dict[int, pd.DataFrame]] = {}
    for path in sorted(folder.glob("clinical_*.csv")):
        stem = path.stem.removeprefix("clinical_")
        table = pd.read_csv(path).set_index("pid")
        if seeded := re.fullmatch(r"(.+)_seed(\d+)", stem):
            models.setdefault(seeded[1], {})[int(seeded[2])] = table
        elif re.fullmatch(r"medsam_jitter\d+", stem):
            models[stem] = {0: table}
    return models


def agreement(per_seed: dict[int, pd.DataFrame], expert: pd.DataFrame) -> dict[str, object]:
    out: dict[str, object] = {"seeds": sorted(per_seed)}
    for column in MEASUREMENTS:
        seed_stats = {}
        for seed, table in per_seed.items():
            pred = table[column].reindex(expert.index).to_numpy()
            seed_stats[seed] = bland_altman(pred, expert[column].to_numpy())
        summary: dict[str, object] = {"unit": MEASUREMENTS[column]}
        for stat in STATS:
            values = [s[stat] for s in seed_stats.values()]
            summary[stat] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        summary["n"] = min(s["n"] for s in seed_stats.values())
        summary["n_dropped"] = max(s["n_dropped"] for s in seed_stats.values())
        summary["reference_mean"] = next(iter(seed_stats.values()))["reference_mean"]
        summary["per_seed"] = {str(k): v for k, v in seed_stats.items()}
        out[column] = summary
    return out


def bland_altman_panel(
    ax: Axes, pred: pd.Series, ref: pd.Series, groups: pd.Series, title: str, unit: str
) -> None:
    """Difference against mean, one point per patient, with bias and 95% limits of agreement."""
    keep = pred.notna() & ref.notna()
    mean, diff = ((pred + ref) / 2)[keep], (pred - ref)[keep]
    for group in ACDC_GROUPS:
        sel = groups[keep] == group
        colour, marker = GROUP_STYLE[group]
        ax.scatter(mean[sel], diff[sel], s=22, c=colour, marker=marker, label=group,
                   edgecolors="white", linewidths=0.6, zorder=3)  # fmt: skip
    ba = bland_altman(pred.to_numpy(), ref.to_numpy())
    ax.axhline(0, color="#c3c2b7", linewidth=0.8, zorder=1)
    ax.axhline(ba["bias"], color="#3d3d3a", linewidth=1.2, zorder=2)
    for edge in ("loa_low", "loa_high"):
        ax.axhline(ba[edge], color="#3d3d3a", linewidth=1, linestyle="--", zorder=2)
    x_right = ax.get_xlim()[1]
    for key, text in (("bias", "bias"), ("loa_high", "+1.96 SD"), ("loa_low", "-1.96 SD")):
        ax.annotate(f"{text} {ba[key]:+.1f}", (x_right, ba[key]), xytext=(-2, 2),
                    textcoords="offset points", ha="right", va="bottom", fontsize=7,
                    color="#3d3d3a")  # fmt: skip
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(f"mean of model and expert ({unit})", fontsize=8)
    ax.set_ylabel(f"model - expert ({unit})", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(color="#ecebe6", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_model(table: pd.DataFrame, expert: pd.DataFrame, title: str, out: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(11, 9.5))
    for ax, column in zip(axes.flat, MEASUREMENTS, strict=True):
        pred = table[column].reindex(expert.index)
        bland_altman_panel(
            ax, pred, expert[column], expert.group, NAMES[column], MEASUREMENTS[column]
        )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save(fig, out)


def plot_ef(models: dict[str, dict[int, pd.DataFrame]], expert: pd.DataFrame, out: Path) -> None:
    """LV ejection fraction, the number cardiologists look at first, for the three segmenters."""
    names = [m for m in LABELS if m in models]
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.6), sharey=True)
    for ax, model in zip(np.atleast_1d(axes), names, strict=True):
        table = models[model][min(models[model])]
        pred = table["lv_ef_pct"].reindex(expert.index)
        seed_note = " (seed 0)" if model.startswith("unet") else ""
        bland_altman_panel(ax, pred, expert.lv_ef_pct, expert.group, LABELS[model] + seed_note, "%")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.suptitle("LV ejection fraction vs. expert, test set (50 patients)", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    save(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    folder = Path("results") / args.split
    expert = pd.read_csv(folder / "clinical_expert.csv").set_index("pid")
    models = load_models(folder)
    summary = {name: agreement(per_seed, expert) for name, per_seed in models.items()}
    payload: dict[str, object] = {
        "split": args.split,
        "n_patients": len(expert),
        "reference": "measurements from the expert labels, same code and spacing",
        "difference": "model minus expert",
        "expert_median_by_group": {
            group: {c: float(expert.loc[expert.group == group, c].median()) for c in MEASUREMENTS}
            for group in ACDC_GROUPS
        },
        **summary,
    }
    out = Path("results") / ("clinical.json" if args.split == "test" else "clinical_val.json")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    for name, block in summary.items():
        print(f"  {name}")
        for column in MEASUREMENTS:
            s = block[column]
            assert isinstance(s, dict)
            print(
                f"    {column:<24} bias {s['bias']['mean']:+7.2f}  "
                f"LoA [{s['loa_low']['mean']:+7.2f}, {s['loa_high']['mean']:+7.2f}]  "
                f"MAE {s['mae']['mean']:6.2f}  r {s['pearson_r']['mean']:.3f}  "
                f"(n {s['n']}, expert mean {s['reference_mean']:.1f} {s['unit']})"
            )
    if args.split == "test":
        for model, per_seed in models.items():
            if model not in LABELS:
                continue
            seed = min(per_seed)
            note = f", seed {seed}" if model.startswith("unet") else ""
            title = f"{LABELS[model]}{note}: agreement with expert, test set (50 patients)"
            plot_model(per_seed[seed], expert, title, Path(f"figures/bland_altman_{model}.png"))
        plot_ef(models, expert, Path("figures/bland_altman_lv_ef.png"))


if __name__ == "__main__":
    main()
