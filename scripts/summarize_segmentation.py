"""Combine per-run results into results/segmentation.json and the comparison figures.

Usage:
    uv run python scripts/summarize_segmentation.py --split test

Reads results/<split>/<model>_seed<k>.csv and medsam_jitter<m>.csv (written by cardiac.evaluate and
cardiac.evaluate_medsam). Headline numbers are post-processed. For each U-Net, a seed's score is the
mean over all patients and phases; the table reports mean ± std of those seed scores.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from cardiac.data.io import ACDC_GROUPS
from cardiac.plots import STRUCTURE_COLOURS, save

UNETS = {"unet_scratch": "U-Net (scratch)", "unet_resnet34": "U-Net (ResNet-34, ImageNet)"}
STRUCTS = ("RV", "MYO", "LV")
METRICS = [f"{m}_{s}" for m in ("dice", "hd95") for s in STRUCTS]


def load_runs(folder: Path) -> dict[str, dict[int, pd.DataFrame]]:
    """{model: {seed: per-patient table (post-processed rows only)}}.

    U-Net files are <model>_seed<k>; each MedSAM jitter level is its own model with one "seed",
    because MedSAM is not trained and its jitter levels must not be averaged together."""
    runs: dict[str, dict[int, pd.DataFrame]] = {}
    for path in sorted(folder.glob("*.csv")):
        table = pd.read_csv(path)
        table = table[table.postprocess == "post"]
        if seeded := re.fullmatch(r"(.+)_seed(\d+)", path.stem):
            runs.setdefault(seeded[1], {})[int(seeded[2])] = table
        elif re.fullmatch(r"medsam_jitter\d+", path.stem):
            runs[path.stem] = {0: table}
    return runs


def seed_summary(per_seed: dict[int, pd.DataFrame]) -> dict[str, object]:
    """Mean ± std across seeds of each seed's mean, overall, by phase and by group."""

    def across(select: str | None, column: str, value: str) -> dict[str, float]:
        means = [
            float((t if select is None else t[t[select] == value])[column].mean())
            for t in per_seed.values()
        ]
        return {
            "mean": float(np.mean(means)),
            "std": float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
        }

    out: dict[str, object] = {"seeds": sorted(per_seed)}
    out["overall"] = {c: across(None, c, "") for c in METRICS}
    out["by_phase"] = {p: {c: across("phase", c, p) for c in METRICS} for p in ("ed", "es")}
    out["by_group"] = {g: {c: across("group", c, g)["mean"] for c in METRICS} for g in ACDC_GROUPS}
    seed_means = [t[[f"dice_{s}" for s in STRUCTS]].to_numpy().mean() for t in per_seed.values()]
    out["mean_dice"] = {
        "mean": float(np.mean(seed_means)),
        "std": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0,
    }
    return out


def per_patient_dice(per_seed: dict[int, pd.DataFrame], structure: str) -> pd.Series:
    """Dice per patient, averaged over seeds and over ED/ES (one number per patient)."""
    stacked = pd.concat(per_seed.values())
    return stacked.groupby("pid")[f"dice_{structure}"].mean()


def compare_unets(runs: dict[str, dict[int, pd.DataFrame]]) -> dict[str, object]:
    """Paired Wilcoxon signed-rank test, pretrained vs scratch, one pair per patient."""
    result: dict[str, object] = {}
    for structure in STRUCTS:
        a = per_patient_dice(runs["unet_resnet34"], structure)
        b = per_patient_dice(runs["unet_scratch"], structure)
        diff = (a - b.reindex(a.index)).dropna()
        test = stats.wilcoxon(diff)
        result[structure] = {
            "mean_difference": float(diff.mean()),
            "patients_better_with_pretrained": int((diff > 0).sum()),
            "n_patients": len(diff),
            "wilcoxon_p": float(test.pvalue),
        }
    return result


def plot_bars(summary: dict[str, dict[str, object]], out: Path) -> None:
    models = [m for m in (*UNETS, "medsam_jitter0") if m in summary]
    labels = {**UNETS, "medsam_jitter0": "MedSAM (oracle boxes)"}
    fig, ax = plt.subplots(figsize=(8, 3.6))
    width = 0.8 / len(models)
    shades = ["#0E7C86", "#3B82F6", "#B45309"]
    for i, model in enumerate(models):
        overall = summary[model]["overall"]
        assert isinstance(overall, dict)
        means = [overall[f"dice_{s}"]["mean"] for s in STRUCTS]
        stds = [overall[f"dice_{s}"]["std"] for s in STRUCTS]
        x = np.arange(len(STRUCTS)) + (i - (len(models) - 1) / 2) * width
        ax.bar(x, means, width, yerr=stds, capsize=3, label=labels[model], color=shades[i])
        for xi, m in zip(x, means, strict=True):
            ax.text(xi, m + 0.01, f"{m:.3f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(STRUCTS)), STRUCTS)
    ax.set(ylabel="Dice (3D, per patient)", ylim=(0, 1.08))
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(
        "Test set (50 patients, ED + ES). Error bars: std over 3 training seeds", fontsize=9
    )
    save(fig, out)


def plot_jitter(summary: dict[str, dict[str, object]], jitters: list[int], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    colours = {"RV": STRUCTURE_COLOURS[1], "MYO": STRUCTURE_COLOURS[2], "LV": STRUCTURE_COLOURS[3]}
    for s in STRUCTS:
        ys = [summary[f"medsam_jitter{j}"]["overall"][f"dice_{s}"]["mean"] for j in jitters]  # type: ignore[index]
        ax.plot(jitters, ys, marker="o", color=colours[s], label=f"MedSAM {s}")
        if "unet_resnet34" in summary:
            ref = summary["unet_resnet34"]["overall"][f"dice_{s}"]["mean"]  # type: ignore[index]
            ax.axhline(ref, color=colours[s], linestyle=":", linewidth=1)
    ax.set(
        xlabel="box jitter: each edge moved by up to ± this many pixels (1 px = 1 mm)",
        ylabel="Dice",
        ylim=(0, 1),
        xticks=jitters,
    )
    ax.legend(fontsize=8, loc="center right")
    ax.set_title("MedSAM vs. sloppier prompts (dotted: pretrained U-Net, no prompt)", fontsize=9)
    save(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    runs = load_runs(Path("results") / args.split)
    summary = {name: seed_summary(per_seed) for name, per_seed in runs.items()}
    payload: dict[str, object] = {"split": args.split, "headline": "post-processed", **summary}
    if all(m in runs for m in UNETS):
        payload["pretrained_vs_scratch"] = compare_unets(runs)
    out = Path("results") / (
        "segmentation.json" if args.split == "test" else "segmentation_val.json"
    )
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    for name, block in summary.items():
        md = block["mean_dice"]
        overall = block["overall"]
        assert isinstance(md, dict) and isinstance(overall, dict)
        parts = "  ".join(
            f"{s} {overall[f'dice_{s}']['mean']:.3f}±{overall[f'dice_{s}']['std']:.3f}"
            for s in STRUCTS
        )
        print(f"  {name:<16} seeds {block['seeds']}  {parts}  mean {md['mean']:.4f}")
    if args.split == "test":
        plot_bars(summary, Path("figures/test_dice_by_model.png"))
        jitters = sorted(int(m.removeprefix("medsam_jitter")) for m in runs if "jitter" in m)
        if jitters:
            plot_jitter(summary, jitters, Path("figures/medsam_jitter.png"))


if __name__ == "__main__":
    main()
