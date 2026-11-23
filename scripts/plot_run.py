"""Training curves and validation overlays for one run. Usage:

    uv run python scripts/plot_run.py --run runs/unet_scratch/<timestamp>-seed0

Saves figures/training_curves_<name>.png and figures/val_overlays_<name>.png.
The overlay patients are picked by a fixed rule (median, worst and best validation Dice from
results/val/<name>_seed<k>.csv), not by looking at the pictures.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from cardiac.data.io import load_frame, load_metadata
from cardiac.infer import load_checkpoint, pick_device, predict_volume
from cardiac.plots import middle_slice, save, show_slice, structure_legend


def plot_curves(log: pd.DataFrame, name: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    fig.subplots_adjust(wspace=0.35)
    axes[0].plot(log.epoch, log.train_loss, color="#0E7C86")
    axes[0].set(xlabel="epoch", ylabel="training loss (Dice + CE)")
    for column, colour in (("RV", "#3B82F6"), ("MYO", "#22C55E"), ("LV", "#EF4444")):
        axes[1].plot(log.epoch, log[f"val_dice_{column}"], color=colour, label=column)
    axes[1].plot(log.epoch, log.val_dice_mean, color="black", label="mean")
    best = log.loc[log.val_dice_mean.idxmax()]
    axes[1].axvline(best.epoch, color="grey", linestyle=":", label=f"best epoch {int(best.epoch)}")
    axes[1].set(xlabel="epoch", ylabel="validation Dice (3D)", ylim=(0.5, 1.0))
    axes[1].legend(fontsize=7, loc="lower right")
    axes[2].plot(log.epoch, log.lr, color="#5B6B7C")
    axes[2].set(xlabel="epoch", ylabel="learning rate")
    fig.suptitle(f"{name}: training run", fontsize=10)
    save(fig, out)


def plot_overlays(run: Path, per_patient: pd.DataFrame, name: str, out: Path) -> None:
    device = pick_device()
    model, config = load_checkpoint(run / "best.pt", device)
    ed = per_patient[(per_patient.phase == "ed") & (per_patient.postprocess == "post")].copy()
    ed["mean_dice"] = ed[["dice_RV", "dice_MYO", "dice_LV"]].mean(axis=1)
    ranked = ed.sort_values("mean_dice").reset_index(drop=True)
    picks = {
        "worst": ranked.iloc[0],
        "median": ranked.iloc[len(ranked) // 2],
        "best": ranked.iloc[-1],
    }
    metas = {m.pid: m for m in load_metadata(Path(config.data.root), "train")}
    fig, axes = plt.subplots(3, 3, figsize=(9, 9.6))
    for row, (label, pick) in enumerate(picks.items()):
        frame = load_frame(Path(config.data.root), metas[pick.pid], "ed")
        pred = predict_volume(model, frame.image, frame.spacing_mm, device)
        k = middle_slice(frame.label)
        show_slice(axes[row, 0], frame.image[:, :, k], None)
        show_slice(axes[row, 1], frame.image[:, :, k], frame.label[:, :, k])
        show_slice(axes[row, 2], frame.image[:, :, k], pred[:, :, k])
        axes[row, 0].set_title(f"{label}: {pick.pid} ({pick.group}), ED slice {k}", fontsize=8)
        axes[row, 1].set_title("expert", fontsize=8)
        axes[row, 2].set_title(f"U-Net (3D mean Dice {pick.mean_dice:.3f})", fontsize=8)
    fig.legend(handles=structure_legend(), loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"{name}: validation patients chosen by Dice rank", fontsize=10)
    save(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    _, config = load_checkpoint(args.run / "best.pt", pick_device())
    name = f"{config.name}_seed{config.train.seed}"
    plot_curves(
        pd.read_csv(args.run / "log.csv"), name, Path(f"figures/training_curves_{name}.png")
    )
    per_patient = pd.read_csv(Path("results/val") / f"{name}.csv")
    plot_overlays(args.run, per_patient, name, Path(f"figures/val_overlays_{name}.png"))
    print(f"saved figures/training_curves_{name}.png and figures/val_overlays_{name}.png")


if __name__ == "__main__":
    main()
