"""Save figures/doubt_maps.png: the most and least doubtful test patients, with their doubt maps.

Usage:
    uv run python scripts/plot_doubt_maps.py --run runs/unet_resnet34/<timestamp>-seed0

Picks patients by the end-diastole doubt score in results/test/uncertainty_<run>.csv (written by
cardiac.evaluate_uncertainty), so it shows the extremes, not hand-picked examples.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from cardiac.data.io import DEFAULT_DATA_ROOT, load_frame, load_metadata
from cardiac.infer import load_checkpoint, pick_device
from cardiac.metrics import dice
from cardiac.plots import middle_slice, save, show_slice, structure_legend
from cardiac.uncertainty import predict_with_doubt

# One hue, light to dark: more doubt = darker purple. Pixels with almost no doubt are left out, so
# the scan shows through everywhere the model is sure.
DOUBT_CMAP = LinearSegmentedColormap.from_list("doubt", ["#d9cdf3", "#9a7ad6", "#3b1f78"])
SHOW_ABOVE = 0.02


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    device = pick_device()
    model, config = load_checkpoint(args.run / "best.pt", device)
    table = pd.read_csv(f"results/test/uncertainty_{config.name}_seed{config.train.seed}.csv")
    ed = table[table.phase == "ed"].sort_values("doubt_score")
    picks = {"least doubtful": ed.iloc[0], "most doubtful": ed.iloc[-1]}
    metas = {m.pid: m for m in load_metadata(DEFAULT_DATA_ROOT, "test")}

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for row_axes, (title, row) in zip(axes, picks.items(), strict=True):
        meta = metas[row.pid]
        frame = load_frame(DEFAULT_DATA_ROOT, meta, "ed")
        label, doubt = predict_with_doubt(model, frame.image, frame.spacing_mm, device)
        k = middle_slice(frame.label)
        mean_dice = np.mean([dice(label == s, frame.label == s) for s in (1, 2, 3)])
        show_slice(row_axes[0], frame.image[:, :, k], frame.label[:, :, k])
        row_axes[0].set_title(f"{title}: {meta.pid} ({meta.group})\nexpert, ED slice {k}",
                              fontsize=8)  # fmt: skip
        show_slice(row_axes[1], frame.image[:, :, k], label[:, :, k])
        row_axes[1].set_title(f"U-Net + TTA\nmean Dice {mean_dice:.3f} (whole heart)", fontsize=8)
        show_slice(row_axes[2], frame.image[:, :, k], None)
        plane = np.ma.masked_less(doubt[:, :, k].T / 255.0, SHOW_ABOVE)
        shown = row_axes[2].imshow(plane, cmap=DOUBT_CMAP, vmin=0, vmax=0.6, origin="lower",
                                   interpolation="nearest")  # fmt: skip
        row_axes[2].set_title(f"doubt map\npatient score {row.doubt_score:.3f}", fontsize=8)
    bar = fig.colorbar(shown, ax=axes[:, 2], fraction=0.05, pad=0.02)
    bar.set_label("doubt (normalised entropy)", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    fig.legend(handles=structure_legend(), loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(
        "Test patients with the lowest and highest doubt score at end-diastole", fontsize=10
    )
    save(fig, Path("figures/doubt_maps.png"))
    print("saved figures/doubt_maps.png")


if __name__ == "__main__":
    main()
