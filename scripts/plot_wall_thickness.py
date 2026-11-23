"""Save figures/wall_thickness_example.png: how wall thickness is measured, on two real ED slices.

Usage:
    uv run python scripts/plot_wall_thickness.py

Uses validation patients only (one dilated, one hypertrophic heart) and their expert labels.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage

from cardiac.clinical import LV, MYO, wall_thickness_mm
from cardiac.data.io import DEFAULT_DATA_ROOT, load_frame, load_metadata
from cardiac.plots import STRUCTURE_COLOURS, middle_slice, save, window

PATIENTS = ("patient001", "patient026")  # DCM and HCM, both in the validation split
MARGIN_PX = 10
# One hue, light to dark: thicker wall = darker orange.
THICKNESS_CMAP = LinearSegmentedColormap.from_list("thickness", ["#f9d2bd", "#eb6834", "#7a2c0c"])


def main() -> None:
    metas = {m.pid: m for m in load_metadata(DEFAULT_DATA_ROOT, "train")}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    for ax, pid in zip(axes, PATIENTS, strict=True):
        meta = metas[pid]
        frame = load_frame(DEFAULT_DATA_ROOT, meta, "ed")
        k = middle_slice(frame.label)
        sl, img = frame.label[:, :, k], frame.image[:, :, k]
        cavity, myo = sl == LV, sl == MYO
        cross = ndimage.generate_binary_structure(2, 1)
        border = myo & ndimage.binary_dilation(~(cavity | myo), structure=cross)
        dist = np.asarray(ndimage.distance_transform_edt(~cavity, sampling=frame.spacing_mm[:2]))
        # A square crop around the heart plus a margin, so big and small hearts both fill a panel.
        heart_x, heart_y = np.nonzero(cavity | myo)
        cx, cy = (heart_x.min() + heart_x.max()) // 2, (heart_y.min() + heart_y.max()) // 2
        half = max(np.ptp(heart_x), np.ptp(heart_y)) // 2 + MARGIN_PX
        xs, ys = slice(cx - half, cx + half), slice(cy - half, cy + half)

        ax.imshow(window(img)[xs, ys].T, cmap="gray", origin="lower", interpolation="nearest")
        ax.contour(cavity[xs, ys].T, levels=[0.5], colors=STRUCTURE_COLOURS[LV], linewidths=1)
        bx, by = np.nonzero(border[xs, ys])
        points = ax.scatter(bx, by, c=dist[xs, ys][bx, by], cmap=THICKNESS_CMAP, vmin=0, vmax=25,
                            s=16, edgecolors="none", zorder=3)  # fmt: skip
        mean_t, max_t = wall_thickness_mm(frame.label, frame.spacing_mm)
        ax.set_title(
            f"{pid} ({meta.group}), ED slice {k}\n"
            f"whole heart: mean {mean_t:.1f} mm, max {max_t:.1f} mm",
            fontsize=9,
        )
        ax.set_axis_off()
    bar = fig.colorbar(points, ax=axes, fraction=0.03, pad=0.02)
    bar.set_label("distance to nearest LV cavity pixel (mm)", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    fig.suptitle(
        "Wall thickness: each outer-border pixel of the myocardium (dots) is measured to the "
        "cavity (red line)",
        fontsize=9,
    )
    save(fig, Path("figures/wall_thickness_example.png"))
    print("saved figures/wall_thickness_example.png")


if __name__ == "__main__":
    main()
