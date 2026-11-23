"""Figures used by the docs and README. Colours match the app's design tokens."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

# Same colours as the frontend: RV blue, myocardium green, LV red.
STRUCTURE_COLOURS = {1: "#3B82F6", 2: "#22C55E", 3: "#EF4444"}
STRUCTURE_NAMES = {1: "RV cavity", 2: "Myocardium", 3: "LV cavity"}
LABEL_CMAP = ListedColormap(["#00000000", *STRUCTURE_COLOURS.values()])


def window(image: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Scales an image to 0-1 between two percentiles, so a few bright pixels don't wash it out."""
    lo, hi = np.percentile(image, [low, high])
    return np.asarray(np.clip((image - lo) / max(hi - lo, 1e-6), 0.0, 1.0))


def show_slice(ax: Axes, image: np.ndarray, label: np.ndarray | None, alpha: float = 0.45) -> None:
    """Draws one 2D slice (X, Y) in radiology orientation, optionally with its label overlay."""
    # NIfTI stores (X, Y); imshow expects (rows, cols), so transpose to put anterior at the top.
    ax.imshow(window(image).T, cmap="gray", origin="lower", interpolation="nearest")
    if label is not None:
        masked = np.ma.masked_equal(label.T, 0)
        ax.imshow(
            masked,
            cmap=LABEL_CMAP,
            vmin=0,
            vmax=3,
            alpha=alpha,
            origin="lower",
            interpolation="nearest",
        )
    ax.set_axis_off()


def structure_legend() -> list[Patch]:
    return [Patch(color=STRUCTURE_COLOURS[k], label=STRUCTURE_NAMES[k]) for k in (1, 2, 3)]


def middle_slice(label: np.ndarray) -> int:
    """Index of the slice with the most LV cavity; a fair 'typical' slice to show."""
    lv_per_slice = (label == 3).sum(axis=(0, 1))
    return int(np.argmax(lv_per_slice))


def save(fig: Figure, path: str | Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
