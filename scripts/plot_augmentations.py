"""Save figures/augmentation_examples.png: one training slice after several random augmentations.

Usage:
    uv run python scripts/plot_augmentations.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cardiac.data.augment import training_transforms
from cardiac.data.io import DEFAULT_DATA_ROOT, load_frame, load_metadata
from cardiac.data.preprocess import prepare
from cardiac.data.splits import load_splits
from cardiac.plots import middle_slice, save, show_slice, structure_legend


def main() -> None:
    train_ids = load_splits()["train"]
    meta = next(m for m in load_metadata(DEFAULT_DATA_ROOT, "train") if m.pid == train_ids[0])
    frame = load_frame(DEFAULT_DATA_ROOT, meta, "ed")
    prepared = prepare(frame.image, frame.spacing_mm, frame.label)
    assert prepared.label is not None
    k = middle_slice(np.moveaxis(prepared.label, 0, 2))
    sample = {"image": prepared.image[k][None], "label": prepared.label[k][None].astype(np.float32)}

    transforms = training_transforms()
    transforms.set_random_state(0)
    fig, axes = plt.subplots(2, 4, figsize=(11, 6))
    show_slice(axes[0, 0], sample["image"][0], sample["label"][0].astype(np.uint8))
    axes[0, 0].set_title(f"original ({meta.pid}, slice {k})", fontsize=8)
    for i, ax in enumerate(axes.flat[1:], start=1):
        out = transforms(dict(sample))
        show_slice(ax, np.asarray(out["image"])[0], np.asarray(out["label"])[0].astype(np.uint8))
        ax.set_title(f"random draw {i}", fontsize=8)
    fig.legend(handles=structure_legend(), loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Training augmentation: same slice, different random transforms", fontsize=10)
    save(fig, Path("figures/augmentation_examples.png"))
    print("saved figures/augmentation_examples.png")


if __name__ == "__main__":
    main()
