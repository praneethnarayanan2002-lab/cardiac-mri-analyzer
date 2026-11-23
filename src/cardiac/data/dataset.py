"""PyTorch datasets built from patient ids.

Training works on 2D slices; validation and evaluation work on whole 3D volumes, because Dice is
reported per patient in 3D. Both start from the same `prepare` step, so they see identical inputs.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from cardiac.data.io import PatientMeta, Phase, load_frame
from cardiac.data.preprocess import Prepared, prepare

PHASES: tuple[Phase, Phase] = ("ed", "es")


@dataclass(frozen=True)
class PreparedVolume:
    pid: str
    group: str
    phase: Phase
    prepared: Prepared
    spacing_mm: tuple[float, float, float]


def load_volumes(
    root: Path, metas: list[PatientMeta], target_mm: float = 1.0, size: int = 192
) -> list[PreparedVolume]:
    volumes = []
    for meta in metas:
        for phase in PHASES:
            frame = load_frame(root, meta, phase)
            prepared = prepare(frame.image, frame.spacing_mm, frame.label, target_mm, size)
            volumes.append(PreparedVolume(meta.pid, meta.group, phase, prepared, frame.spacing_mm))
    return volumes


class SliceDataset(Dataset[dict[str, torch.Tensor]]):
    """Every slice of every given volume, held in memory (~250 MB for the 80 training patients)."""

    def __init__(
        self,
        volumes: list[PreparedVolume],
        transform: Callable[[dict[str, np.ndarray]], dict[str, object]] | None = None,
    ) -> None:
        images, labels = [], []
        for vol in volumes:
            assert vol.prepared.label is not None
            images.append(vol.prepared.image)
            labels.append(vol.prepared.label)
        # Add the channel axis: (N, 1, H, W) is what both the transforms and the U-Net expect.
        self.images = np.concatenate(images)[:, None]
        self.labels = np.concatenate(labels)[:, None]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample: dict[str, object] = {
            "image": self.images[index],
            "label": self.labels[index].astype(np.float32),
        }
        if self.transform is not None:
            sample = self.transform(sample)  # type: ignore[arg-type]
        image = torch.as_tensor(np.asarray(sample["image"]), dtype=torch.float32)
        label = torch.as_tensor(np.asarray(sample["label"]), dtype=torch.float32).round().long()
        return {"image": image, "label": label}
