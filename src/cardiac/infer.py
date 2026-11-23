"""Running a trained segmenter on whole 3D frames, plus post-processing."""

from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch import nn

from cardiac.config import Config, DataConfig, TrainConfig
from cardiac.data.preprocess import prepare, restore
from cardiac.models.unet import build_scratch_unet
from cardiac.models.unet_pretrained import build_pretrained_unet

CAVITIES = (1, 3)  # RV and LV; the myocardium is a ring and must keep its hole


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(config: Config, imagenet_weights: bool = True) -> nn.Module:
    """`imagenet_weights=False` skips the ImageNet download when a checkpoint replaces them."""
    if config.model == "unet_scratch":
        return build_scratch_unet(config.dropout)
    if config.model == "unet_resnet34":
        return build_pretrained_unet("imagenet" if imagenet_weights else None)
    raise ValueError(f"unknown model {config.model!r}")


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, Config]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    raw = dict(checkpoint["config"])
    config = Config(
        **{k: v for k, v in raw.items() if k not in ("data", "train")},
        data=DataConfig(**raw["data"]),
        train=TrainConfig(**raw["train"]),
    )
    model = build_model(config, imagenet_weights=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config


@torch.no_grad()
def predict_probabilities(
    model: nn.Module, slices: np.ndarray, device: torch.device, batch_size: int = 32
) -> np.ndarray:
    """(S, H, W) normalised slices -> (S, 4, H, W) softmax probabilities."""
    model.eval()
    outputs = []
    for start in range(0, len(slices), batch_size):
        batch = torch.from_numpy(slices[start : start + batch_size, None]).to(device)
        outputs.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def remove_small_components(label: np.ndarray, min_fraction: float = 0.1) -> np.ndarray:
    """Per structure, drops 3D blobs smaller than `min_fraction` of that structure's largest blob.

    Keeping only the single largest blob was tried first and deleted real RV: with 10 mm between
    slices the thin RV crescent can come out as two genuine pieces (docs/decisions.md D15)."""
    out = label.copy()
    full_connectivity = np.ones((3, 3, 3), dtype=bool)
    for value in np.unique(label):
        if value == 0:
            continue
        components, n = ndimage.label(label == value, structure=full_connectivity)
        if n <= 1:
            continue
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        too_small = np.flatnonzero(sizes < min_fraction * sizes.max())
        out[np.isin(components, too_small[too_small > 0])] = 0
    return out


def fill_cavity_holes(label: np.ndarray) -> np.ndarray:
    """Fills background holes inside the LV and RV cavities, slice by slice."""
    out = label.copy()
    for k in range(label.shape[2]):
        plane = out[:, :, k]
        for value in CAVITIES:
            filled = ndimage.binary_fill_holes(plane == value)
            plane[filled & (plane == 0)] = value
    return out


def postprocess(label: np.ndarray) -> np.ndarray:
    return fill_cavity_holes(remove_small_components(label))


def predict_volume(
    model: nn.Module,
    image: np.ndarray,
    spacing_mm: tuple[float, float, float],
    device: torch.device,
    target_mm: float = 1.0,
    size: int = 192,
    apply_postprocess: bool = True,
    centre_xy: tuple[float, float] | None = None,
) -> np.ndarray:
    """(X, Y, Z) image on its own grid -> (X, Y, Z) uint8 label map on the same grid.

    `centre_xy` moves the crop window (see `prepare`); None is the centre crop used everywhere in
    training and evaluation."""
    prepared = prepare(image, spacing_mm, None, target_mm, size, centre_xy)
    probs = predict_probabilities(model, prepared.image, device)
    label = restore(probs.argmax(axis=1).astype(np.uint8), prepared.geometry)
    return postprocess(label) if apply_postprocess else label
