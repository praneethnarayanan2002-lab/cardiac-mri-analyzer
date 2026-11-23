"""Train a segmenter. Usage:

    uv run python -m cardiac.train --config configs/unet_scratch.yaml [--seed 1] [--max-epochs 2]

Writes runs/<name>/<timestamp>-seed<k>/ with config.yaml, log.csv, best.pt and git_commit.txt.
"""

import argparse
import csv
import math
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from monai.data import DataLoader
from monai.losses import DiceCELoss
from torch import nn

from cardiac.config import Config, load_config
from cardiac.data.augment import training_transforms
from cardiac.data.dataset import PreparedVolume, SliceDataset, load_volumes
from cardiac.data.io import load_metadata
from cardiac.data.splits import load_splits
from cardiac.infer import build_model, pick_device, predict_probabilities
from cardiac.metrics import STRUCTURES, dice
from cardiac.models.unet import count_parameters
from cardiac.repro import git_commit, set_seed


def warmup_cosine(warmup_epochs: int, max_epochs: int) -> Callable[[int], float]:
    """Learning-rate multiplier per epoch: linear ramp from ~0 to 1, then a cosine curve to 0."""

    def factor(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return factor


def validate(
    model: nn.Module, volumes: list[PreparedVolume], device: torch.device
) -> dict[str, float]:
    """Mean 3D Dice per structure over all validation volumes, ED and ES, before post-processing."""
    per_structure: dict[str, list[float]] = {name: [] for name in STRUCTURES.values()}
    for vol in volumes:
        truth = vol.prepared.label
        assert truth is not None
        pred = predict_probabilities(model, vol.prepared.image, device).argmax(axis=1)
        for value, name in STRUCTURES.items():
            per_structure[name].append(dice(pred == value, truth == value))
    scores = {f"val_dice_{name}": float(np.mean(v)) for name, v in per_structure.items()}
    scores["val_dice_mean"] = float(np.mean(list(scores.values())))
    return scores


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        images, labels = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def make_run_dir(config: Config) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / config.name / f"{stamp}-seed{config.train.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    (run_dir / "git_commit.txt").write_text(git_commit() + "\n")
    return run_dir


def load_split_volumes(config: Config) -> tuple[list[PreparedVolume], list[PreparedVolume]]:
    root = Path(config.data.root)
    splits = load_splits(Path(config.data.splits))
    metas = {m.pid: m for m in load_metadata(root, "train")}
    spacing, size = config.data.target_spacing_mm, config.data.size
    train_vols = load_volumes(root, [metas[i] for i in splits["train"]], spacing, size)
    val_vols = load_volumes(root, [metas[i] for i in splits["val"]], spacing, size)
    return train_vols, val_vols


def train(config: Config) -> Path:
    set_seed(config.train.seed)
    device = pick_device()
    run_dir = make_run_dir(config)
    train_vols, val_vols = load_split_volumes(config)
    dataset = SliceDataset(train_vols, training_transforms() if config.train.augment else None)
    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        persistent_workers=config.train.num_workers > 0,
        drop_last=True,
    )
    model = build_model(config).to(device)
    print(
        f"{config.name} seed {config.train.seed} on {device}: {count_parameters(model):,} params, "
        f"{len(dataset)} training slices, {len(val_vols)} validation volumes -> {run_dir}"
    )

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(config.train.warmup_epochs, config.train.max_epochs)
    )
    fit(model, loader, val_vols, loss_fn, optimizer, schedule, device, config, run_dir)
    return run_dir


def fit(
    model: nn.Module,
    loader: DataLoader,
    val_vols: list[PreparedVolume],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    schedule: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    config: Config,
    run_dir: Path,
) -> None:
    best, since_best = -1.0, 0
    with (run_dir / "log.csv").open("w", newline="") as f:
        writer = None
        for epoch in range(config.train.max_epochs):
            start = time.time()
            lr = optimizer.param_groups[0]["lr"]
            train_loss = train_one_epoch(model, loader, loss_fn, optimizer, device)
            schedule.step()
            row = {"epoch": epoch, "lr": lr, "train_loss": train_loss}
            row |= validate(model, val_vols, device)
            row["seconds"] = round(time.time() - start, 1)
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            f.flush()

            improved = row["val_dice_mean"] > best
            if improved:
                best, since_best = row["val_dice_mean"], 0
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": config.to_dict(),
                        "epoch": epoch,
                        "val_dice_mean": best,
                    },
                    run_dir / "best.pt",
                )
            else:
                since_best += 1
            print(
                f"epoch {epoch:3d}  loss {train_loss:.4f}  val dice {row['val_dice_mean']:.4f}"
                f"{'  *' if improved else ''}  ({row['seconds']}s)",
                flush=True,
            )
            if since_best >= config.train.patience:
                print(f"early stop: no improvement for {config.train.patience} epochs")
                break
    print(f"best val mean Dice {best:.4f} -> {run_dir / 'best.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a cardiac segmenter.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-epochs", type=int, help="override, e.g. for a quick smoke test")
    parser.add_argument("--name", help="override the run name")
    args = parser.parse_args()
    config = load_config(args.config)
    pairs = (("seed", args.seed), ("max_epochs", args.max_epochs))
    overrides = {key: value for key, value in pairs if value is not None}
    config = replace(config, train=replace(config.train, **overrides))
    if args.name:
        config = replace(config, name=args.name)
    train(config)


if __name__ == "__main__":
    main()
