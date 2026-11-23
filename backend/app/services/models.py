"""Everything the backend loads once at startup: two U-Nets, the diagnosis model and its
explainer, the normal ranges and the doubt threshold, all from files the research pipeline wrote."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
import torch
from torch import nn
from xgboost import XGBClassifier

from backend.app.config import Settings
from cardiac.config import Config
from cardiac.infer import load_checkpoint
from cardiac.repro import git_commit

# The fraction of validation patients the app flags for review: the 20% with the highest doubt.
FLAG_FRACTION = 0.2


class ModelsMissingError(RuntimeError):
    """Raised at startup when trained weights or results are not where the settings say."""


@dataclass
class UNet:
    model: nn.Module
    config: Config
    run: str


@dataclass
class ModelBundle:
    unet: UNet  # scratch U-Net: second opinion layer in the viewer
    unet_pretrained: (
        UNet  # ResNet-34 U-Net with TTA: measurements, doubt and diagnosis come from it
    )
    classifier: XGBClassifier
    explainer: Any  # shap.TreeExplainer
    normal_ranges: dict[str, dict[str, float]]
    doubt_threshold: float
    device: torch.device
    code_version: str


def find_run(runs_dir: Path, name: str, seed: int) -> Path:
    matches = sorted((runs_dir / name).glob(f"*-seed{seed}"))
    if not matches:
        raise ModelsMissingError(
            f"no trained run {runs_dir / name}/*-seed{seed}. Train with `make train` or download "
            "the weights (see README)."
        )
    return matches[-1]


def load_unet(runs_dir: Path, name: str, seed: int, device: torch.device) -> UNet:
    run = find_run(runs_dir, name, seed)
    model, config = load_checkpoint(run / "best.pt", device)
    model.eval()
    return UNet(model=model, config=config, run=run.name)


def doubt_threshold(results_dir: Path, seed: int) -> float:
    """80th percentile of per-patient doubt scores on the validation patients, never test."""
    path = results_dir / "val" / f"uncertainty_unet_resnet34_seed{seed}.csv"
    if not path.exists():
        raise ModelsMissingError(f"{path} not found; run cardiac.evaluate_uncertainty on val")
    per_patient = pd.read_csv(path).groupby("pid").doubt_score.mean()
    return float(np.quantile(per_patient, 1.0 - FLAG_FRACTION))


def load_models(settings: Settings) -> ModelBundle:
    device = torch.device(settings.device)
    # OMP_NUM_THREADS=1 was set before torch loaded (backend/app/__init__.py); give torch its own
    # threads back for CPU inference.
    torch.set_num_threads(os.cpu_count() or 1)
    classifier_path = settings.runs_dir / "classifier" / "xgboost.json"
    ranges_path = settings.results_dir / "normal_ranges.json"
    for path in (classifier_path, ranges_path):
        if not path.exists():
            raise ModelsMissingError(f"{path} not found; run `make eval` (see README)")
    classifier = XGBClassifier()
    classifier.load_model(classifier_path)
    return ModelBundle(
        unet=load_unet(settings.runs_dir, "unet_scratch", settings.seed, device),
        unet_pretrained=load_unet(settings.runs_dir, "unet_resnet34", settings.seed, device),
        classifier=classifier,
        explainer=shap.TreeExplainer(classifier),
        normal_ranges=json.loads(ranges_path.read_text())["ranges"],
        doubt_threshold=doubt_threshold(settings.results_dir, settings.seed),
        device=device,
        code_version=git_commit(),
    )
