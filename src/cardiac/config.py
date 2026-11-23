"""YAML configs loaded into typed dataclasses, so a typo in a key fails loudly."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: str = "data/raw/acdc"
    splits: str = "data/splits.json"
    target_spacing_mm: float = 1.0
    size: int = 192


@dataclass
class TrainConfig:
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    max_epochs: int = 150
    patience: int = 20
    num_workers: int = 4
    augment: bool = True
    seed: int = 0


@dataclass
class Config:
    name: str
    model: str
    dropout: float = 0.1
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls: type, values: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text())
    data = _build(DataConfig, raw.pop("data", {}))
    train = _build(TrainConfig, raw.pop("train", {}))
    config: Config = _build(Config, {**raw, "data": data, "train": train})
    return config
