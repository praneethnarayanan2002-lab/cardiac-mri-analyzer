"""Where the backend finds its models and data. Every path can be overridden with an environment
variable, so the same code runs from a fresh clone, in tests, and on another machine."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    runs_dir: Path = field(default_factory=lambda: _path("CARDIAC_RUNS_DIR", "runs"))
    results_dir: Path = field(default_factory=lambda: _path("CARDIAC_RESULTS_DIR", "results"))
    samples_dir: Path = field(
        default_factory=lambda: _path("CARDIAC_SAMPLES_DIR", "data/cache/samples")
    )
    # CPU by default so the app runs on any machine; "mps" or "cuda" if available and wanted.
    device: str = field(default_factory=lambda: os.environ.get("CARDIAC_DEVICE", "cpu"))
    case_ttl_s: int = 3600  # uploaded cases are forgotten after an hour
    max_upload_mb: int = 200
    # Which trained run serves each layer: seed 0 of each U-Net.
    seed: int = 0
    allowed_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")


def get_settings() -> Settings:
    return Settings()
