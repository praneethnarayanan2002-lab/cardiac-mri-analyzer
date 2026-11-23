"""Reproducibility helpers: seeding and recording the code version."""

import random
import subprocess

import numpy as np
import torch
from monai.utils import set_determinism


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_determinism(seed=seed)


def git_commit() -> str:
    """Current commit hash, flagged if the working tree has uncommitted changes."""

    def run(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()

    commit = run("rev-parse", "HEAD") or "unknown"
    return commit + (" (uncommitted changes)" if run("status", "--porcelain") else "")
