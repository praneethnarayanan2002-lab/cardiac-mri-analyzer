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
    """Current commit hash, flagged if code or config has uncommitted changes.

    results/ and figures/ are outputs, not code, so new or rewritten files there don't count."""

    def run(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()

    commit = run("rev-parse", "HEAD") or "unknown"
    changes = run("status", "--porcelain", "--", ".", ":(exclude)results", ":(exclude)figures")
    return commit + (" (uncommitted changes)" if changes else "")
