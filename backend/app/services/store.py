"""Cases the app can show: precomputed samples (from disk) and uploads (in memory, for an hour).

A case holds uint8 arrays only, ready to slice into PNGs: the windowed images, the label maps of
each layer, the doubt maps and, for samples, the windowed cine. Nothing uploaded is written to disk.
"""

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Case:
    case_id: str
    summary: dict[str, object]
    images: dict[str, np.ndarray]  # phase -> (X, Y, Z) uint8
    layers: dict[str, dict[str, np.ndarray]]  # phase -> layer -> (X, Y, Z) uint8
    cine: np.ndarray | None = None  # (X, Y, Z, T) uint8, samples only
    created: float = field(default_factory=time.time)


def save_sample(case: Case, folder: Path) -> None:
    """Writes <folder>/<case_id>.json (summary) and .npz (arrays). Used by cache_sample_cases.py."""
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {f"image_{p}": a for p, a in case.images.items()}
    for phase, layers in case.layers.items():
        arrays.update({f"layer_{phase}_{name}": a for name, a in layers.items()})
    if case.cine is not None:
        arrays["cine"] = case.cine
    np.savez_compressed(folder / f"{case.case_id}.npz", allow_pickle=False, **arrays)
    (folder / f"{case.case_id}.json").write_text(json.dumps(case.summary, indent=2) + "\n")


def load_sample(folder: Path, case_id: str) -> Case:
    summary = json.loads((folder / f"{case_id}.json").read_text())
    images: dict[str, np.ndarray] = {}
    layers: dict[str, dict[str, np.ndarray]] = {}
    cine = None
    with np.load(folder / f"{case_id}.npz") as data:
        for key in data.files:
            if key.startswith("image_"):
                images[key.removeprefix("image_")] = data[key]
            elif key.startswith("layer_"):
                _, phase, name = key.split("_", 2)
                layers.setdefault(phase, {})[name] = data[key]
            elif key == "cine":
                cine = data[key]
    return Case(case_id=case_id, summary=summary, images=images, layers=layers, cine=cine)


class CaseStore:
    """Thread-safe: FastAPI runs sync endpoints in a thread pool."""

    def __init__(self, samples_dir: Path, ttl_s: int) -> None:
        self.samples_dir = samples_dir
        self.ttl_s = ttl_s
        self._uploads: dict[str, Case] = {}
        self._samples: dict[str, Case] = {}
        self._lock = threading.Lock()

    def sample_ids(self) -> list[str]:
        if not self.samples_dir.exists():
            return []
        return sorted(p.stem for p in self.samples_dir.glob("*.json"))

    def add_upload(
        self,
        summary: dict[str, object],
        images: dict[str, np.ndarray],
        layers: dict[str, dict[str, np.ndarray]],
    ) -> Case:
        case = Case(case_id=uuid.uuid4().hex[:12], summary=summary, images=images, layers=layers)
        case.summary["id"] = case.case_id
        with self._lock:
            self._expire()
            self._uploads[case.case_id] = case
        return case

    def get(self, case_id: str) -> Case | None:
        with self._lock:
            self._expire()
            if case_id in self._uploads:
                return self._uploads[case_id]
            if case_id in self._samples:
                return self._samples[case_id]
        if case_id in self.sample_ids():
            case = load_sample(self.samples_dir, case_id)
            with self._lock:
                self._samples[case_id] = case
            return case
        return None

    def _expire(self) -> None:
        now = time.time()
        for case_id in [k for k, c in self._uploads.items() if now - c.created > self.ttl_s]:
            del self._uploads[case_id]
