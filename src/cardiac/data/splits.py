"""Patient-level train / validation / test splits.

Slices from one patient are near-duplicates of each other, so splitting by slice would put the same
heart in training and validation and make validation scores meaningless. Every split here is a list
of patient ids, never slices.
"""

import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from cardiac.data.io import PatientMeta

DEFAULT_SPLITS_PATH = Path("data/splits.json")


def make_splits(
    train_pool: list[PatientMeta], test: list[PatientMeta], n_val: int = 20, seed: int = 0
) -> dict[str, list[str]]:
    """Carves a validation set out of the official training patients, stratified by group.

    The official 50 test patients stay untouched as the test split.
    """
    pids = [m.pid for m in train_pool]
    groups = [m.group for m in train_pool]
    train_ids, val_ids = train_test_split(
        pids, test_size=n_val, stratify=groups, random_state=seed, shuffle=True
    )
    splits = {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(m.pid for m in test),
    }
    assert_disjoint(splits)
    return splits


def assert_disjoint(splits: dict[str, list[str]]) -> None:
    names = list(splits)
    for i, a in enumerate(names):
        if len(set(splits[a])) != len(splits[a]):
            raise ValueError(f"split {a!r} lists a patient twice")
        for b in names[i + 1 :]:
            shared = set(splits[a]) & set(splits[b])
            if shared:
                raise ValueError(f"patients in both {a!r} and {b!r}: {sorted(shared)}")


def save_splits(splits: dict[str, list[str]], path: Path, provenance: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"provenance": provenance, **splits}
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_splits(path: Path = DEFAULT_SPLITS_PATH) -> dict[str, list[str]]:
    payload = json.loads(path.read_text())
    splits = {name: payload[name] for name in ("train", "val", "test")}
    assert_disjoint(splits)
    return splits
