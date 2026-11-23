"""Write data/splits.json: 80 train / 20 val (stratified by group) + the official 50 test patients.

Usage:
    uv run python scripts/make_splits.py
"""

import argparse
from collections import Counter
from pathlib import Path

from cardiac.data.io import ACDC_GROUPS, DEFAULT_DATA_ROOT, load_metadata
from cardiac.data.splits import DEFAULT_SPLITS_PATH, make_splits, save_splits

SEED = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_SPLITS_PATH)
    args = parser.parse_args()

    train_pool = load_metadata(args.root, "train")
    test = load_metadata(args.root, "test")
    splits = make_splits(train_pool, test, n_val=20, seed=SEED)
    provenance = {
        "seed": SEED,
        "rule": "val = 20 of the 100 official training patients, stratified by group; "
        "test = the 50 official test patients",
    }
    save_splits(splits, args.out, provenance)

    group_of = {m.pid: m.group for m in train_pool + test}
    print(f"Wrote {args.out}")
    for name, pids in splits.items():
        counts = Counter(group_of[p] for p in pids)
        per_group = ", ".join(f"{g} {counts[g]}" for g in ACDC_GROUPS)
        print(f"  {name:<5} {len(pids):>3} patients  ({per_group})")


if __name__ == "__main__":
    main()
