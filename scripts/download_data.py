"""Download the ACDC mirror from the Hugging Face Hub, check it, and print a summary.

Usage:
    uv run python scripts/download_data.py            # download (if needed) + verify
    uv run python scripts/download_data.py --no-cine  # skip the 4D cine files (~1 GB)
    uv run python scripts/download_data.py --verify-only
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

from cardiac.data.io import (
    ACDC_GROUPS,
    DEFAULT_DATA_ROOT,
    PatientMeta,
    load_metadata,
    load_nifti,
    patient_files,
)

REPO_ID = "mathpluscode/ACDC"
# Pinning the dataset commit means a re-download in a year gets byte-identical files.
REVISION = "4a312b610f395eda5dc0aa36f087dd6ca9991e41"


def download(root: Path, include_cine: bool) -> None:
    patterns = ["*.csv", "*.md", "*_sax_ed.nii.gz", "*_sax_es.nii.gz", "*_gt.nii.gz"]
    if include_cine:
        patterns.append("*_sax_t.nii.gz")
    snapshot_download(
        REPO_ID, repo_type="dataset", revision=REVISION, local_dir=root, allow_patterns=patterns
    )


def verify(root: Path, include_cine: bool) -> list[str]:
    """Returns a list of problems; an empty list means the folder is complete."""
    problems = []
    for split, expected in (("train", 100), ("test", 50)):
        metas = load_metadata(root, split)
        if len(metas) != expected:
            problems.append(f"{split}.csv has {len(metas)} patients, expected {expected}")
        for meta in metas:
            for kind, path in patient_files(root, meta).items():
                if kind == "cine" and not include_cine:
                    continue
                if not path.exists():
                    problems.append(f"missing {path.relative_to(root)}")
    return problems


def summarise(root: Path) -> None:
    for split in ("train", "test"):
        metas = load_metadata(root, split)
        print(f"\n{split}: {len(metas)} patients")
        counts = Counter(m.group for m in metas)
        print("  per group: " + ", ".join(f"{g} {counts[g]}" for g in ACDC_GROUPS))
        _print_spacing_table(root, metas)


def _print_spacing_table(root: Path, metas: list[PatientMeta]) -> None:
    shapes, spacings = [], []
    for meta in metas:
        image, spacing = load_nifti(patient_files(root, meta)["ed_image"])
        shapes.append(image.shape)
        spacings.append(spacing)
    shape_arr, spacing_arr = np.array(shapes), np.array(spacings)
    original = np.array([m.original_spacing_mm for m in metas])
    rows = [
        ("mirror shape (X, Y, Z)", shape_arr),
        ("mirror spacing mm", spacing_arr),
        ("original spacing mm", original),
    ]
    for label, arr in rows:
        lo, hi = arr.min(axis=0), arr.max(axis=0)
        span = "  ".join(f"{a:g}-{b:g}" for a, b in zip(lo, hi, strict=True))
        print(f"  {label:<24} {span}")
    thick = sum(m.original_spacing_mm[2] < 10 for m in metas)
    print(f"  patients whose original slices were thinner than 10 mm: {thick}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--no-cine", action="store_true", help="skip the 4D cine files")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    include_cine = not args.no_cine

    if not args.verify_only:
        print(f"Downloading {REPO_ID}@{REVISION[:7]} to {args.root} ...")
        download(args.root, include_cine)

    problems = verify(args.root, include_cine)
    if problems:
        print(f"Found {len(problems)} problem(s):")
        for problem in problems[:20]:
            print("  -", problem)
        return 1
    print("All expected files are present.")
    summarise(args.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
