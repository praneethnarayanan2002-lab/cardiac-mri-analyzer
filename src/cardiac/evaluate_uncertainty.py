"""TTA predictions, doubt scores and their Dice, per patient and phase. Usage:

    uv run python -m cardiac.evaluate_uncertainty --run runs/unet_resnet34/<folder> --split val

Writes results/<split>/uncertainty_<run name>_seed<k>.csv: one row per patient and phase with the
doubt score and the Dice / HD95 of the TTA-averaged, post-processed prediction. The test split needs
--confirm-test.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cardiac.data.io import load_frame
from cardiac.evaluate import SPLIT_SOURCE
from cardiac.infer import load_checkpoint, pick_device
from cardiac.measure import split_patients
from cardiac.metrics import evaluate_volume
from cardiac.repro import git_commit
from cardiac.uncertainty import doubt_score, predict_with_doubt


def main() -> None:
    parser = argparse.ArgumentParser(description="TTA uncertainty per patient.")
    parser.add_argument("--run", type=Path, required=True, help="run folder containing best.pt")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--confirm-test", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.confirm_test:
        parser.error("the test set is used once per model, at the end; add --confirm-test")

    eval_commit = git_commit()
    device = pick_device()
    model, config = load_checkpoint(args.run / "best.pt", device)
    root = Path(config.data.root)
    rows = []
    metas = split_patients(args.split)
    assert all(m.split == SPLIT_SOURCE[args.split] for m in metas)
    for i, meta in enumerate(metas):
        for phase in ("ed", "es"):
            frame = load_frame(root, meta, phase)
            label, doubt = predict_with_doubt(
                model,
                frame.image,
                frame.spacing_mm,
                device,
                config.data.target_spacing_mm,
                config.data.size,
            )
            rows.append(
                {
                    "pid": meta.pid,
                    "group": meta.group,
                    "phase": phase,
                    "doubt_score": doubt_score(doubt, label),
                    **evaluate_volume(label, frame.label, frame.spacing_mm),
                }
            )
        print(f"  {i + 1}/{len(metas)} {meta.pid}", flush=True)

    stem = f"uncertainty_{config.name}_seed{config.train.seed}"
    out_dir = Path("results") / args.split
    pd.DataFrame(rows).to_csv(out_dir / f"{stem}.csv", index=False, float_format="%.5f")
    info = {
        "run": str(args.run),
        "train_git_commit": (args.run / "git_commit.txt").read_text().strip(),
        "eval_git_commit": eval_commit,
        "split": args.split,
        "tta": "8 views: rotations by 0/90/180/270 degrees, each with and without a flip",
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"wrote {out_dir / stem}.csv and .json")


if __name__ == "__main__":
    np.seterr(invalid="ignore")
    main()
