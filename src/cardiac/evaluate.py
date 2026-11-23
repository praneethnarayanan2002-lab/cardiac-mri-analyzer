"""Evaluate a trained checkpoint per patient, in 3D, on the original image grid. Usage:

    uv run python -m cardiac.evaluate --run runs/unet_scratch/<timestamp>-seed0 --split val

Writes results/<split>/<run name>_seed<k>.json (summary) and .csv (one row per patient and phase).
The test split needs --confirm-test, because it is evaluated once per model, at the end.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cardiac.data.io import ACDC_GROUPS, load_frame, load_metadata
from cardiac.data.splits import load_splits
from cardiac.infer import load_checkpoint, pick_device, postprocess, predict_volume
from cardiac.metrics import STRUCTURES, evaluate_volume
from cardiac.repro import git_commit

SPLIT_SOURCE = {"train": "train", "val": "train", "test": "test"}  # which official folder


def evaluate_run(checkpoint: Path, split: str, device: torch.device) -> pd.DataFrame:
    model, config = load_checkpoint(checkpoint, device)
    root = Path(config.data.root)
    ids = set(load_splits(Path(config.data.splits))[split])
    metas = [m for m in load_metadata(root, SPLIT_SOURCE[split]) if m.pid in ids]  # type: ignore[arg-type]
    rows = []
    for meta in metas:
        for phase in ("ed", "es"):
            frame = load_frame(root, meta, phase)
            raw = predict_volume(
                model,
                frame.image,
                frame.spacing_mm,
                device,
                config.data.target_spacing_mm,
                config.data.size,
                apply_postprocess=False,
            )
            for post, pred in (("raw", raw), ("post", postprocess(raw))):
                scores = evaluate_volume(pred, frame.label, frame.spacing_mm)
                rows.append(
                    {
                        "pid": meta.pid,
                        "group": meta.group,
                        "phase": phase,
                        "postprocess": post,
                        **scores,
                    }
                )
    return pd.DataFrame(rows)


def _mean_std(values: pd.Series) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "n_missing": int(values.isna().sum()),
    }


def summarise(table: pd.DataFrame) -> dict[str, object]:
    """Headline = post-processed; raw is kept to measure what post-processing changes."""
    metrics = [f"{m}_{s}" for m in ("dice", "hd95") for s in STRUCTURES.values()]
    summary: dict[str, object] = {}
    for post in ("post", "raw"):
        part = table[table.postprocess == post]
        block: dict[str, object] = {
            "overall": {m: _mean_std(part[m]) for m in metrics},
            "by_phase": {
                p: {m: _mean_std(g[m]) for m in metrics} for p, g in part.groupby("phase")
            },
            "by_group": {
                grp: {m: float(part.loc[part.group == grp, m].mean()) for m in metrics}
                for grp in ACDC_GROUPS
            },
        }
        dice_cols = [f"dice_{s}" for s in STRUCTURES.values()]
        block["mean_dice"] = float(part[dice_cols].to_numpy().mean())
        summary[post] = block
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint per patient in 3D.")
    parser.add_argument("--run", type=Path, required=True, help="run folder containing best.pt")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--confirm-test", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.confirm_test:
        parser.error("the test set is used once per model, at the end; add --confirm-test")

    device = pick_device()
    table = evaluate_run(args.run / "best.pt", args.split, device)
    _, config = load_checkpoint(args.run / "best.pt", device)
    out_dir = Path("results") / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{config.name}_seed{config.train.seed}"
    table.to_csv(out_dir / f"{stem}.csv", index=False, float_format="%.4f")
    checkpoint = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
    payload = {
        "run": str(args.run),
        "train_git_commit": (args.run / "git_commit.txt").read_text().strip(),
        "eval_git_commit": git_commit(),
        "split": args.split,
        "n_patients": int(table.pid.nunique()),
        "best_epoch": int(checkpoint["epoch"]),
        **summarise(table),
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2) + "\n")
    post = payload["post"]
    assert isinstance(post, dict)
    print(f"{stem} on {args.split} ({payload['n_patients']} patients), post-processed:")
    for name in STRUCTURES.values():
        d, h = post["overall"][f"dice_{name}"], post["overall"][f"hd95_{name}"]
        print(
            f"  {name:<4} Dice {d['mean']:.3f} ± {d['std']:.3f}   "
            f"HD95 {h['mean']:.2f} ± {h['std']:.2f} mm"
        )
    print(f"  mean Dice {post['mean_dice']:.4f} (raw {payload['raw']['mean_dice']:.4f})")
    print(f"wrote {out_dir / stem}.json and .csv")


if __name__ == "__main__":
    np.seterr(invalid="ignore")
    main()
