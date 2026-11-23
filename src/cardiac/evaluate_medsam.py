"""Evaluate MedSAM with oracle boxes at several jitter levels. Usage:

    uv run python -m cardiac.evaluate_medsam --split val
    uv run python -m cardiac.evaluate_medsam --split test --confirm-test

Writes results/<split>/medsam_jitter<m>.json and .csv for m in 0, 5, 10, 20 px. Same metrics, same
post-processing and same summary as the U-Nets, so the numbers are directly comparable.
"""

import argparse
import zlib

import numpy as np
import pandas as pd

from cardiac.data.io import DEFAULT_DATA_ROOT, load_frame, load_metadata
from cardiac.data.splits import load_splits
from cardiac.evaluate import SPLIT_SOURCE, save_results
from cardiac.infer import pick_device, postprocess
from cardiac.metrics import evaluate_volume
from cardiac.models.medsam import BOX_PADDING_PX, REPO, REVISION, load_medsam, segment_volume
from cardiac.repro import git_commit

JITTERS_PX = (0, 5, 10, 20)
SEED = 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MedSAM with oracle box prompts.")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--confirm-test", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.confirm_test:
        parser.error("the test set is used once per model, at the end; add --confirm-test")

    eval_commit = git_commit()
    device = pick_device()
    model, processor = load_medsam(device)
    ids = set(load_splits()[args.split])
    metas = [m for m in load_metadata(DEFAULT_DATA_ROOT, SPLIT_SOURCE[args.split]) if m.pid in ids]  # type: ignore[arg-type]
    rows: dict[int, list[dict[str, object]]] = {m: [] for m in JITTERS_PX}
    for i, meta in enumerate(metas):
        for phase in ("ed", "es"):
            frame = load_frame(DEFAULT_DATA_ROOT, meta, phase)
            # A fixed seed per patient and phase: the same boxes every time this is run.
            rng = np.random.default_rng([SEED, zlib.crc32(f"{meta.pid}-{phase}".encode())])
            volumes = segment_volume(
                model, processor, frame.image, frame.label, device, JITTERS_PX, rng
            )
            for magnitude, raw in volumes.items():
                for post, pred in (("raw", raw), ("post", postprocess(raw))):
                    scores = evaluate_volume(pred, frame.label, frame.spacing_mm)
                    rows[magnitude].append(
                        {
                            "pid": meta.pid,
                            "group": meta.group,
                            "phase": phase,
                            "postprocess": post,
                            **scores,
                        }
                    )
        print(f"  {i + 1}/{len(metas)} {meta.pid}", flush=True)

    for magnitude in JITTERS_PX:
        info = {
            "model": f"{REPO}@{REVISION[:7]}",
            "prompt": f"oracle boxes from the expert label, padded {BOX_PADDING_PX} px, "
            f"each edge jittered by up to {magnitude} px (seed {SEED})",
            "eval_git_commit": eval_commit,
        }
        save_results(pd.DataFrame(rows[magnitude]), args.split, f"medsam_jitter{magnitude}", info)


if __name__ == "__main__":
    np.seterr(invalid="ignore")
    main()
