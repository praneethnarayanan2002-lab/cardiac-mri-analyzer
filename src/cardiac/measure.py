"""Clinical measurements per patient, from expert masks or from a model's predicted masks. Usage:

    uv run python -m cardiac.measure --expert --split train
    uv run python -m cardiac.measure --run runs/unet_scratch/<timestamp>-seed0 --split val
    uv run python -m cardiac.measure --medsam --split val

Writes results/<split>/clinical_<source>.csv (one row per patient, columns from
cardiac.clinical.MEASUREMENTS and INDEXED) and a .json with where the numbers came from. The test
split needs --confirm-test.

Predicted masks are recomputed with the same frozen model and the same deterministic inference and
post-processing as cardiac.evaluate. To prove they are the same masks that were scored, each run's
Dice is recomputed and compared with results/<split>/<source>.csv.
"""

import argparse
import json
import zlib
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cardiac.clinical import measure_patient
from cardiac.data.io import DEFAULT_DATA_ROOT, PatientMeta, load_frame, load_metadata
from cardiac.data.splits import load_splits
from cardiac.evaluate import SPLIT_SOURCE
from cardiac.evaluate_medsam import JITTERS_PX, SEED
from cardiac.infer import load_checkpoint, pick_device, postprocess, predict_volume
from cardiac.metrics import STRUCTURES, dice
from cardiac.repro import git_commit

# For one patient: {source stem: (ED label, ES label)}, all on the patient's own grid.
Predictor = Callable[[PatientMeta], dict[str, tuple[np.ndarray, np.ndarray]]]


def split_patients(split: str) -> list[PatientMeta]:
    ids = set(load_splits()[split])
    return [m for m in load_metadata(DEFAULT_DATA_ROOT, SPLIT_SOURCE[split]) if m.pid in ids]  # type: ignore[arg-type]


def expert_predictor() -> tuple[Predictor, dict[str, object]]:
    def predict(meta: PatientMeta) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        ed = load_frame(DEFAULT_DATA_ROOT, meta, "ed").label
        es = load_frame(DEFAULT_DATA_ROOT, meta, "es").label
        return {"expert": (ed, es)}

    return predict, {"source": "expert labels from the ACDC mirror"}


def run_predictor(run: Path, device: torch.device) -> tuple[Predictor, dict[str, object]]:
    model, config = load_checkpoint(run / "best.pt", device)
    stem = f"{config.name}_seed{config.train.seed}"

    def predict(meta: PatientMeta) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        labels = []
        for phase in ("ed", "es"):
            frame = load_frame(DEFAULT_DATA_ROOT, meta, phase)
            labels.append(
                predict_volume(
                    model,
                    frame.image,
                    frame.spacing_mm,
                    device,
                    config.data.target_spacing_mm,
                    config.data.size,
                )
            )
        return {stem: (labels[0], labels[1])}

    info: dict[str, object] = {
        "source": stem,
        "run": str(run),
        "train_git_commit": (run / "git_commit.txt").read_text().strip(),
    }
    return predict, info


def medsam_predictor(device: torch.device) -> tuple[Predictor, dict[str, object]]:
    from cardiac.models.medsam import load_medsam, segment_volume

    model, processor = load_medsam(device)

    def predict(meta: PatientMeta) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        per_phase = []
        for phase in ("ed", "es"):
            frame = load_frame(DEFAULT_DATA_ROOT, meta, phase)
            # The same seed per patient and phase as cardiac.evaluate_medsam, so the same boxes.
            rng = np.random.default_rng([SEED, zlib.crc32(f"{meta.pid}-{phase}".encode())])
            volumes = segment_volume(
                model, processor, frame.image, frame.label, device, JITTERS_PX, rng
            )
            per_phase.append({m: postprocess(v) for m, v in volumes.items()})
        return {f"medsam_jitter{m}": (per_phase[0][m], per_phase[1][m]) for m in JITTERS_PX}

    return predict, {"source": "medsam_jitter<m>", "jitters_px": list(JITTERS_PX)}


def measure(
    metas: list[PatientMeta], predict: Predictor
) -> Iterator[tuple[str, dict[str, object], dict[str, float]]]:
    """Yields (source stem, measurement row, Dice per structure and phase) for every patient."""
    for i, meta in enumerate(metas):
        frames = {p: load_frame(DEFAULT_DATA_ROOT, meta, p) for p in ("ed", "es")}
        spacing = frames["ed"].spacing_mm
        for stem, (ed, es) in predict(meta).items():
            row: dict[str, object] = {"pid": meta.pid, "group": meta.group}
            row["height_cm"], row["weight_kg"] = meta.height_cm, meta.weight_kg
            row.update(measure_patient(ed, es, spacing, meta.height_cm, meta.weight_kg))
            scores = {
                f"{phase}_{name}": dice(pred == value, frames[phase].label == value)
                for phase, pred in (("ed", ed), ("es", es))
                for value, name in STRUCTURES.items()
            }
            yield stem, row, scores
        print(f"  {i + 1}/{len(metas)} {meta.pid}", flush=True)


def check_same_masks(split: str, stem: str, scores: pd.DataFrame) -> float | None:
    """Largest Dice difference vs the evaluated run; None if there is nothing to compare with."""
    path = Path("results") / split / f"{stem}.csv"
    if not path.exists():
        return None
    scored = pd.read_csv(path)
    scored = scored[scored.postprocess == "post"].set_index(["pid", "phase"])
    worst = 0.0
    for pid, row in scores.iterrows():
        for phase in ("ed", "es"):
            for name in STRUCTURES.values():
                old = float(scored.loc[(pid, phase), f"dice_{name}"])
                worst = max(worst, abs(old - float(row[f"{phase}_{name}"])))
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(description="Clinical measurements per patient.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--expert", action="store_true", help="measure the expert labels")
    source.add_argument("--run", type=Path, help="run folder containing best.pt")
    source.add_argument("--medsam", action="store_true", help="MedSAM with oracle boxes")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--confirm-test", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.confirm_test:
        parser.error("the test set is used once per model, at the end; add --confirm-test")

    eval_commit = git_commit()  # before any output is written
    device = pick_device()
    if args.expert:
        predict, info = expert_predictor()
    elif args.run is not None:
        predict, info = run_predictor(args.run, device)
    else:
        predict, info = medsam_predictor(device)

    rows: dict[str, list[dict[str, object]]] = {}
    scores: dict[str, dict[str, dict[str, float]]] = {}
    for stem, row, dice_scores in measure(split_patients(args.split), predict):
        rows.setdefault(stem, []).append(row)
        scores.setdefault(stem, {})[str(row["pid"])] = dice_scores

    out_dir = Path("results") / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, stem_rows in rows.items():
        table = pd.DataFrame(stem_rows)
        table.to_csv(out_dir / f"clinical_{stem}.csv", index=False, float_format="%.4f")
        payload = {**info, "source": stem, "split": args.split, "eval_git_commit": eval_commit}
        payload["n_patients"] = len(table)
        if stem != "expert":
            worst = check_same_masks(args.split, stem, pd.DataFrame(scores[stem]).T)
            payload["max_dice_difference_vs_evaluation"] = worst
            print(f"{stem}: largest Dice difference vs results/{args.split}/{stem}.csv = {worst}")
        (out_dir / f"clinical_{stem}.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {out_dir}/clinical_{stem}.csv and .json ({len(table)} patients)")


if __name__ == "__main__":
    np.seterr(invalid="ignore")
    main()
