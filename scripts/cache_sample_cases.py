"""Precompute the app's 5 sample patients into data/cache/samples/ (git-ignored).

Usage:
    PYTHONPATH=. uv run python scripts/cache_sample_cases.py   # or: make samples DEVICE=mps

One test patient per diagnosis group, chosen by a fixed rule (docs/decisions.md D29): within each
group, sort the 10 test patients by the pretrained U-Net's (seed 0) mean Dice and take the 5th, the
lower of the two middle ones. Each sample gets everything an upload gets, plus the expert and MedSAM
layers and the cine, so the app can show all four models side by side.
"""

import json
import zlib
from pathlib import Path

# Must run before torch or xgboost load (docs/decisions.md D28).
import backend.app  # noqa: F401  # isort: skip

import numpy as np
import pandas as pd

from backend.app.config import get_settings
from backend.app.services.analysis import analyse
from backend.app.services.images import window_volume
from backend.app.services.models import load_models
from backend.app.services.store import Case, save_sample
from cardiac.data.io import (
    ACDC_GROUPS,
    DEFAULT_DATA_ROOT,
    load_frame,
    load_metadata,
    load_nifti,
    patient_files,
)
from cardiac.evaluate_medsam import SEED
from cardiac.infer import postprocess
from cardiac.models.medsam import load_medsam, segment_volume

DICE = ["dice_RV", "dice_MYO", "dice_LV"]


def choose_samples(results: Path) -> dict[str, str]:
    """{group: pid}: per group, the 5th of 10 test patients by pretrained seed-0 mean Dice."""
    table = pd.read_csv(results / "test" / "unet_resnet34_seed0.csv")
    table = table[table.postprocess == "post"]
    per_patient = table.assign(dice=table[DICE].mean(axis=1)).groupby(["group", "pid"]).dice.mean()
    chosen = {}
    for group in ACDC_GROUPS:
        ranked = per_patient[group].sort_values()
        chosen[group] = str(ranked.index[(len(ranked) - 1) // 2])
    return chosen


def main() -> None:
    settings = get_settings()
    models = load_models(settings)
    medsam, processor = load_medsam(models.device)
    chosen = choose_samples(settings.results_dir)
    metas = {m.pid: m for m in load_metadata(DEFAULT_DATA_ROOT, "test")}
    for group, pid in chosen.items():
        meta = metas[pid]
        frames = {p: load_frame(DEFAULT_DATA_ROOT, meta, p) for p in ("ed", "es")}
        spacing = frames["ed"].spacing_mm
        images = {p: f.image for p, f in frames.items()}
        result = analyse(models, images, spacing, meta.height_cm, meta.weight_kg)
        for phase, frame in frames.items():
            # Same boxes as cardiac.evaluate_medsam: same seed per patient and phase, no jitter.
            rng = np.random.default_rng([SEED, zlib.crc32(f"{pid}-{phase}".encode())])
            boxes = segment_volume(
                medsam, processor, frame.image, frame.label, models.device, (0,), rng
            )
            result.layers[phase]["medsam"] = postprocess(boxes[0])
            result.layers[phase]["expert"] = frame.label
        cine, _ = load_nifti(patient_files(DEFAULT_DATA_ROOT, meta)["cine"])
        summary = {
            "id": pid,
            "source": "sample",
            "pid": pid,
            "true_group": group,
            "layers": ["image", "unet", "unet_pretrained", "medsam", "expert", "doubt"],
            "n_frames": int(cine.shape[3]),
            # The metadata counts frames from 1; arrays count from 0.
            "ed_frame": meta.ed_frame - 1,
            "es_frame": meta.es_frame - 1,
            **result.summary,
        }
        case = Case(
            case_id=pid,
            summary=summary,
            images={p: window_volume(v) for p, v in images.items()},
            layers=result.layers,
            cine=window_volume(np.asarray(cine, dtype=np.float32)),
        )
        save_sample(case, settings.samples_dir)
        d = summary["diagnosis"]
        assert isinstance(d, dict)
        print(
            f"{group:<5} {pid}: predicted {d['predicted']}, "
            f"doubt flagged {summary['uncertainty']['flagged']}"
        )  # type: ignore[index]
    # The choice is a result of the evaluation, so it is recorded with the results, not the cache.
    record = {"rule": "per group, 5th of 10 test patients by pretrained seed-0 mean Dice", **chosen}
    (settings.results_dir / "samples.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {len(chosen)} samples to {settings.samples_dir}")


if __name__ == "__main__":
    main()
