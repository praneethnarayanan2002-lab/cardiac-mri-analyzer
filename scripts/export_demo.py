"""Export the 5 sample patients and the test-set metrics as static files for the public demo.

Usage:
    PYTHONPATH=. uv run python scripts/export_demo.py      # or: make demo

Reads the sample cache (`make samples`) and results/, and writes frontend/public/demo/, the layout
`frontend/src/data/staticSource.ts` expects:

    samples.json                              like GET /api/samples
    cases/<id>.json                           like GET /api/cases/<id>
    slices/<id>/<phase>/<layer>/<k>.png       like GET /api/cases/<id>/slice/<phase>/<k>?layer=
    cine/<id>/<k>/<frame>.png                 like GET /api/cases/<id>/cine/<k>?frame=
    reports/<id>.pdf                          like GET /api/cases/<id>/report.pdf
    metrics.json                              like GET /api/metrics/summary

Every file is produced by the same backend code that serves the live app, so the demo and the app
can't disagree. Prints the total size, which must stay under the plan's 60 MB budget.
"""

import json
import shutil
from pathlib import Path

# Must run before torch or xgboost load (docs/decisions.md D28).
import backend.app  # noqa: F401  # isort: skip

from backend.app.config import get_settings
from backend.app.report import build_report
from backend.app.services.images import png_bytes
from backend.app.services.metrics import metrics_summary
from backend.app.services.store import CaseStore

OUT = Path("frontend/public/demo")
BUDGET_MB = 60

DATA_NOTICE = """# Data in this folder

The images and outlines here come from five test patients of the ACDC dataset:

O. Bernard et al., "Deep Learning Techniques for Automatic MRI Cardiac Multi-Structures Segmentation
and Diagnosis: Is the Problem Solved?", IEEE Transactions on Medical Imaging, 37(11), 2018.
https://www.creatis.insa-lyon.fr/Challenge/acdc/

ACDC is licensed CC BY-NC-SA 4.0 (attribution, non-commercial, share-alike). These files are derived
from it (resampled, windowed, exported as PNG, with model outlines added) and are shared under the
same licence, for non-commercial research and teaching only. The code of this project is
MIT-licensed; that licence does not cover this folder.
"""


def write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main() -> None:
    settings = get_settings()
    store = CaseStore(settings.samples_dir, settings.case_ttl_s)
    ids = store.sample_ids()
    if not ids:
        raise SystemExit(f"no samples in {settings.samples_dir}; run `make samples` first")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    samples = []
    for case_id in ids:
        case = store.get(case_id)
        assert case is not None
        s = case.summary
        samples.append(
            {
                "id": case_id,
                "pid": s["pid"],
                "true_group": s["true_group"],
                "predicted": s["diagnosis"]["predicted"],  # type: ignore[index]
                "flagged": s["uncertainty"]["flagged"],  # type: ignore[index]
            }
        )
        write(OUT / "cases" / f"{case_id}.json", json.dumps(s, indent=1).encode())
        for phase, image in case.images.items():
            volumes = {"image": image, **case.layers[phase]}
            for layer, volume in volumes.items():
                for k in range(volume.shape[2]):
                    path = OUT / "slices" / case_id / phase / layer / f"{k}.png"
                    write(path, png_bytes(volume[:, :, k]))
        if case.cine is not None:
            for k in range(case.cine.shape[2]):
                for t in range(case.cine.shape[3]):
                    write(
                        OUT / "cine" / case_id / str(k) / f"{t}.png",
                        png_bytes(case.cine[:, :, k, t]),
                    )
        write(OUT / "reports" / f"{case_id}.pdf", build_report(case))
        print(f"  {case_id}: exported")

    write(OUT / "samples.json", json.dumps(samples, indent=1).encode())
    metrics = metrics_summary(settings.results_dir)
    write(OUT / "metrics.json", json.dumps(metrics, separators=(",", ":")).encode())
    write(OUT / "DATA_LICENSE.md", DATA_NOTICE.encode())

    files = [p for p in OUT.rglob("*") if p.is_file()]
    total_mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"wrote {len(files)} files, {total_mb:.1f} MB, to {OUT}")
    if total_mb > BUDGET_MB:
        raise SystemExit(f"demo is {total_mb:.1f} MB, over the {BUDGET_MB} MB budget")


if __name__ == "__main__":
    main()
