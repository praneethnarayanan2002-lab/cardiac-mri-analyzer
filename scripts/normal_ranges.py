"""Write results/normal_ranges.json: what "normal" means for each measurement in this project.

Usage:
    uv run python scripts/normal_ranges.py

The range is mean ± 2 SD over the normal (NOR) patients among the 100 training patients, measured
from their expert masks with the same code as every other number here. It is not a clinical
reference range; it is the right comparison for numbers measured this way (docs/decisions.md D30).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cardiac.classify import FEATURES, add_derived
from cardiac.clinical import INDEXED, MEASUREMENTS


def main() -> None:
    table = pd.concat(
        [pd.read_csv(f"results/{s}/clinical_expert.csv") for s in ("train", "val")],
        ignore_index=True,
    )
    normal = add_derived(table[table.group == "NOR"])
    ranges = {}
    for key in dict.fromkeys([*MEASUREMENTS, *INDEXED.values(), *FEATURES]):
        values = normal[key].dropna().to_numpy(dtype=float)
        mean, sd = float(values.mean()), float(values.std(ddof=1))
        ranges[key] = {
            "low": mean - 2 * sd,
            "high": mean + 2 * sd,
            "mean": mean,
            "sd": sd,
            "n": len(values),
        }
    payload = {
        "definition": "mean ± 2 SD of the NOR training patients' expert-mask measurements",
        "patients": sorted(normal.pid.tolist()),
        "ranges": ranges,
    }
    Path("results/normal_ranges.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote results/normal_ranges.json from {len(normal)} NOR training patients")
    for key in ("lv_ef_pct", "lv_edvi_ml_m2", "rv_ef_pct", "myo_mass_index_g_m2"):
        r = ranges[key]
        print(f"  {key:<22} {r['low']:6.1f} - {r['high']:6.1f}  (mean {r['mean']:.1f})")
    assert np.isfinite([r["sd"] for r in ranges.values()]).all()


if __name__ == "__main__":
    main()
