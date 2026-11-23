import { useState } from "react";

import type { Measurement } from "../../data/types";
import { formatValue } from "../../lib/format";
import { useCountUp } from "../../lib/useCountUp";
import { Card } from "../ui/controls";
import { RangeBar } from "./RangeBar";
import { StatusChip } from "./StatusChip";

// The headline numbers first; the rest behind "Show all".
const HEADLINE = [
  "lv_ef_pct",
  "lv_edvi_ml_m2",
  "rv_ef_pct",
  "rv_edvi_ml_m2",
  "myo_mass_index_g_m2",
  "wall_thickness_max_mm",
];

function Row({ m }: { m: Measurement }) {
  const shown = useCountUp(m.value ?? 0);
  return (
    <li className="grid grid-cols-[1fr_auto] items-center gap-x-3 gap-y-1.5 py-2.5">
      <span className="text-sm text-text">{m.label}</span>
      <span className="tabular text-right text-sm font-medium text-text">
        {m.value === null ? "–" : formatValue(shown, m.unit)}
      </span>
      {m.range && m.value !== null ? (
        <RangeBar value={m.value} range={m.range} status={m.status} unit={m.unit} />
      ) : (
        <span />
      )}
      <span className="text-right">
        <StatusChip status={m.status} />
      </span>
    </li>
  );
}

export function MeasurementsCard({ measurements }: { measurements: Measurement[] }) {
  const [all, setAll] = useState(false);
  const shown = all ? measurements : measurements.filter((m) => HEADLINE.includes(m.key));
  return (
    <Card
      title="Measurements"
      delay={0}
      action={
        <button type="button" className="text-xs font-medium text-primary hover:underline" onClick={() => setAll(!all)}>
          {all ? "Show fewer" : `Show all ${measurements.length}`}
        </button>
      }
    >
      <ul className="divide-y divide-border">
        {shown.map((m) => (
          <Row key={m.key} m={m} />
        ))}
      </ul>
      <p className="mt-2 text-xs leading-relaxed text-muted">
        Shaded band: mean ± 2 SD of the 20 normal patients in the training set, measured from expert
        outlines with the same code. Not a clinical reference range.
      </p>
    </Card>
  );
}
