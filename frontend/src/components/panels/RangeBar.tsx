import type { NormalRange, Status } from "../../data/types";
import { formatValue } from "../../lib/format";
import { rangePositions } from "../../lib/normalRanges";

const MARKER: Record<Status, string> = {
  normal: "bg-ok",
  borderline: "bg-warn",
  abnormal: "bg-alert",
  unknown: "bg-muted",
};

/** A track with the normal band shaded and a marker at the value. */
export function RangeBar({
  value,
  range,
  status,
  unit,
}: {
  value: number;
  range: NormalRange;
  status: Status;
  unit: string;
}) {
  const p = rangePositions(value, range);
  const pct = (x: number) => `${(x * 100).toFixed(2)}%`;
  return (
    <div
      className="relative h-2 w-full rounded-full bg-surface-2"
      role="img"
      aria-label={`${formatValue(value, unit)}; normal range ${formatValue(range.low, unit)} to ${formatValue(range.high, unit)}`}
    >
      <div
        className="absolute inset-y-0 rounded-full bg-primary-soft"
        style={{ left: pct(p.normalStart), width: pct(p.normalEnd - p.normalStart) }}
      />
      <div
        className={`absolute top-1/2 h-3.5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-surface ${MARKER[status]}`}
        style={{ left: pct(p.marker) }}
      />
    </div>
  );
}
