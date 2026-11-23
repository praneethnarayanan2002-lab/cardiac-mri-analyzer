import type { NormalRange, Status } from "../data/types";

/** Where things sit on a range bar, as fractions 0-1 of its width.
 *
 *  The bar spans the normal range plus one range-width of margin on each side, widened if needed
 *  so the value always fits. Values far outside still show, pinned near the edge. */
export interface RangePositions {
  normalStart: number;
  normalEnd: number;
  marker: number;
  min: number;
  max: number;
}

export function rangePositions(value: number, range: NormalRange): RangePositions {
  const width = Math.max(range.high - range.low, 1e-9);
  let min = range.low - width;
  let max = range.high + width;
  if (value < min) min = value - 0.05 * (max - value);
  if (value > max) max = value + 0.05 * (value - min);
  const at = (x: number) => (x - min) / (max - min);
  return { normalStart: at(range.low), normalEnd: at(range.high), marker: at(value), min, max };
}

/** Same rule as backend/app/services/analysis.range_status (docs/decisions.md D30). */
export function statusFor(value: number | null, range: NormalRange | null): Status {
  if (value === null || range === null || Number.isNaN(value)) return "unknown";
  const z = range.sd > 0 ? Math.abs(value - range.mean) / range.sd : 0;
  if (z <= 2) return "normal";
  return z <= 3 ? "borderline" : "abnormal";
}

export const STATUS_LABEL: Record<Status, string> = {
  normal: "Normal",
  borderline: "Borderline",
  abnormal: "Outside range",
  unknown: "No range",
};
