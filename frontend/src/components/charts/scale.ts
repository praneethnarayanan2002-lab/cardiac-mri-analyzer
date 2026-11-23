/** A linear scale, "nice" ticks, padded extents and the shared chart margin. */

export interface Scale {
  (x: number): number;
  domain: [number, number];
  range: [number, number];
}

export function linear(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const f = ((x: number) => r0 + ((x - d0) / (d1 - d0 || 1)) * (r1 - r0)) as Scale;
  f.domain = domain;
  f.range = range;
  return f;
}

export function ticks([lo, hi]: [number, number], count = 5): number[] {
  const span = hi - lo || 1;
  const raw = span / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * power).find((s) => s >= raw) ?? power * 10;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

/** Pads a data extent so marks don't sit on the frame. */
export function extent(values: number[], pad = 0.08): [number, number] {
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const p = (hi - lo || Math.abs(hi) || 1) * pad;
  return [lo - p, hi + p];
}

export const MARGIN = { top: 12, right: 16, bottom: 40, left: 48 };
