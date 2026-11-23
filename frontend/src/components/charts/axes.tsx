import type { ReactNode } from "react";

import { type Scale, ticks } from "./scale";

/** Recessive axes and grid, and an in-chart tooltip. */

export function Axes({
  x,
  y,
  xLabel,
  yLabel,
  xTicks,
  yTicks,
  format = (v) => String(v),
}: {
  x: Scale;
  y: Scale;
  xLabel: string;
  yLabel: string;
  xTicks?: number[];
  yTicks?: number[];
  format?: (v: number) => string;
}) {
  const [x0, x1] = x.range;
  const [y0, y1] = y.range;
  const xs = xTicks ?? ticks(x.domain);
  const ys = yTicks ?? ticks(y.domain);
  return (
    <g className="text-[10px]" fill="var(--muted)">
      {ys.map((v) => (
        <g key={`y${v}`}>
          <line x1={x0} x2={x1} y1={y(v)} y2={y(v)} stroke="var(--border)" strokeWidth={1} />
          <text x={x0 - 6} y={y(v)} textAnchor="end" dominantBaseline="middle">
            {format(v)}
          </text>
        </g>
      ))}
      {xs.map((v) => (
        <text key={`x${v}`} x={x(v)} y={y0 + 14} textAnchor="middle">
          {format(v)}
        </text>
      ))}
      <line x1={x0} x2={x1} y1={y0} y2={y0} stroke="var(--muted)" strokeOpacity={0.5} />
      <text x={(x0 + x1) / 2} y={y0 + 32} textAnchor="middle" className="text-[11px]">
        {xLabel}
      </text>
      <text
        transform={`translate(${x0 - 38} ${(y0 + y1) / 2}) rotate(-90)`}
        textAnchor="middle"
        className="text-[11px]"
      >
        {yLabel}
      </text>
    </g>
  );
}

/** A tooltip box drawn in SVG, kept inside the chart frame. */
export function Tip({ x, y, width, lines }: { x: number; y: number; width: number; lines: ReactNode[] }) {
  const w = 170;
  const h = 14 * lines.length + 10;
  const left = Math.min(Math.max(x + 10, 4), width - w - 4);
  const top = Math.max(y - h - 8, 4);
  return (
    <g pointerEvents="none">
      <rect x={left} y={top} width={w} height={h} rx={6} fill="var(--surface)" stroke="var(--border)" />
      {lines.map((line, i) => (
        <text key={i} x={left + 8} y={top + 16 + i * 14} className="text-[11px]" fill={i ? "var(--muted)" : "var(--text)"}>
          {line}
        </text>
      ))}
    </g>
  );
}
