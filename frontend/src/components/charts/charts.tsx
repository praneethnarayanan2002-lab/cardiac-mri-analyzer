import { useState } from "react";

import type { Group, MetricsSummary, ModelKey, Structure } from "../../data/types";
import { GROUP_ORDER } from "../../lib/format";
import { Axes, Tip } from "./axes";
import { MODEL_NAMES } from "./models";
import { extent, linear, MARGIN } from "./scale";

// Categorical slots from the validated default palette, in fixed order; MedSAM gets the orange so
// it reads as "different kind of model".
const MODEL_COLOURS: Record<ModelKey, string> = {
  unet_scratch: "#2a78d6",
  unet_resnet34: "#1baf7a",
  medsam_jitter0: "#eb6834",
};
const GROUP_COLOURS: Record<Group, string> = {
  NOR: "#2a78d6",
  MINF: "#eb6834",
  DCM: "#1baf7a",
  HCM: "#eda100",
  RV: "#e87ba4",
};
const STRUCTURES: Structure[] = ["RV", "MYO", "LV"];
const STRUCTURE_COLOURS: Record<Structure, string> = { RV: "#3B82F6", MYO: "#22C55E", LV: "#EF4444" };

function ModelKey({ models }: { models: ModelKey[] }) {
  return (
    <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
      {models.map((m) => (
        <li key={m} className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: MODEL_COLOURS[m] }} />
          {MODEL_NAMES[m]}
        </li>
      ))}
    </ul>
  );
}

/** Test-set Dice per structure and model; whiskers are ± std over training seeds. */
export function DiceBars({ data }: { data: MetricsSummary["segmentation"] }) {
  const models = (Object.keys(MODEL_NAMES) as ModelKey[]).filter((m) => data.dice[m]);
  const [hover, setHover] = useState<{ m: ModelKey; s: Structure } | null>(null);
  const W = 560;
  const H = 240;
  const x0 = MARGIN.left;
  const x1 = W - MARGIN.right;
  const y = linear([0, 1], [H - MARGIN.bottom, MARGIN.top]);
  const group = (x1 - x0) / STRUCTURES.length;
  const bar = Math.min(28, (group * 0.7) / models.length);
  const gap = 2;
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Dice by structure and model">
        <Axes x={linear([0, 1], [x0, x1])} y={y} xTicks={[]} yTicks={[0, 0.25, 0.5, 0.75, 1]} xLabel="" yLabel="Dice" />
        {STRUCTURES.map((s, si) => {
          const cx = x0 + group * (si + 0.5);
          return (
            <g key={s}>
              <text x={cx} y={H - MARGIN.bottom + 16} textAnchor="middle" className="text-[11px]" fill="var(--text)">
                {s}
              </text>
              {models.map((m, mi) => {
                const v = data.dice[m][s];
                const left = cx + (mi - models.length / 2) * (bar + gap);
                const top = y(v.mean);
                return (
                  <g key={m} onMouseEnter={() => setHover({ m, s })} onMouseLeave={() => setHover(null)}>
                    <rect x={left - 2} y={MARGIN.top} width={bar + 4} height={H - MARGIN.bottom - MARGIN.top} fill="transparent" />
                    <path
                      d={`M${left},${y(0)} V${top + 4} Q${left},${top} ${left + 4},${top} H${left + bar - 4} Q${left + bar},${top} ${left + bar},${top + 4} V${y(0)} Z`}
                      fill={MODEL_COLOURS[m]}
                      opacity={hover && (hover.m !== m || hover.s !== s) ? 0.45 : 1}
                    />
                    {v.std > 0 && (
                      <line x1={left + bar / 2} x2={left + bar / 2} y1={y(v.mean - v.std)} y2={y(v.mean + v.std)} stroke="var(--text)" strokeWidth={1.5} />
                    )}
                  </g>
                );
              })}
            </g>
          );
        })}
        {hover && (
          <Tip
            x={x0 + group * (STRUCTURES.indexOf(hover.s) + 0.5)}
            y={y(data.dice[hover.m][hover.s].mean)}
            width={W}
            lines={[
              `${MODEL_NAMES[hover.m]}, ${hover.s}`,
              `Dice ${data.dice[hover.m][hover.s].mean.toFixed(3)}` +
                (data.dice[hover.m][hover.s].std > 0 ? ` ± ${data.dice[hover.m][hover.s].std.toFixed(3)}` : ""),
              `HD95 ${data.hd95[hover.m][hover.s].mean.toFixed(2)} mm`,
            ]}
          />
        )}
      </svg>
      <ModelKey models={models} />
    </div>
  );
}

/** Model − expert against their mean, one dot per test patient, with bias and 95% limits. */
export function BlandAltman({
  data,
  model,
  measurement,
}: {
  data: MetricsSummary["clinical"];
  model: ModelKey;
  measurement: string;
}) {
  const points = data.points[model]?.[measurement] ?? [];
  const stats = data.stats[model]?.[measurement];
  const unit = data.units[measurement] ?? "";
  const [hover, setHover] = useState<number | null>(null);
  if (!stats || points.length === 0) return <p className="text-sm text-muted">No data.</p>;
  const W = 560;
  const H = 300;
  const means = points.map((p) => (p.model + p.expert) / 2);
  const diffs = points.map((p) => p.model - p.expert);
  // The dots are seed 0, so the lines are seed 0's statistics (from results/clinical.json).
  const { bias, loa_low: lo, loa_high: hi } = stats.seed0;
  const x = linear(extent(means), [MARGIN.left, W - MARGIN.right - 64]);
  const y = linear(extent([...diffs, lo, hi, 0]), [H - MARGIN.bottom, MARGIN.top]);
  const [xa, xb] = x.range;
  const line = (v: number, label: string, dash?: string) => (
    <g>
      <line x1={xa} x2={xb} y1={y(v)} y2={y(v)} stroke="var(--text)" strokeWidth={label === "bias" ? 1.5 : 1} strokeDasharray={dash} />
      <text x={xb + 6} y={y(v)} dominantBaseline="middle" className="text-[10px]" fill="var(--muted)">
        {label} {v >= 0 ? "+" : ""}
        {v.toFixed(1)}
      </text>
    </g>
  );
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Bland-Altman plot">
        <Axes x={x} y={y} xLabel={`mean of model and expert (${unit})`} yLabel={`model − expert (${unit})`} />
        <line x1={xa} x2={xb} y1={y(0)} y2={y(0)} stroke="var(--border)" strokeWidth={2} />
        {line(hi, "+LoA", "4 3")}
        {line(bias, "bias")}
        {line(lo, "−LoA", "4 3")}
        {points.map((p, i) => (
          <circle
            key={p.pid}
            cx={x(means[i]!)}
            cy={y(diffs[i]!)}
            r={hover === i ? 6 : 4.5}
            fill={GROUP_COLOURS[p.group]}
            stroke="var(--surface)"
            strokeWidth={1.5}
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover(null)}
          />
        ))}
        {hover !== null && (
          <Tip
            x={x(means[hover]!)}
            y={y(diffs[hover]!)}
            width={W}
            lines={[
              `${points[hover]!.pid} (${points[hover]!.group})`,
              `model ${points[hover]!.model.toFixed(1)}, expert ${points[hover]!.expert.toFixed(1)} ${unit}`,
              `difference ${diffs[hover]!.toFixed(1)} ${unit}`,
            ]}
          />
        )}
      </svg>
      <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
        {GROUP_ORDER.map((g) => (
          <li key={g} className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full" style={{ background: GROUP_COLOURS[g] }} />
            {g}
          </li>
        ))}
      </ul>
      <p className="mt-2 text-xs text-muted">
        Over 3 seeds: bias {stats.bias.mean.toFixed(1)} ± {stats.bias.std.toFixed(1)}, limits of agreement{" "}
        {stats.loa_low.mean.toFixed(1)} to {stats.loa_high.mean.toFixed(1)} {unit}, mean absolute error{" "}
        {stats.mae.mean.toFixed(1)} {unit}, Pearson r {stats.pearson_r.mean.toFixed(3)}. Dots and lines:
        seed {data.points_seed}.
      </p>
    </div>
  );
}

/** Rows: expert group; columns: predicted group; counts of test patients. */
export function ConfusionMatrix({ matrix, title }: { matrix: number[][]; title: string }) {
  const max = Math.max(...matrix.flat(), 1);
  const correct = matrix.reduce((a, row, i) => a + (row[i] ?? 0), 0);
  const total = matrix.flat().reduce((a, b) => a + b, 0);
  return (
    <figure>
      <figcaption className="mb-2 text-xs font-medium text-text">
        {title} <span className="font-normal text-muted">· {correct}/{total} right</span>
      </figcaption>
      <table className="tabular w-full table-fixed border-separate border-spacing-0.5 text-center text-xs">
        <thead>
          <tr>
            <th className="w-12 text-left font-normal text-muted">
              <span className="sr-only">expert group</span>
            </th>
            {GROUP_ORDER.map((g) => (
              <th key={g} scope="col" className="font-normal text-muted">
                {g}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={GROUP_ORDER[i]}>
              <th scope="row" className="text-left font-normal text-muted">
                {GROUP_ORDER[i]}
              </th>
              {row.map((n, j) => {
                const strength = n / max;
                return (
                  <td
                    key={j}
                    title={`expert ${GROUP_ORDER[i]}, predicted ${GROUP_ORDER[j]}: ${n}`}
                    className="h-8 rounded"
                    style={{
                      background: n ? `color-mix(in oklab, var(--primary) ${Math.round(15 + 70 * strength)}%, var(--surface))` : "var(--surface-2)",
                      color: strength > 0.55 ? "white" : "var(--text)",
                      outline: i === j ? "1.5px solid var(--primary)" : undefined,
                    }}
                  >
                    {n || ""}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-1 text-[11px] text-muted">rows: expert group · columns: predicted</p>
    </figure>
  );
}

/** MedSAM Dice per structure as box edges are moved by up to ± n pixels. */
export function JitterCurve({ data }: { data: MetricsSummary["segmentation"] }) {
  const W = 560;
  const H = 240;
  const px = data.jitter.map((j) => j.px);
  const x = linear([0, Math.max(...px)], [MARGIN.left, W - MARGIN.right - 40]);
  const y = linear([0, 1], [H - MARGIN.bottom, MARGIN.top]);
  const [hover, setHover] = useState<number | null>(null);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="MedSAM Dice against box jitter">
      <Axes x={x} y={y} xTicks={px} yTicks={[0, 0.25, 0.5, 0.75, 1]} xLabel="box edges moved by up to ± px (1 px = 1 mm)" yLabel="Dice" />
      {STRUCTURES.map((s) => (
        <g key={s}>
          <polyline
            points={data.jitter.map((j) => `${x(j.px)},${y(j[s])}`).join(" ")}
            fill="none"
            stroke={STRUCTURE_COLOURS[s]}
            strokeWidth={2}
          />
          {data.jitter.map((j, i) => (
            <circle key={j.px} cx={x(j.px)} cy={y(j[s])} r={4} fill={STRUCTURE_COLOURS[s]} stroke="var(--surface)" strokeWidth={1.5} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} />
          ))}
          <text x={x(Math.max(...px)) + 8} y={y(data.jitter.at(-1)![s])} dominantBaseline="middle" className="text-[11px]" fill="var(--text)">
            {s}
          </text>
        </g>
      ))}
      {hover !== null && (
        <Tip
          x={x(data.jitter[hover]!.px)}
          y={y(data.jitter[hover]!.LV)}
          width={W}
          lines={[`jitter ± ${data.jitter[hover]!.px} px`, ...STRUCTURES.map((s) => `${s} Dice ${data.jitter[hover]![s].toFixed(3)}`)]}
        />
      )}
    </svg>
  );
}

/** Keep the most confident patients first: mean error of those kept, against coverage. */
export function RiskCoverage({ data }: { data: MetricsSummary["uncertainty"] }) {
  const W = 560;
  const H = 260;
  const curve = data.curve_seed0;
  const x = linear([0, 1], [MARGIN.left, W - MARGIN.right]);
  const y = linear(extent([...curve.map((c) => c.risk), ...curve.map((c) => c.oracle), data.random_risk_seed0]), [H - MARGIN.bottom, MARGIN.top]);
  const [hover, setHover] = useState<number | null>(null);
  const path = (key: "risk" | "oracle") => curve.map((c) => `${x(c.coverage)},${y(c[key])}`).join(" ");
  return (
    <div>
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="w-full"
      role="img"
      aria-label="Risk-coverage curve"
      onMouseLeave={() => setHover(null)}
      onMouseMove={(e) => {
        const box = e.currentTarget.getBoundingClientRect();
        const cov = ((e.clientX - box.left) * (W / box.width) - x.range[0]) / (x.range[1] - x.range[0]);
        setHover(Math.max(0, Math.min(curve.length - 1, Math.round(cov * curve.length) - 1)));
      }}
    >
      <Axes x={x} y={y} xTicks={[0, 0.25, 0.5, 0.75, 1]} format={(v) => (v <= 1 && v >= 0 && Number.isInteger(v * 4) ? `${Math.round(v * 100)}%` : v.toFixed(3))} xLabel="patients kept, most confident first" yLabel="mean error (1 − Dice)" />
      <line x1={x(0)} x2={x(1)} y1={y(data.random_risk_seed0)} y2={y(data.random_risk_seed0)} stroke="var(--muted)" strokeDasharray="4 3" />
      <polyline points={path("oracle")} fill="none" stroke="var(--muted)" strokeWidth={1.5} strokeDasharray="1 3" />
      <polyline points={path("risk")} fill="none" stroke="var(--primary)" strokeWidth={2} />
      {hover !== null && (
        <>
          <line x1={x(curve[hover]!.coverage)} x2={x(curve[hover]!.coverage)} y1={y.range[0]} y2={y.range[1]} stroke="var(--border)" />
          <circle cx={x(curve[hover]!.coverage)} cy={y(curve[hover]!.risk)} r={4} fill="var(--primary)" />
          <Tip
            x={x(curve[hover]!.coverage)}
            y={y(curve[hover]!.risk)}
            width={W}
            lines={[
              `keep ${Math.round(curve[hover]!.coverage * 100)}% of patients`,
              `mean Dice of kept ${(1 - curve[hover]!.risk).toFixed(3)}`,
              `perfect ordering ${(1 - curve[hover]!.oracle).toFixed(3)}`,
            ]}
          />
        </>
      )}
    </svg>
    <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
      <li className="flex items-center gap-1.5"><span className="h-0.5 w-5 bg-primary" />ordered by doubt score (seed 0)</li>
      <li className="flex items-center gap-1.5"><span className="h-0 w-5 border-t-2 border-dotted border-muted" />perfect ordering</li>
      <li className="flex items-center gap-1.5"><span className="h-0 w-5 border-t border-dashed border-muted" />random ordering</li>
    </ul>
    </div>
  );
}
