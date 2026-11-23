import type { CaseSummary } from "../../data/types";
import { formatValue, GROUP_NAMES } from "../../lib/format";
import { Card } from "../ui/controls";

/** The top SHAP reasons for the suggested group, as bars and plain sentences. Bars toward the
 *  group grow right in the primary colour; bars away from it grow left in grey. */
export function WhyCard({ summary }: { summary: CaseSummary }) {
  const reasons = summary.why;
  const largest = Math.max(...reasons.map((r) => Math.abs(r.shap)), 1e-9);
  const group = summary.diagnosis.predicted;
  return (
    <Card title="Why" delay={80}>
      <p className="mb-3 text-xs text-muted">
        What moved the model toward or away from{" "}
        <span className="font-medium text-text">{GROUP_NAMES[group]}</span>, largest effect first.
      </p>
      <ul className="space-y-3">
        {reasons.map((r) => {
          const width = `${((Math.abs(r.shap) / largest) * 50).toFixed(1)}%`;
          const toward = r.shap > 0;
          return (
            <li key={r.feature}>
              <p className="text-sm text-text">
                {r.text}{" "}
                <span className="tabular text-xs text-muted">({formatValue(r.value, r.unit)})</span>
              </p>
              <div className="relative mt-1 h-1.5 rounded-full bg-surface-2" aria-hidden="true">
                <span className="absolute inset-y-[-3px] left-1/2 w-px bg-border" />
                <span
                  className={`absolute inset-y-0 rounded-full ${toward ? "left-1/2 bg-primary" : "right-1/2 bg-muted/60"}`}
                  style={{ width }}
                />
              </div>
            </li>
          );
        })}
      </ul>
      <p className="mt-3 text-xs leading-relaxed text-muted">
        SHAP values explain this model, not the disease. "High" and "low" are relative to the normal
        patients' average.
      </p>
    </Card>
  );
}
