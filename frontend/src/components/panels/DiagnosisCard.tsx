import type { CaseSummary, Group } from "../../data/types";
import { GROUP_NAMES, GROUP_ORDER, percent } from "../../lib/format";
import { Card } from "../ui/controls";

export function DiagnosisCard({ summary }: { summary: CaseSummary }) {
  const { predicted, probabilities } = summary.diagnosis;
  const ordered = [...GROUP_ORDER].sort((a, b) => probabilities[b] - probabilities[a]);
  return (
    <Card title="Suggested group" delay={40}>
      <p className="text-lg font-semibold text-text">
        {GROUP_NAMES[predicted]} <span className="text-sm font-normal text-muted">({predicted})</span>
      </p>
      {summary.true_group && (
        <p className="mt-0.5 text-xs text-muted">
          Expert label for this sample: {GROUP_NAMES[summary.true_group]} ({summary.true_group})
          {summary.true_group !== predicted && " — the model gets this one wrong"}
        </p>
      )}
      <ul className="mt-3 space-y-2" aria-label="Probability per group">
        {ordered.map((g: Group) => (
          <li key={g} className="grid grid-cols-[3.5rem_1fr_3rem] items-center gap-2 text-xs">
            <span className={g === predicted ? "font-semibold text-text" : "text-muted"}>{g}</span>
            <span className="h-2 rounded-full bg-surface-2">
              <span
                className={`block h-2 rounded-full transition-[width] duration-300 ${g === predicted ? "bg-primary" : "bg-muted/50"}`}
                style={{ width: percent(probabilities[g], 1) }}
              />
            </span>
            <span className="tabular text-right text-text">{percent(probabilities[g])}</span>
          </li>
        ))}
      </ul>
      <p className="mt-3 text-xs leading-relaxed text-muted">
        XGBoost on the measurements, trained on expert outlines of 100 training patients. Its
        test-set accuracy is on the Model comparison page.
      </p>
    </Card>
  );
}
