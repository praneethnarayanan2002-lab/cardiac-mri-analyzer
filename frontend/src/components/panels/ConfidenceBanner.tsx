import type { CaseSummary } from "../../data/types";
import { Icon } from "../ui/controls";

/** Shown when the doubt score is in the top 20% of validation patients (docs/decisions.md D31). */
export function ConfidenceBanner({ summary }: { summary: CaseSummary }) {
  const { score, threshold, flagged } = summary.uncertainty;
  if (!flagged) {
    return (
      <p className="flex items-center gap-2 text-xs text-muted">
        <Icon name="check" className="h-3.5 w-3.5 text-ok" />
        <span>
          Doubt score <span className="tabular">{score.toFixed(3)}</span>, below the review
          threshold of <span className="tabular">{threshold.toFixed(3)}</span>.
        </span>
      </p>
    );
  }
  return (
    <div
      role="status"
      className="rise flex gap-3 rounded-[var(--radius-card)] border border-warn/40 bg-warn/10 p-3 text-sm"
    >
      <Icon name="alert" className="mt-0.5 h-4 w-4 shrink-0 text-warn" />
      <div>
        <p className="font-medium text-text">Check this one: the model is less sure than usual.</p>
        <p className="mt-0.5 text-xs text-muted">
          Doubt score <span className="tabular">{score.toFixed(3)}</span> is above{" "}
          <span className="tabular">{threshold.toFixed(3)}</span>, the level only the 20% most
          doubtful validation patients reached. Turn on the doubt map (D) to see where. A flag means
          "review", not "wrong".
        </p>
      </div>
    </div>
  );
}
