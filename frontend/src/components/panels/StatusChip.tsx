import type { Status } from "../../data/types";
import { STATUS_LABEL } from "../../lib/normalRanges";
import { Icon } from "../ui/controls";

const STYLE: Record<Status, { tone: string; icon: "check" | "alert" | "info" }> = {
  normal: { tone: "text-ok", icon: "check" },
  borderline: { tone: "text-warn", icon: "alert" },
  abnormal: { tone: "text-alert", icon: "alert" },
  unknown: { tone: "text-muted", icon: "info" },
};

/** Status always comes with an icon and a word, never colour alone. */
export function StatusChip({ status }: { status: Status }) {
  const { tone, icon } = STYLE[status];
  return (
    <span className={`inline-flex items-center gap-1 text-xs font-medium ${tone}`}>
      <Icon name={icon} className="h-3.5 w-3.5" />
      {STATUS_LABEL[status]}
    </span>
  );
}
