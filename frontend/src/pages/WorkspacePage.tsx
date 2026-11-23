import { Link, useParams } from "react-router-dom";

import { ConfidenceBanner } from "../components/panels/ConfidenceBanner";
import { DiagnosisCard } from "../components/panels/DiagnosisCard";
import { MeasurementsCard } from "../components/panels/MeasurementsCard";
import { WhyCard } from "../components/panels/WhyCard";
import { ErrorState, Icon, Skeleton } from "../components/ui/controls";
import { Legend } from "../components/viewer/Legend";
import { ScanViewer } from "../components/viewer/ScanViewer";
import { ViewerControls } from "../components/viewer/ViewerControls";
import { dataSource } from "../data/dataSource";
import type { CaseSummary } from "../data/types";
import { GROUP_NAMES, MODEL_LABELS } from "../lib/format";
import { useLoad } from "../lib/useLoad";
import { useViewer } from "../stores/viewerStore";

export function WorkspacePage() {
  const { id = "" } = useParams();
  const { layer, compareLayer, compare, showDoubt } = useViewer();
  const { data: summary, error, retry } = useLoad(() => dataSource.getCase(id).then(startViewer), id);

  if (error) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-10">
        <ErrorState message={error} onRetry={retry} />
        <Link to="/" className="mt-4 inline-block text-sm text-primary hover:underline">
          Back to the samples
        </Link>
      </div>
    );
  }
  if (!summary) {
    return (
      <div className="mx-auto grid max-w-7xl gap-6 px-4 py-6 lg:grid-cols-[3fr_2fr]">
        <Skeleton className="h-[60vh]" />
        <div className="grid gap-4">
          <Skeleton className="h-64" />
          <Skeleton className="h-40" />
        </div>
      </div>
    );
  }

  const title = summary.pid ?? "Uploaded scan";
  return (
    <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
      <header className="rise mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-muted">
            {summary.source === "sample" ? "Sample test patient" : "Your upload"}
          </p>
          <h1 className="text-2xl font-semibold text-text">
            {title}
            {summary.true_group && (
              <span className="ml-2 text-base font-normal text-muted">
                expert: {GROUP_NAMES[summary.true_group]}
              </span>
            )}
          </h1>
        </div>
        <a
          href={dataSource.reportUrl(summary.id)}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-2 rounded-lg bg-primary px-3 py-2 text-sm font-medium text-white hover:opacity-90"
        >
          <Icon name="download" />
          PDF report
        </a>
      </header>

      <div className="grid gap-6 lg:grid-cols-[3fr_2fr]">
        <div className="flex min-w-0 flex-col gap-3">
          <div className={`grid gap-3 ${compare ? "sm:grid-cols-2" : ""}`}>
            <div className={compare ? "aspect-square" : "aspect-square max-h-[70vh] w-full"}>
              <ScanViewer summary={summary} layer={layer} caption={compare ? MODEL_LABELS[layer] : undefined} />
            </div>
            {compare && (
              <div className="aspect-square">
                <ScanViewer summary={summary} layer={compareLayer} caption={MODEL_LABELS[compareLayer]} />
              </div>
            )}
          </div>
          <ViewerControls summary={summary} />
          <Legend showDoubt={showDoubt} />
          <p className="text-xs text-muted">
            Scroll or ↑/↓ to move through slices · Ctrl/⌘ + scroll to zoom, drag to pan, double-click
            to reset · Space plays the heartbeat · 1–4 switch outline · D doubt map · C compare
          </p>
          <ConfidenceBanner summary={summary} />
        </div>
        <aside className="flex flex-col gap-4">
          <MeasurementsCard measurements={summary.measurements} />
          <DiagnosisCard summary={summary} />
          <WhyCard summary={summary} />
        </aside>
      </div>
    </div>
  );
}

/** Resets the viewer for a newly loaded case: middle slice (usually the heart at its widest),
 *  pretrained U-Net outline, expert (or scratch U-Net) as the compare side. */
function startViewer(summary: CaseSummary): CaseSummary {
  const n = summary.shape[2] ?? 1;
  const second = summary.layers.includes("expert") ? "expert" : "unet";
  useViewer.getState().reset(n, Math.floor(n / 2), "unet_pretrained", second);
  return summary;
}
