import { useEffect } from "react";

import type { CaseSummary, MaskLayer, Phase } from "../../data/types";
import { MODEL_LABELS, MODEL_ORDER } from "../../lib/format";
import { useViewer } from "../../stores/viewerStore";
import { Icon, Segmented, Toggle } from "../ui/controls";

/** Slice slider, ED/ES, cine, model, opacity, doubt, compare. Keyboard: ↑/↓ slice, Space play,
 *  1-4 model, D doubt map, C compare (MASTER_PLAN 9.4). */
export function ViewerControls({ summary }: { summary: CaseSummary }) {
  const v = useViewer();
  const canPlay = summary.n_frames > 0;
  const models = MODEL_ORDER.map((m) => ({
    value: m,
    label: MODEL_LABELS[m],
    disabled: !summary.layers.includes(m),
    title: summary.layers.includes(m) ? undefined : "Sample patients only",
  }));

  useKeyboard(summary, canPlay);
  useCine(summary, canPlay);

  return (
    <div className="flex flex-col gap-3 rounded-[var(--radius-card)] border border-viewer-border bg-viewer-surface p-3 text-viewer-text">
      <div className="flex items-center gap-3">
        <label htmlFor="slice" className="w-20 shrink-0 text-xs text-viewer-muted">
          Slice <span className="tabular text-viewer-text">{v.slice + 1}</span>/{v.nSlices}
        </label>
        <input
          id="slice"
          type="range"
          min={0}
          max={v.nSlices - 1}
          value={v.slice}
          onChange={(e) => v.setSlice(Number(e.target.value))}
          className="w-full accent-[var(--primary)]"
          aria-valuetext={`slice ${v.slice + 1} of ${v.nSlices}, base to apex`}
        />
        <span className="hidden shrink-0 text-[11px] text-viewer-muted sm:block">base → apex</span>
      </div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <Segmented<Phase>
          dark
          label="Cardiac phase"
          value={v.phase}
          onChange={(phase) => v.set({ phase, playing: false })}
          options={[
            { value: "ed", label: "ED", title: "End-diastole: heart fully filled" },
            { value: "es", label: "ES", title: "End-systole: heart fully squeezed" },
          ]}
        />
        <button
          type="button"
          disabled={!canPlay}
          onClick={() => v.set({ playing: !v.playing })}
          title={canPlay ? "Play the heartbeat (Space)" : "Cine is available for sample patients"}
          aria-label={v.playing ? "Pause heartbeat" : "Play heartbeat"}
          className="inline-flex items-center gap-1.5 rounded-lg border border-viewer-border px-2.5 py-1 text-xs font-medium text-viewer-text hover:bg-viewer-bg disabled:opacity-40"
        >
          <Icon name={v.playing ? "pause" : "play"} className="h-3.5 w-3.5" />
          {v.playing ? "Pause" : "Cine"}
        </button>
        <Segmented<MaskLayer>
          dark
          label="Outline from"
          value={v.layer}
          onChange={(layer) => v.set({ layer })}
          options={models}
        />
      </div>
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
        <label className="flex items-center gap-2 text-xs text-viewer-muted">
          Opacity
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={v.opacity}
            onChange={(e) => v.set({ opacity: Number(e.target.value) })}
            className="w-24 accent-[var(--primary)]"
            aria-label="Outline opacity"
          />
        </label>
        <Toggle
          dark
          checked={v.showDoubt}
          onChange={(showDoubt) => v.set({ showDoubt })}
          label="Doubt map"
        />
        <Toggle
          dark
          checked={v.compare}
          onChange={(compare) => v.set({ compare })}
          label="Compare"
        />
        {v.compare && (
          <Segmented<MaskLayer>
            dark
            label="Compare with"
            value={v.compareLayer}
            onChange={(compareLayer) => v.set({ compareLayer })}
            options={models}
          />
        )}
      </div>
    </div>
  );
}

function useKeyboard(summary: CaseSummary, canPlay: boolean) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (target.closest("input, textarea, select, [contenteditable]")) return;
      const v = useViewer.getState();
      const model = MODEL_ORDER[Number(event.key) - 1];
      if (event.key === "ArrowUp" || event.key === "ArrowDown") {
        event.preventDefault();
        v.step(event.key === "ArrowUp" ? -1 : 1);
      } else if (event.key === " " && canPlay) {
        event.preventDefault();
        v.set({ playing: !v.playing });
      } else if (model && summary.layers.includes(model)) {
        v.set({ layer: model });
      } else if (event.key === "d" || event.key === "D") {
        v.set({ showDoubt: !v.showDoubt });
      } else if (event.key === "c" || event.key === "C") {
        v.set({ compare: !v.compare });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [summary, canPlay]);
}

/** One heartbeat per second: the mirror has no frame timing (docs/decisions.md D9), so the real
 *  frame rate is unknown. Pauses when the tab is hidden, because requestAnimationFrame does. */
function useCine(summary: CaseSummary, canPlay: boolean) {
  const playing = useViewer((s) => s.playing);
  useEffect(() => {
    if (!playing || !canPlay) return;
    const msPerFrame = 1000 / summary.n_frames;
    let last = performance.now();
    let id = requestAnimationFrame(function tick(now) {
      if (now - last >= msPerFrame) {
        last = now;
        const v = useViewer.getState();
        v.set({ frame: (v.frame + 1) % summary.n_frames });
      }
      id = requestAnimationFrame(tick);
    });
    return () => cancelAnimationFrame(id);
  }, [playing, canPlay, summary.n_frames]);
}
