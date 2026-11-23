import { useCallback, useEffect, useRef, useState } from "react";

import { dataSource } from "../../data/dataSource";
import type { CaseSummary, MaskLayer } from "../../data/types";
import { MODEL_LABELS } from "../../lib/format";
import { getLayer, preload } from "../../lib/imageCache";
import { useViewer } from "../../stores/viewerStore";

interface Props {
  summary: CaseSummary;
  layer: MaskLayer;
  /** Shown in the corner; compare mode labels each side. */
  caption?: string;
}

const FADE_MS = 200;

/** One scan on a dark canvas: greyscale slice, coloured mask on top, optional doubt map.
 *  The image is drawn smoothly; masks and doubt with hard pixel edges, so a voxel looks like one. */
export function ScanViewer({ summary, layer, caption }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const layers = useRef<{
    image: HTMLCanvasElement | null;
    mask: HTMLCanvasElement | null;
    prevMask: HTMLCanvasElement | null;
    doubt: HTMLCanvasElement | null;
    fadeStart: number;
  }>({ image: null, mask: null, prevMask: null, doubt: null, fadeStart: 0 });
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  // The cross-fade redraws itself every frame; it calls the latest draw() through this ref.
  const drawRef = useRef<() => void>(() => undefined);

  const { phase, slice, opacity, showDoubt, playing, frame, zoom, pan, set } = useViewer();
  const hasLayer = summary.layers.includes(layer);
  const cine = playing && summary.n_frames > 0;
  const imageUrl = cine
    ? dataSource.cineUrl(summary.id, slice, frame)
    : dataSource.sliceUrl(summary.id, phase, slice, "image");
  const maskUrl = !cine && hasLayer ? dataSource.sliceUrl(summary.id, phase, slice, layer) : null;
  const doubtUrl =
    !cine && showDoubt && summary.layers.includes("doubt")
      ? dataSource.sliceUrl(summary.id, phase, slice, "doubt")
      : null;

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    const { image, mask, prevMask, doubt, fadeStart } = layers.current;
    if (!canvas || !container || !image) return;
    const dpr = window.devicePixelRatio || 1;
    const width = container.clientWidth;
    const height = container.clientHeight;
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const scale = Math.min(width / image.width, height / image.height) * zoom * dpr;
    const x0 = (canvas.width - image.width * scale) / 2 + pan.x * dpr;
    const y0 = (canvas.height - image.height * scale) / 2 + pan.y * dpr;
    ctx.setTransform(scale, 0, 0, scale, x0, y0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(image, 0, 0);
    ctx.imageSmoothingEnabled = false;
    const t = Math.min((performance.now() - fadeStart) / FADE_MS, 1);
    if (prevMask && t < 1) {
      ctx.globalAlpha = opacity * (1 - t);
      ctx.drawImage(prevMask, 0, 0);
    }
    if (mask) {
      ctx.globalAlpha = opacity * (prevMask ? t : 1);
      ctx.drawImage(mask, 0, 0);
    }
    if (doubt) {
      ctx.globalAlpha = 0.9;
      ctx.drawImage(doubt, 0, 0);
    }
    ctx.globalAlpha = 1;
    if (prevMask && t < 1) requestAnimationFrame(() => drawRef.current());
  }, [opacity, zoom, pan]);

  useEffect(() => {
    drawRef.current = draw;
    draw();
  }, [draw]);

  // Load the current slice's layers, then draw. A model switch keeps the old mask for a cross-fade.
  useEffect(() => {
    let cancelled = false;
    Promise.all([
      getLayer(imageUrl, "image"),
      maskUrl ? getLayer(maskUrl, "labels") : Promise.resolve(null),
      doubtUrl ? getLayer(doubtUrl, "doubt") : Promise.resolve(null),
    ])
      .then(([image, mask, doubt]) => {
        if (cancelled) return;
        const current = layers.current;
        const switched = mask !== null && current.mask !== null && current.mask !== mask;
        current.prevMask = switched ? current.mask : null;
        current.fadeStart = performance.now();
        current.image = image;
        current.mask = mask;
        current.doubt = doubt;
        setError(null);
        setReady(true);
        requestAnimationFrame(draw);
      })
      .catch((e: Error) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [imageUrl, maskUrl, doubtUrl, draw]);

  // Preload every slice of both phases for this layer, so scrolling never stutters.
  useEffect(() => {
    const urls = [];
    for (const p of ["ed", "es"] as const) {
      for (let k = 0; k < summary.shape[2]!; k++) {
        urls.push({ url: dataSource.sliceUrl(summary.id, p, k, "image"), kind: "image" as const });
        if (hasLayer) {
          urls.push({ url: dataSource.sliceUrl(summary.id, p, k, layer), kind: "labels" as const });
        }
      }
    }
    preload(urls);
  }, [summary, layer, hasLayer]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(() => draw());
    observer.observe(container);
    return () => observer.disconnect();
  }, [draw]);

  // Wheel scrolls slices (MASTER_PLAN 9.3); with Ctrl or Cmd it zooms, like most image viewers.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const state = useViewer.getState();
      if (event.ctrlKey || event.metaKey) {
        const next = Math.min(8, Math.max(1, state.zoom * (event.deltaY < 0 ? 1.1 : 1 / 1.1)));
        state.set({ zoom: next, pan: next === 1 ? { x: 0, y: 0 } : state.pan });
      } else {
        state.step(event.deltaY > 0 ? 1 : -1);
      }
    };
    container.addEventListener("wheel", onWheel, { passive: false });
    return () => container.removeEventListener("wheel", onWheel);
  }, []);

  const drag = useRef<{ x: number; y: number } | null>(null);

  return (
    <div
      ref={containerRef}
      className="relative h-full min-h-[280px] w-full cursor-grab touch-none overflow-hidden rounded-[var(--radius-card)] bg-viewer-bg active:cursor-grabbing"
      onPointerDown={(e) => {
        drag.current = { x: e.clientX, y: e.clientY };
        e.currentTarget.setPointerCapture(e.pointerId);
      }}
      onPointerMove={(e) => {
        if (!drag.current || zoom === 1) return;
        const dx = e.clientX - drag.current.x;
        const dy = e.clientY - drag.current.y;
        drag.current = { x: e.clientX, y: e.clientY };
        set({ pan: { x: pan.x + dx, y: pan.y + dy } });
      }}
      onPointerUp={() => (drag.current = null)}
      onDoubleClick={() => set({ zoom: 1, pan: { x: 0, y: 0 } })}
      role="img"
      aria-label={`${phase === "ed" ? "End-diastole" : "End-systole"}, slice ${slice + 1} of ${summary.shape[2]}, ${MODEL_LABELS[layer]} outline`}
    >
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
      {!ready && !error && <div className="skeleton absolute inset-6 opacity-20" />}
      <div className="pointer-events-none absolute left-2 top-2 flex flex-col gap-0.5 rounded-md bg-viewer-bg/75 px-2 py-1 text-xs text-viewer-text">
        {caption && <span className="font-medium">{caption}</span>}
        <span className="tabular text-viewer-muted">
          {cine
            ? `cine frame ${frame + 1}/${summary.n_frames}`
            : `${phase.toUpperCase()} · slice ${slice + 1}/${summary.shape[2]}`}
        </span>
      </div>
      {!hasLayer && !cine && (
        <div className="pointer-events-none absolute bottom-3 left-3 rounded-md bg-viewer-surface/90 px-2 py-1 text-xs text-viewer-muted">
          {MODEL_LABELS[layer]} outlines exist for sample patients only
        </div>
      )}
      {error && (
        <div className="absolute inset-0 grid place-items-center p-6 text-center text-sm text-viewer-text">
          <div>
            <p>Couldn't load this slice.</p>
            <p className="mt-1 text-viewer-muted">{error}</p>
          </div>
        </div>
      )}
    </div>
  );
}
