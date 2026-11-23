import { create } from "zustand";

import type { MaskLayer, Phase } from "../data/types";

/** Everything about how the scan is being viewed. Shared by the viewer(s), the controls and the
 *  keyboard handler, so compare mode's two viewers stay on the same slice for free. */
export interface ViewerState {
  phase: Phase;
  slice: number;
  nSlices: number;
  layer: MaskLayer;
  compareLayer: MaskLayer;
  compare: boolean;
  opacity: number; // 0-1
  showDoubt: boolean;
  playing: boolean;
  frame: number;
  zoom: number;
  pan: { x: number; y: number };
  reset(nSlices: number, startSlice: number, layer: MaskLayer, compareLayer: MaskLayer): void;
  setSlice(k: number): void;
  step(delta: number): void;
  set(partial: Partial<ViewerState>): void;
}

export const useViewer = create<ViewerState>((set, get) => ({
  phase: "ed",
  slice: 0,
  nSlices: 1,
  layer: "unet_pretrained",
  compareLayer: "expert",
  compare: false,
  opacity: 0.45,
  showDoubt: false,
  playing: false,
  frame: 0,
  zoom: 1,
  pan: { x: 0, y: 0 },
  reset: (nSlices, startSlice, layer, compareLayer) =>
    set({
      nSlices,
      slice: startSlice,
      layer,
      compareLayer,
      phase: "ed",
      compare: false,
      showDoubt: false,
      playing: false,
      frame: 0,
      zoom: 1,
      pan: { x: 0, y: 0 },
    }),
  setSlice: (k) => set({ slice: Math.max(0, Math.min(get().nSlices - 1, k)) }),
  step: (delta) => get().setSlice(get().slice + delta),
  set: (partial) => set(partial),
}));
