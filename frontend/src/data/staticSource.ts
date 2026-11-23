import type { DataSource } from "./dataSource";
import { getJson } from "./http";
import type { CaseSummary, MetricsSummary, SampleCase } from "./types";

// Files written by scripts/export_demo.py into public/demo/ (docs/flow.md, demo export).
const root = `${import.meta.env.BASE_URL}demo`;

/** The public demo: precomputed files, no server. Uploads are not possible. */
export const staticSource: DataSource = {
  canUpload: false,
  listSamples: () => getJson<SampleCase[]>(`${root}/samples.json`),
  getCase: (id) => getJson<CaseSummary>(`${root}/cases/${id}.json`),
  sliceUrl: (id, phase, k, layer) => `${root}/slices/${id}/${phase}/${layer}/${k}.png`,
  cineUrl: (id, k, frame) => `${root}/cine/${id}/${k}/${frame}.png`,
  getMetricsSummary: () => getJson<MetricsSummary>(`${root}/metrics.json`),
  reportUrl: (id) => `${root}/reports/${id}.pdf`,
  analyze: () => Promise.reject(new Error("Uploads need the full app; see the README.")),
};
