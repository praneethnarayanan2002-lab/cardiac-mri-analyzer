import type { DataSource } from "./dataSource";
import { getJson } from "./http";
import type { CaseSummary, MetricsSummary, SampleCase } from "./types";

/** The full app: the FastAPI backend, proxied to /api by the Vite dev server. */
export const apiSource: DataSource = {
  canUpload: true,
  listSamples: () => getJson<SampleCase[]>("/api/samples"),
  getCase: (id) => getJson<CaseSummary>(`/api/cases/${encodeURIComponent(id)}`),
  sliceUrl: (id, phase, k, layer) =>
    `/api/cases/${encodeURIComponent(id)}/slice/${phase}/${k}?layer=${layer}`,
  cineUrl: (id, k, frame) => `/api/cases/${encodeURIComponent(id)}/cine/${k}?frame=${frame}`,
  getMetricsSummary: () => getJson<MetricsSummary>("/api/metrics/summary"),
  reportUrl: (id) => `/api/cases/${encodeURIComponent(id)}/report.pdf`,
  async analyze({ ed, es, heightCm, weightKg }) {
    const form = new FormData();
    form.append("ed", ed);
    form.append("es", es);
    form.append("height_cm", String(heightCm));
    form.append("weight_kg", String(weightKg));
    const body = await getJson<{ case_id: string }>("/api/analyze", { method: "POST", body: form });
    return body.case_id;
  },
};
