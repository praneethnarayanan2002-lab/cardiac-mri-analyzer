import { apiSource } from "./apiSource";
import { staticSource } from "./staticSource";
import type { CaseSummary, Layer, MetricsSummary, Phase, SampleCase, UploadInput } from "./types";

/** Every page talks to this interface, never to fetch() directly (MASTER_PLAN 9.6). The full app
 *  uses the FastAPI backend; the public demo reads files exported by scripts/export_demo.py. */
export interface DataSource {
  listSamples(): Promise<SampleCase[]>;
  getCase(id: string): Promise<CaseSummary>;
  sliceUrl(id: string, phase: Phase, k: number, layer: Layer): string;
  cineUrl(id: string, k: number, frame: number): string;
  getMetricsSummary(): Promise<MetricsSummary>;
  reportUrl(id: string): string;
  canUpload: boolean;
  analyze(input: UploadInput): Promise<string>;
}

export const dataSource: DataSource =
  import.meta.env.VITE_DATA_MODE === "static" ? staticSource : apiSource;
