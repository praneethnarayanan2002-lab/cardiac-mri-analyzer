// Mirrors backend/app/schemas.py and services/metrics.py by hand. Keep them in step.

export type Phase = "ed" | "es";
export type Layer = "image" | "unet" | "unet_pretrained" | "medsam" | "expert" | "doubt";
export type MaskLayer = Exclude<Layer, "image" | "doubt">;
export type Status = "normal" | "borderline" | "abnormal" | "unknown";
export type Group = "NOR" | "MINF" | "DCM" | "HCM" | "RV";

export interface NormalRange {
  low: number;
  high: number;
  mean: number;
  sd: number;
  n: number;
}

export interface Measurement {
  key: string;
  label: string;
  unit: string;
  value: number | null;
  range: NormalRange | null;
  status: Status;
}

export interface Reason {
  feature: string;
  label: string;
  value: number;
  unit: string;
  shap: number;
  text: string;
}

export interface CaseSummary {
  id: string;
  source: "sample" | "upload";
  pid: string | null;
  true_group: Group | null;
  spacing_mm: number[];
  shape: number[]; // [X, Y, slices]
  n_frames: number;
  ed_frame: number | null;
  es_frame: number | null;
  layers: Layer[];
  height_cm: number;
  weight_kg: number;
  bsa_m2: number;
  measurements: Measurement[];
  diagnosis: { predicted: Group; probabilities: Record<Group, number> };
  why: Reason[];
  uncertainty: {
    score: number;
    per_phase: Record<Phase, number>;
    threshold: number;
    flagged: boolean;
  };
  models: Record<string, string>;
}

export interface SampleCase {
  id: string;
  pid: string;
  true_group: Group;
  predicted: Group;
  flagged: boolean;
}

export interface MeanStd {
  mean: number;
  std: number;
}

export type Structure = "RV" | "MYO" | "LV";
export type ModelKey = "unet_scratch" | "unet_resnet34" | "medsam_jitter0";

export interface BlandAltmanStats {
  bias: MeanStd;
  loa_low: MeanStd;
  loa_high: MeanStd;
  mae: MeanStd;
  pearson_r: MeanStd;
  seed0: { bias: number; loa_low: number; loa_high: number };
}

export interface MetricsSummary {
  segmentation: {
    dice: Record<ModelKey, Record<Structure, MeanStd>>;
    hd95: Record<ModelKey, Record<Structure, MeanStd>>;
    jitter: ({ px: number } & Record<Structure, number>)[];
    n_patients: number;
  };
  clinical: {
    stats: Record<ModelKey, Record<string, BlandAltmanStats>>;
    points: Record<
      ModelKey,
      Record<string, { pid: string; group: Group; model: number; expert: number }[]>
    >;
    units: Record<string, string>;
    points_seed: number;
  };
  diagnosis: {
    train: Record<string, unknown>;
    test: Record<
      string,
      Record<string, { accuracy: MeanStd; macro_f1: MeanStd; confusion: number[][] }>
    >;
  };
  uncertainty: {
    spearman_rho: MeanStd;
    aurc: MeanStd;
    curve_seed0: { coverage: number; risk: number; oracle: number }[];
    random_risk_seed0: number;
  };
}

export interface UploadInput {
  ed: File;
  es: File;
  heightCm: number;
  weightKg: number;
}
