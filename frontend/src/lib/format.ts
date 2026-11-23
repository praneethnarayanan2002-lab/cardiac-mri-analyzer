import type { Group, MaskLayer } from "../data/types";

/** "56.3 %", "2.9" (no unit, no trailing space), "–" for missing. */
export function formatValue(value: number | null, unit: string, digits = 1): string {
  if (value === null || Number.isNaN(value)) return "–";
  return `${value.toFixed(digits)}${unit ? ` ${unit}` : ""}`;
}

export function percent(p: number, digits = 0): string {
  return `${(p * 100).toFixed(digits)}%`;
}

export const GROUP_NAMES: Record<Group, string> = {
  NOR: "Normal",
  MINF: "Previous heart attack",
  DCM: "Dilated cardiomyopathy",
  HCM: "Hypertrophic cardiomyopathy",
  RV: "Abnormal right ventricle",
};

export const GROUP_ORDER: Group[] = ["NOR", "MINF", "DCM", "HCM", "RV"];

export const MODEL_LABELS: Record<MaskLayer, string> = {
  unet: "U-Net",
  unet_pretrained: "Pretrained U-Net",
  medsam: "MedSAM",
  expert: "Expert",
};

/** Keyboard order 1-4 (MASTER_PLAN 9.4). */
export const MODEL_ORDER: MaskLayer[] = ["unet", "unet_pretrained", "medsam", "expert"];
