import type { ModelKey } from "../../data/types";

export const MODEL_NAMES: Record<ModelKey, string> = {
  unet_scratch: "U-Net (scratch)",
  unet_resnet34: "U-Net (ResNet-34)",
  medsam_jitter0: "MedSAM (oracle boxes)",
};
