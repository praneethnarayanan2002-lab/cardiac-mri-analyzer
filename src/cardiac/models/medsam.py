"""MedSAM, zero-shot, prompted with boxes drawn around the expert labels ("oracle boxes").

MedSAM never sees our training data. For every slice and every structure present in the expert
label, a box around that structure (padded by 5 px) is given as the prompt, and MedSAM returns one
binary mask per box. The masks are merged with priority LV > MYO > RV, because the myocardium box
also contains the LV and MedSAM may fill it in.

Because the boxes come from the expert answer, this measures the *best case* for MedSAM, not what it
would do on a new scan without help (docs/decisions.md D19).
"""

import numpy as np
import torch
from transformers import SamModel, SamProcessor

REPO = "wanglab/medsam-vit-base"
REVISION = "de8488bca37bb1d4fb190f612c516126d739ce3b"
BOX_PADDING_PX = 5
# Painted in this order, so a later structure overwrites an earlier one where masks overlap.
PAINT_ORDER = (1, 2, 3)  # RV, then MYO, then LV


def load_medsam(device: torch.device) -> tuple[SamModel, SamProcessor]:
    processor = SamProcessor.from_pretrained(REPO, revision=REVISION)
    model = SamModel.from_pretrained(REPO, revision=REVISION)
    # transformers types .to() for its own arguments only; a torch.device works at runtime.
    model.to(device)  # type: ignore[arg-type]
    model.eval()
    return model, processor


def to_rgb_uint8(slice_2d: np.ndarray) -> np.ndarray:
    """MedSAM was trained on 0-255 RGB images; MR slices are clipped to 0.5-99.5 % and stretched."""
    lo, hi = np.percentile(slice_2d, [0.5, 99.5])
    scaled = np.clip((slice_2d - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    return np.repeat(scaled.astype(np.uint8)[..., None], 3, axis=2)


def box_from_mask(mask: np.ndarray, pad: int = BOX_PADDING_PX) -> np.ndarray:
    """[x0, y0, x1, y1] around a 2D mask, padded and clipped to the image. x = column, y = row."""
    rows, cols = np.nonzero(mask)
    h, w = mask.shape
    return np.array(
        [
            max(cols.min() - pad, 0),
            max(rows.min() - pad, 0),
            min(cols.max() + pad, w - 1),
            min(rows.max() + pad, h - 1),
        ],
        dtype=np.float64,
    )


def jitter_box(
    box: np.ndarray, magnitude: int, rng: np.random.Generator, shape: tuple[int, int]
) -> np.ndarray:
    """Moves each edge independently by a random whole number of pixels in [-magnitude, magnitude].

    Moving edges independently both shifts and resizes the box, like a quickly drawn real prompt."""
    if magnitude == 0:
        return box
    moved = box + rng.integers(-magnitude, magnitude + 1, size=4)
    h, w = shape
    x0, x1 = np.clip(sorted((moved[0], moved[2])), 0, w - 1)
    y0, y1 = np.clip(sorted((moved[1], moved[3])), 0, h - 1)
    return np.array([x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)], dtype=np.float64)


@torch.no_grad()
def segment_slice(
    model: SamModel,
    processor: SamProcessor,
    slice_2d: np.ndarray,
    boxes_per_jitter: dict[int, dict[int, np.ndarray]],
    device: torch.device,
) -> dict[int, np.ndarray]:
    """One image embedding, then one decoder pass per jitter level.

    `boxes_per_jitter[m][structure]` is the box for that structure at jitter m. Returns a merged
    (H, W) uint8 label map per jitter level."""
    rgb = to_rgb_uint8(slice_2d)
    inputs = processor(rgb, return_tensors="pt")
    embedding = model.get_image_embeddings(inputs["pixel_values"].to(device))
    results = {}
    for magnitude, boxes in boxes_per_jitter.items():
        label = np.zeros(slice_2d.shape, dtype=np.uint8)
        if boxes:
            structures = [s for s in PAINT_ORDER if s in boxes]
            box_list = [[boxes[s].tolist() for s in structures]]
            prompt = processor(rgb, input_boxes=box_list, return_tensors="pt")
            out = model(
                image_embeddings=embedding,
                input_boxes=prompt["input_boxes"].to(device, torch.float32),
                multimask_output=False,
            )
            masks = processor.image_processor.post_process_masks(
                out.pred_masks.cpu(), prompt["original_sizes"], prompt["reshaped_input_sizes"]
            )[0][:, 0].numpy()
            for structure, mask in zip(structures, masks, strict=True):
                label[mask] = structure
        results[magnitude] = label
    return results


def segment_volume(
    model: SamModel,
    processor: SamProcessor,
    image: np.ndarray,
    expert_label: np.ndarray,
    device: torch.device,
    jitters: tuple[int, ...],
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    """(X, Y, Z) image + expert label -> {jitter: (X, Y, Z) MedSAM label map}.

    Slices or structures absent from the expert label get no box and stay background: the oracle
    also tells MedSAM *which* slices contain each structure."""
    volumes = {m: np.zeros(image.shape, dtype=np.uint8) for m in jitters}
    for k in range(image.shape[2]):
        plane = expert_label[:, :, k]
        clean = {s: box_from_mask(plane == s) for s in PAINT_ORDER if (plane == s).any()}
        boxes = {
            m: {s: jitter_box(b, m, rng, plane.shape) for s, b in clean.items()} for m in jitters
        }
        for m, label in segment_slice(model, processor, image[:, :, k], boxes, device).items():
            volumes[m][:, :, k] = label
    return volumes
