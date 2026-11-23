"""Save docs/images/slices.gif: one sample patient scrolled from base to apex, pretrained U-Net
outline next to the expert's, the way the app's viewer shows them.

Usage:
    uv run python scripts/make_readme_gif.py

Reads the exported demo files (make demo), so the GIF shows exactly what the public demo serves.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEMO = Path("frontend/public/demo")
PATIENT = "patient140"
PHASE = "ed"
SCALE = 2
ALPHA = 0.45
COLOURS = {1: (59, 130, 246), 2: (34, 197, 94), 3: (239, 68, 68)}  # RV, MYO, LV
BACKGROUND = (11, 17, 23)  # the viewer's dark surround
TEXT = (230, 237, 243)


def overlay(k: int, layer: str) -> Image.Image:
    folder = DEMO / "slices" / PATIENT / PHASE
    grey = np.asarray(Image.open(folder / "image" / f"{k}.png"), dtype=np.float32)
    labels = np.asarray(Image.open(folder / layer / f"{k}.png"))
    rgb = np.repeat(grey[:, :, None], 3, axis=2)
    for value, colour in COLOURS.items():
        mask = labels == value
        rgb[mask] = (1 - ALPHA) * rgb[mask] + ALPHA * np.array(colour, dtype=np.float32)
    image = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    return image.resize((image.width * SCALE, image.height * SCALE), Image.Resampling.NEAREST)


def main() -> None:
    n = len(list((DEMO / "slices" / PATIENT / PHASE / "image").glob("*.png")))
    font = ImageFont.load_default(size=18)
    frames = []
    for k in range(n):
        left, right = overlay(k, "unet_pretrained"), overlay(k, "expert")
        pad, top = 16, 40
        frame = Image.new("RGB", (left.width * 2 + pad * 3, left.height + top + pad), BACKGROUND)
        frame.paste(left, (pad, top))
        frame.paste(right, (left.width + pad * 2, top))
        draw = ImageDraw.Draw(frame)
        draw.text((pad, 10), "Pretrained U-Net", fill=TEXT, font=font)
        draw.text((left.width + pad * 2, 10), "Expert", fill=TEXT, font=font)
        label = f"slice {k + 1}/{n}, base to apex"
        width = draw.textlength(label, font=font)
        draw.text((frame.width - pad - width, 10), label, fill=(139, 155, 171), font=font)
        # 128 colours are plenty for grey plus three outline colours and keep the file small.
        frames.append(frame.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    # Base to apex and back, so the loop has no jump.
    sequence = frames + frames[-2:0:-1]
    out = Path("docs/images/slices.gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    sequence[0].save(
        out, save_all=True, append_images=sequence[1:], duration=350, loop=0, optimize=True
    )
    print(f"saved {out} ({out.stat().st_size / 1e3:.0f} kB, {len(sequence)} frames)")


if __name__ == "__main__":
    main()
