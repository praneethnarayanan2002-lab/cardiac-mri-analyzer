// Label and doubt PNGs arrive as greyscale pixel values; these turn them into RGBA in the browser
// (docs/decisions.md D33). Same structure colours as the figures (cardiac.plots).

export type Rgb = readonly [number, number, number];

export const STRUCTURE_RGB: Record<1 | 2 | 3, Rgb> = {
  1: [59, 130, 246], // RV cavity, #3B82F6
  2: [34, 197, 94], // myocardium, #22C55E
  3: [239, 68, 68], // LV cavity, #EF4444
};

export const STRUCTURE_NAMES: Record<1 | 2 | 3, string> = {
  1: "RV cavity",
  2: "Myocardium",
  3: "LV cavity",
};

/** Label image (pixel value = label 0-3, any channel) -> RGBA, background transparent.
 *  `pixels` is the RGBA data of the decoded PNG; the red channel holds the label. */
export function colourLabels(pixels: Uint8ClampedArray): Uint8ClampedArray {
  const out = new Uint8ClampedArray(pixels.length);
  for (let i = 0; i < pixels.length; i += 4) {
    const label = pixels[i] as number;
    if (label >= 1 && label <= 3) {
      const [r, g, b] = STRUCTURE_RGB[label as 1 | 2 | 3];
      out[i] = r;
      out[i + 1] = g;
      out[i + 2] = b;
      out[i + 3] = 255;
    }
  }
  return out;
}

// One hue, light to dark (more doubt = darker, more opaque purple), as in figures/doubt_maps.png.
const DOUBT_LIGHT: Rgb = [217, 205, 243];
const DOUBT_DARK: Rgb = [59, 31, 120];
/** Doubt below this (normalised entropy 0-1) is left transparent so the scan shows through. */
export const DOUBT_SHOW_ABOVE = 0.02;
/** Doubt at or above this is drawn at full colour; the scale saturates there. */
export const DOUBT_SATURATE = 0.6;

/** Doubt image (pixel value = normalised entropy x 255) -> RGBA. */
export function colourDoubt(pixels: Uint8ClampedArray): Uint8ClampedArray {
  const out = new Uint8ClampedArray(pixels.length);
  for (let i = 0; i < pixels.length; i += 4) {
    const doubt = (pixels[i] as number) / 255;
    if (doubt < DOUBT_SHOW_ABOVE) continue;
    const t = Math.min(doubt / DOUBT_SATURATE, 1);
    for (let c = 0; c < 3; c++) {
      out[i + c] = Math.round(DOUBT_LIGHT[c]! + t * (DOUBT_DARK[c]! - DOUBT_LIGHT[c]!));
    }
    out[i + 3] = Math.round(140 + 115 * t);
  }
  return out;
}
