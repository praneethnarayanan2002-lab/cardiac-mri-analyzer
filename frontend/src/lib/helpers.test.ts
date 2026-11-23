import { describe, expect, it } from "vitest";

import type { NormalRange } from "../data/types";
import { colourDoubt, colourLabels, DOUBT_SHOW_ABOVE, STRUCTURE_RGB } from "./colormap";
import { formatValue } from "./format";
import { rangePositions, statusFor } from "./normalRanges";

/** RGBA pixels whose red channel holds the given values, like a decoded greyscale PNG. */
function pixels(...values: number[]): Uint8ClampedArray {
  return new Uint8ClampedArray(values.flatMap((v) => [v, v, v, 255]));
}

describe("colourLabels", () => {
  it("maps labels 1-3 to the structure colours and leaves background transparent", () => {
    const out = colourLabels(pixels(0, 1, 2, 3));
    expect(Array.from(out.slice(0, 4))).toEqual([0, 0, 0, 0]);
    expect(Array.from(out.slice(4, 7))).toEqual([...STRUCTURE_RGB[1]]);
    expect(Array.from(out.slice(8, 11))).toEqual([...STRUCTURE_RGB[2]]);
    expect(Array.from(out.slice(12, 15))).toEqual([...STRUCTURE_RGB[3]]);
    expect([out[7], out[11], out[15]]).toEqual([255, 255, 255]);
  });

  it("ignores values that are not labels", () => {
    expect(Array.from(colourLabels(pixels(4, 200)))).toEqual(new Array(8).fill(0));
  });
});

describe("colourDoubt", () => {
  it("is transparent where the model is sure and more opaque where it doubts more", () => {
    const below = Math.floor(DOUBT_SHOW_ABOVE * 255) - 1;
    const out = colourDoubt(pixels(below, 60, 200));
    expect(out[3]).toBe(0);
    expect(out[7]!).toBeGreaterThan(0);
    expect(out[11]!).toBeGreaterThan(out[7]!);
  });
});

describe("normal ranges", () => {
  const range: NormalRange = { low: 50, high: 70, mean: 60, sd: 5, n: 20 };

  it("follows the backend's 2 SD / 3 SD rule", () => {
    expect(statusFor(60, range)).toBe("normal");
    expect(statusFor(70, range)).toBe("normal");
    expect(statusFor(72, range)).toBe("borderline");
    expect(statusFor(80, range)).toBe("abnormal");
    expect(statusFor(null, range)).toBe("unknown");
    expect(statusFor(60, null)).toBe("unknown");
  });

  it("places the normal band and the marker on the bar", () => {
    const p = rangePositions(60, range);
    expect(p.normalStart).toBeCloseTo(1 / 3);
    expect(p.normalEnd).toBeCloseTo(2 / 3);
    expect(p.marker).toBeCloseTo(0.5);
  });

  it("widens the bar so a far-out value stays visible", () => {
    const p = rangePositions(200, range);
    expect(p.marker).toBeGreaterThan(0.9);
    expect(p.marker).toBeLessThan(1);
    expect(p.normalEnd).toBeLessThan(p.marker);
  });
});

describe("formatValue", () => {
  it("drops the space when there is no unit and marks missing values", () => {
    expect(formatValue(56.34, "%")).toBe("56.3 %");
    expect(formatValue(2.94, "")).toBe("2.9");
    expect(formatValue(null, "mL")).toBe("–");
  });
});
