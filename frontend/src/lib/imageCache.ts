import { colourDoubt, colourLabels } from "./colormap";

/** Decoded slices, kept so scrolling and cine playback never wait for the network twice. Each
 *  entry is a canvas ready to drawImage(): greyscale as is, labels and doubt already coloured. */
export type Kind = "image" | "labels" | "doubt";

const cache = new Map<string, Promise<HTMLCanvasElement>>();

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.decoding = "async";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`could not load ${url}`));
    img.src = url;
  });
}

async function decode(url: string, kind: Kind): Promise<HTMLCanvasElement> {
  const img = await loadImage(url);
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d", { willReadFrequently: kind !== "image" });
  if (!ctx) throw new Error("canvas 2D context unavailable");
  ctx.drawImage(img, 0, 0);
  if (kind !== "image") {
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const rgba = kind === "labels" ? colourLabels(data.data) : colourDoubt(data.data);
    data.data.set(rgba);
    ctx.putImageData(data, 0, 0);
  }
  return canvas;
}

export function getLayer(url: string, kind: Kind): Promise<HTMLCanvasElement> {
  let entry = cache.get(url);
  if (!entry) {
    entry = decode(url, kind);
    // A failed load should be retried next time, not remembered.
    entry.catch(() => cache.delete(url));
    cache.set(url, entry);
  }
  return entry;
}

/** Start loading in the background; errors surface when the slice is actually drawn. */
export function preload(urls: { url: string; kind: Kind }[]): void {
  for (const { url, kind } of urls) getLayer(url, kind).catch(() => undefined);
}
