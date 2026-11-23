import { STRUCTURE_NAMES, STRUCTURE_RGB } from "../../lib/colormap";

/** Colour key for the outlines. Names are always written out, so colour is never the only cue. */
export function Legend({ showDoubt }: { showDoubt: boolean }) {
  return (
    <ul className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted" aria-label="Legend">
      {([1, 2, 3] as const).map((k) => (
        <li key={k} className="flex items-center gap-1.5">
          <span
            className="h-3 w-3 rounded-sm"
            style={{ background: `rgb(${STRUCTURE_RGB[k].join(" ")})` }}
            aria-hidden="true"
          />
          {STRUCTURE_NAMES[k]}
        </li>
      ))}
      {showDoubt && (
        <li className="flex items-center gap-1.5">
          <span
            className="h-3 w-8 rounded-sm"
            style={{ background: "linear-gradient(90deg, #d9cdf3, #3b1f78)" }}
            aria-hidden="true"
          />
          Doubt: low → high
        </li>
      )}
    </ul>
  );
}
