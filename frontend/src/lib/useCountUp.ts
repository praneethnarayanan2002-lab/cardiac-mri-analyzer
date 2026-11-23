import { useEffect, useState } from "react";

/** Animates a number from 0 to `target` once, on first show (MASTER_PLAN 9.4). Skipped when the
 *  user prefers reduced motion. */
export function useCountUp(target: number, ms = 450): number {
  const reduce =
    typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const [value, setValue] = useState(reduce ? target : 0);
  useEffect(() => {
    if (reduce) {
      setValue(target);
      return;
    }
    const start = performance.now();
    let id = requestAnimationFrame(function tick(now) {
      const t = Math.min((now - start) / ms, 1);
      setValue(target * (1 - (1 - t) ** 3));
      if (t < 1) id = requestAnimationFrame(tick);
    });
    return () => cancelAnimationFrame(id);
  }, [target, ms, reduce]);
  return value;
}
