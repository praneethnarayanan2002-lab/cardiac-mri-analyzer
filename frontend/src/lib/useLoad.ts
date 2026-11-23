import { useCallback, useEffect, useState } from "react";

/** Loads data for a key (a case id, say). `data` and `error` belong to the latest key only, so a
 *  stale answer for a previous case is never shown. State is set only in promise callbacks. */
export function useLoad<T>(
  load: () => Promise<T>,
  key: string,
): { data: T | null; error: string | null; retry: () => void } {
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState<{ key: string; data?: T; error?: string } | null>(null);
  const tag = `${key}#${attempt}`;

  useEffect(() => {
    let current = true;
    load().then(
      (data) => current && setState({ key: tag, data }),
      (e: Error) => current && setState({ key: tag, error: e.message }),
    );
    return () => {
      current = false;
    };
    // `load` is expected to change with `key`; depending on the tag avoids reloading on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tag]);

  const retry = useCallback(() => setAttempt((a) => a + 1), []);
  const fresh = state?.key === tag;
  return {
    data: fresh ? (state.data ?? null) : null,
    error: fresh ? (state.error ?? null) : null,
    retry,
  };
}
