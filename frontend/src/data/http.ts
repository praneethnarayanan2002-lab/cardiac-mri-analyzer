/** fetch() + JSON with the server's error message, not just a status code, when it fails. */
export async function getJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d) => d.msg).join("; ");
    } catch {
      // not JSON: keep the status line
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}
