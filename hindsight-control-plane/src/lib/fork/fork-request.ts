// fork: cost-accounting
// Error type shared by the fork's cost and quota clients: keeps the HTTP status and the `error` text
// the control-plane proxy (llm-cost-proxy.ts) returns, which a bare status message would drop.

/** A non-404 failure of a fork endpoint, with the `error` text the control-plane proxy returned. */
export class ForkRequestError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string | null,
    what: "quota" | "cost" = "quota"
  ) {
    super(`${what} request failed with status ${status}${detail ? `: ${detail}` : ""}`);
  }
}

/** The proxy answers `{ error: string }` on failure; anything else yields no detail. */
export async function errorDetail(response: Response): Promise<string | null> {
  try {
    const body: unknown = await response.json();
    if (body && typeof body === "object" && "error" in body && typeof body.error === "string") {
      return body.error || null;
    }
  } catch {
    // Not JSON: the status alone is all there is.
  }
  return null;
}
