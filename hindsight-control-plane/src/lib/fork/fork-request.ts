// fork: cost-accounting
// Error type shared by the fork's cost and quota clients: keeps the HTTP status and the `error` text
// the control-plane proxy (llm-cost-proxy.ts) returns, which a bare status message would drop.

// The proxy's own text when the dataplane gave no reason (unreachable, a non-JSON body, a detail that
// is not a string). It says nothing the panels' localized load-error message does not, so
// `errorDetail` drops it. Kept here, not in the server-only proxy module, so client code can compare.
export const COST_LOAD_ERROR = "Failed to get LLM cost data";
export const QUOTA_LOAD_ERROR = "Failed to get LLM quota data";

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

/**
 * The proxy answers `{ error: string }` on failure. Its generic load-error text, and anything not of
 * that shape, yields no detail.
 */
export async function errorDetail(response: Response): Promise<string | null> {
  try {
    const body: unknown = await response.json();
    if (body && typeof body === "object" && "error" in body && typeof body.error === "string") {
      const text = body.error;
      return text && text !== COST_LOAD_ERROR && text !== QUOTA_LOAD_ERROR ? text : null;
    }
  } catch {
    // Not JSON: the status alone is all there is.
  }
  return null;
}
