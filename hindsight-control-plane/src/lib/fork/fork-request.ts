// fork: cost-accounting
// Request helpers shared by the fork's cost and quota clients: the GET that maps 404 to null, and the
// error type that keeps the HTTP status and the `error` text the control-plane proxy
// (llm-cost-proxy.ts) returns, which a bare status message would drop. Also imported by that
// server-side proxy for the load-error constants, so it must stay free of client-only modules.

import { withBasePath } from "@/lib/base-path";

// The proxy's own text when the dataplane gave no reason (unreachable, a non-JSON body, a detail that
// is not a string). It says nothing the panels' localized load-error message does not, so
// `errorDetail` drops it. Kept here, not in the server-only proxy module, so client code can compare.
export const COST_LOAD_ERROR = "Failed to get LLM cost data";
export const QUOTA_LOAD_ERROR = "Failed to get LLM quota data";

/** A non-404 failure of a fork endpoint, with the `error` text the control-plane proxy returned. */
class ForkRequestError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string | null,
    what: "quota" | "cost"
  ) {
    super(`${what} request failed with status ${status}${detail ? `: ${detail}` : ""}`);
  }
}

/**
 * The proxy answers `{ error: string }` on failure. Its generic load-error text, and anything not of
 * that shape, yields no detail.
 */
async function errorDetail(response: Response): Promise<string | null> {
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

/** The proxy's error text carried by a failed fork request, or null for any other error. */
export function errorDetailOf(error: unknown): string | null {
  return error instanceof ForkRequestError ? error.detail : null;
}

/**
 * GETs a fork endpoint through the control-plane proxy. Returns null when the endpoint is absent
 * (404), which means the feature is off; throws ForkRequestError on any other failure.
 */
export async function fetchJsonOrNull<T>(path: string, what: "quota" | "cost"): Promise<T | null> {
  const response = await fetch(withBasePath(path), { method: "GET" });
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new ForkRequestError(response.status, await errorDetail(response), what);
  }
  return (await response.json()) as T;
}
