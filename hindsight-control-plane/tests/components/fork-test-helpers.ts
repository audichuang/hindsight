// fork: cost-accounting
import { vi } from "vitest";

/** Stubs the global fetch with one fixed status and JSON body, and returns the spy. */
export function stubFetch(status: number, body: unknown = {}) {
  const spy = vi.fn(async (...args: [RequestInfo | URL, RequestInit?]) => {
    void args; // the spy records the call; the response does not depend on it
    return new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", spy);
  return spy;
}
