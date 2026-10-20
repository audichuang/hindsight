// fork: cost-accounting
// Server side of the fork cost and quota proxies. Forwards one GET to a dataplane endpoint and maps
// failures to endpoint-specific messages. A dataplane `detail` string is passed through; otherwise the
// message is an English generic one, which the clients drop (`errorDetail` in fork-request.ts) since
// the cost and quota panels already show their own localized load-error copy. Those generic texts
// live in fork-request.ts, so client code and tests import them from there.

import { NextResponse } from "next/server";
import { COST_LOAD_ERROR, QUOTA_LOAD_ERROR } from "@/lib/fork/fork-request";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

export type LLMForkEndpoint =
  | "/llm-requests/cost"
  | "/llm-requests/cost/rows"
  | "/llm-requests/quota";

function loadErrorFor(endpoint: LLMForkEndpoint): string {
  return endpoint === "/llm-requests/quota" ? QUOTA_LOAD_ERROR : COST_LOAD_ERROR;
}

/** Forwards `GET /llm-requests/cost[/rows]` or `GET /llm-requests/quota` for one bank, passing the query string through unchanged. */
export async function proxyLLMCost(
  request: Request,
  bankId: string | undefined,
  endpoint: LLMForkEndpoint
): Promise<NextResponse> {
  if (!bankId) {
    return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
  }
  const loadError = loadErrorFor(endpoint);
  try {
    const query = new URL(request.url).searchParams.toString();
    const response = await fetch(
      dataplaneBankUrl(bankId, `${endpoint}${query ? `?${query}` : ""}`),
      {
        method: "GET",
        headers: getDataplaneHeaders(),
      }
    );
    const body: unknown = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = (body as { detail?: unknown } | null)?.detail;
      return NextResponse.json(
        { error: typeof detail === "string" && detail ? detail : loadError },
        { status: response.status }
      );
    }
    return NextResponse.json(body, { status: 200 });
  } catch (error) {
    console.error(`Error proxying ${endpoint}:`, error);
    return NextResponse.json({ error: loadError }, { status: 500 });
  }
}
