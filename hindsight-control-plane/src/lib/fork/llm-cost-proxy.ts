// fork: cost-accounting
// Server side of the fork cost proxies. Forwards one GET to a dataplane endpoint and maps
// failures to a message. The message is English only: the cost panel
// shows its own localized copy, so no i18n catalog key is needed (see src/components/fork/).

import { NextResponse } from "next/server";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

export const COST_LOAD_ERROR = "Failed to get LLM cost data";

export type LLMForkEndpoint = "/llm-requests/cost" | "/llm-requests/cost/rows";

/** Forwards `GET /llm-requests/cost[/rows]` for one bank, passing the query string through unchanged. */
export async function proxyLLMCost(
  request: Request,
  bankId: string | undefined,
  endpoint: LLMForkEndpoint
): Promise<NextResponse> {
  if (!bankId) {
    return NextResponse.json({ error: "bank_id is required" }, { status: 400 });
  }
  const loadError = COST_LOAD_ERROR;
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
