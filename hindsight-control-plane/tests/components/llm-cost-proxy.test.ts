import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/hindsight-client", () => ({
  dataplaneBankUrl: (bankId: string, path: string) =>
    `http://dataplane.test/v1/default/banks/${bankId}${path}`,
  getDataplaneHeaders: () => ({}),
}));

import { COST_LOAD_ERROR, QUOTA_LOAD_ERROR, proxyLLMCost } from "@/lib/fork/llm-cost-proxy";
import { GET as costGET } from "@/app/api/banks/[bankId]/llm-requests/cost/route";
import { GET as rowsGET } from "@/app/api/banks/[bankId]/llm-requests/cost/rows/route";
import { GET as quotaGET } from "@/app/api/banks/[bankId]/llm-requests/quota/route";

function stubFetch(status: number, body: unknown) {
  const spy = vi.fn(
    async (..._args: [RequestInfo | URL, RequestInit?]) =>
      new Response(JSON.stringify(body), {
        status,
        headers: { "Content-Type": "application/json" },
      })
  );
  vi.stubGlobal("fetch", spy);
  return spy;
}

function req(path: string): Request {
  return new Request(`http://localhost${path}`);
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("cost proxy messages", () => {
  it("uses a cost-specific message, not the stats one, when the dataplane fails without detail", async () => {
    stubFetch(500, {});
    const response = await proxyLLMCost(req("/x"), "bank-a", "/llm-requests/cost");
    expect(response.status).toBe(500);
    expect(await response.json()).toEqual({ error: COST_LOAD_ERROR });
    expect(COST_LOAD_ERROR).not.toMatch(/stats/i);
  });

  it("passes the dataplane detail through on a 4xx", async () => {
    stubFetch(400, { detail: "period must be one of 1d, 7d, 30d" });
    const response = await proxyLLMCost(req("/x?period=2d"), "bank-a", "/llm-requests/cost");
    expect(response.status).toBe(400);
    expect(await response.json()).toEqual({ error: "period must be one of 1d, 7d, 30d" });
  });

  it("returns the cost message when the dataplane body is not JSON at all", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("<html>bad gateway</html>", { status: 502 }))
    );
    const response = await proxyLLMCost(req("/x"), "bank-a", "/llm-requests/cost");
    expect(response.status).toBe(502);
    expect(await response.json()).toEqual({ error: COST_LOAD_ERROR });
  });

  it("maps a network failure to a 500 with the cost message", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Promise.reject(new Error("ECONNREFUSED")))
    );
    const response = await proxyLLMCost(req("/x"), "bank-a", "/llm-requests/cost");
    expect(response.status).toBe(500);
    expect(await response.json()).toEqual({ error: COST_LOAD_ERROR });
  });

  it("uses the quota message, not the cost one, when the quota endpoint fails without detail", async () => {
    stubFetch(500, {});
    const response = await proxyLLMCost(req("/x"), "bank-a", "/llm-requests/quota");
    expect(response.status).toBe(500);
    expect(await response.json()).toEqual({ error: QUOTA_LOAD_ERROR });
    expect(QUOTA_LOAD_ERROR).not.toEqual(COST_LOAD_ERROR);
  });

  it("maps a quota network failure to a 500 with the quota message", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Promise.reject(new Error("ECONNREFUSED")))
    );
    const response = await proxyLLMCost(req("/x"), "bank-a", "/llm-requests/quota");
    expect(response.status).toBe(500);
    expect(await response.json()).toEqual({ error: QUOTA_LOAD_ERROR });
  });

  it("keeps the quota 404 so the card can hide itself", async () => {
    stubFetch(404, { detail: "Not Found" });
    const response = await proxyLLMCost(req("/x"), "bank-a", "/llm-requests/quota");
    expect(response.status).toBe(404);
  });

  it("rejects a missing bank id before any request", async () => {
    const spy = stubFetch(200, {});
    const response = await proxyLLMCost(req("/x"), undefined, "/llm-requests/cost");
    expect(response.status).toBe(400);
    expect(spy).not.toHaveBeenCalled();
  });
});

describe("cost routes", () => {
  it("forward the query string to the summary endpoint", async () => {
    const spy = stubFetch(200, { models: [] });
    const response = await costGET(
      req("/api/banks/bank-a/llm-requests/cost?period=7d&operation=retain"),
      {
        params: Promise.resolve({ bankId: "bank-a" }),
      }
    );
    expect(response.status).toBe(200);
    const url = String(spy.mock.calls[0][0]);
    expect(url).toContain("/llm-requests/cost?period=7d&operation=retain");
    expect(url).not.toContain("/llm-requests/cost/rows");
  });

  it("forward the quota request to the quota endpoint only", async () => {
    const spy = stubFetch(200, { bank_id: "bank-a", scope: "server", providers: [] });
    const response = await quotaGET(req("/api/banks/bank-a/llm-requests/quota"), {
      params: Promise.resolve({ bankId: "bank-a" }),
    });
    expect(response.status).toBe(200);
    const url = String(spy.mock.calls[0][0]);
    expect(url).toContain("/llm-requests/quota");
    expect(url).not.toContain("/llm-requests/cost");
  });

  it("forward the ids to the rows endpoint", async () => {
    const spy = stubFetch(200, { items: {} });
    const response = await rowsGET(req("/api/banks/bank-a/llm-requests/cost/rows?ids=a,b"), {
      params: Promise.resolve({ bankId: "bank-a" }),
    });
    expect(response.status).toBe(200);
    expect(String(spy.mock.calls[0][0])).toContain("/llm-requests/cost/rows?ids=a%2Cb");
  });
});
