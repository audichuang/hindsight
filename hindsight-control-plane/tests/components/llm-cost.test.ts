import { afterEach, describe, expect, it, vi } from "vitest";
import { STRINGS } from "@/components/fork/llm-cost-panel";
import {
  browserTimeZone,
  fetchLLMCostSummary,
  fetchLLMRowCosts,
  formatUsd,
  isPartlyProviderReported,
  llmCostPanelEnabled,
} from "@/lib/fork/llm-cost";
import { COST_LOAD_ERROR } from "@/lib/fork/fork-request";

/** Stubs the global fetch with one fixed status and JSON body, and returns the spy. */
function stubFetch(status: number, body: unknown = {}) {
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

/** The requested URL as a parsed URL, so encoding is checked the way the server decodes it. */
function requestedUrl(spy: ReturnType<typeof stubFetch>): URL {
  return new URL(String(spy.mock.calls[0][0]), "http://localhost");
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("formatUsd", () => {
  it("shows an em dash for a missing cost", () => {
    expect(formatUsd(null)).toBe("—");
  });

  it("shows dust as a floor marker instead of a rounded zero", () => {
    expect(formatUsd(0.00001)).toBe("<$0.0001");
    expect(formatUsd(0.00009)).toBe("<$0.0001");
  });

  it("shows exact zero as zero", () => {
    expect(formatUsd(0)).toBe("$0.0000");
  });

  it("uses four decimals below one dollar", () => {
    expect(formatUsd(0.000401)).toBe("$0.0004");
    expect(formatUsd(0.0001)).toBe("$0.0001");
    expect(formatUsd(0.5)).toBe("$0.5000");
  });

  it("uses two decimals from one dollar up", () => {
    expect(formatUsd(1)).toBe("$1.00");
    expect(formatUsd(12.3456)).toBe("$12.35");
  });
});

/** Every key path in an object, so nested tables (reasons, billing) are compared too. */
function keyPaths(value: unknown, prefix = ""): string[] {
  if (value === null || typeof value !== "object") {
    return [prefix];
  }
  return Object.entries(value as Record<string, unknown>)
    .sort(([a], [b]) => a.localeCompare(b))
    .flatMap(([key, child]) => keyPaths(child, prefix ? `${prefix}.${key}` : key));
}

describe("cost panel STRINGS", () => {
  const reference = keyPaths(STRINGS.en);

  it("has an English table with the expected top-level shape", () => {
    expect(reference.length).toBeGreaterThan(0);
    expect(reference).toContain("title");
    expect(reference).toContain("reasons.no_usage");
    expect(reference).toContain("billing.metered");
  });

  it("gives every locale the same keys as en", () => {
    for (const [locale, table] of Object.entries(STRINGS)) {
      expect(keyPaths(table), `locale ${locale}`).toEqual(reference);
    }
  });

  it("has no empty values", () => {
    for (const [locale, table] of Object.entries(STRINGS)) {
      for (const path of keyPaths(table)) {
        const value = path
          .split(".")
          .reduce<unknown>((node, key) => (node as Record<string, unknown>)[key], table);
        expect(typeof value === "string" && value.length > 0, `${locale}.${path}`).toBe(true);
      }
    }
  });
});

describe("fetchLLMCostSummary", () => {
  it("resolves to null on 404, which means the feature is off", async () => {
    stubFetch(404, { detail: "Not Found" });
    await expect(fetchLLMCostSummary("bank", { period: "7d", tz: "UTC" })).resolves.toBeNull();
  });

  it("rejects on any other failure instead of hiding it", async () => {
    stubFetch(500, { detail: "boom" });
    await expect(fetchLLMCostSummary("bank", { period: "7d", tz: "UTC" })).rejects.toThrow(
      "status 500"
    );
  });

  it("keeps the proxy's error text so the panel can say why", async () => {
    stubFetch(400, { error: "Unknown time zone: Mars/Base" });
    await expect(fetchLLMCostSummary("bank", { period: "7d", tz: "UTC" })).rejects.toMatchObject({
      status: 400,
      detail: "Unknown time zone: Mars/Base",
    });
  });

  it("drops the proxy's generic load-error text, which repeats the panel's own message", async () => {
    stubFetch(500, { error: COST_LOAD_ERROR });
    await expect(fetchLLMCostSummary("bank", { period: "7d", tz: "UTC" })).rejects.toMatchObject({
      status: 500,
      detail: null,
    });
  });

  it("has no detail when the failure body is not the proxy's shape", async () => {
    stubFetch(502, "upstream down");
    await expect(fetchLLMCostSummary("bank", { period: "7d", tz: "UTC" })).rejects.toMatchObject({
      status: 502,
      detail: null,
    });
  });

  it("resolves to the parsed body on 200", async () => {
    const body = { bank_id: "bank", models: [], cost_usd: 0 };
    stubFetch(200, body);
    await expect(fetchLLMCostSummary("bank", { period: "7d", tz: "UTC" })).resolves.toEqual(body);
  });

  it("encodes the bank id and sends the period", async () => {
    const spy = stubFetch(200, {});
    await fetchLLMCostSummary("my bank/1", { period: "30d", tz: "UTC" });
    const url = requestedUrl(spy);
    expect(url.pathname).toBe("/api/banks/my%20bank%2F1/llm-requests/cost");
    expect(url.searchParams.get("period")).toBe("30d");
  });

  it("sends the time zone so day buckets are local to the browser", async () => {
    const spy = stubFetch(200, {});
    await fetchLLMCostSummary("bank", { period: "7d", tz: "Asia/Taipei" });
    expect(requestedUrl(spy).searchParams.get("tz")).toBe("Asia/Taipei");
  });
});

describe("browserTimeZone", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("returns the runtime's IANA zone", () => {
    const zone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    expect(browserTimeZone()).toBe(zone || "UTC");
  });

  it("falls back to UTC when Intl throws", () => {
    vi.spyOn(Intl, "DateTimeFormat").mockImplementation(() => {
      throw new Error("no intl");
    });
    expect(browserTimeZone()).toBe("UTC");
  });

  it("falls back to UTC when the runtime reports an empty zone", () => {
    vi.spyOn(Intl, "DateTimeFormat").mockImplementation(
      () => ({ resolvedOptions: () => ({ timeZone: "" }) }) as unknown as Intl.DateTimeFormat
    );
    expect(browserTimeZone()).toBe("UTC");
  });
});

describe("isPartlyProviderReported", () => {
  const counts = (actual: number, estimated = 0) => ({
    actual,
    estimated,
    included: 0,
    unknown: 0,
  });

  it("is true for metered rows with some but not all calls provider-reported", () => {
    expect(isPartlyProviderReported("metered", 3, counts(1, 2))).toBe(true);
    expect(isPartlyProviderReported("metered", 3, counts(2, 1))).toBe(true);
  });

  it("is false when no calls or all calls are provider-reported", () => {
    expect(isPartlyProviderReported("metered", 3, counts(0, 3))).toBe(false);
    expect(isPartlyProviderReported("metered", 3, counts(3))).toBe(false);
  });

  it("is false for plan and local rows", () => {
    expect(isPartlyProviderReported("plan", 3, counts(1, 2))).toBe(false);
    expect(isPartlyProviderReported("local", 3, counts(1, 2))).toBe(false);
  });
});

describe("fetchLLMRowCosts", () => {
  it("resolves to null on 404", async () => {
    stubFetch(404, {});
    await expect(fetchLLMRowCosts("bank", ["a"])).resolves.toBeNull();
  });

  it("rejects on a server error", async () => {
    stubFetch(502, {});
    await expect(fetchLLMRowCosts("bank", ["a"])).rejects.toThrow("status 502");
  });

  it("sends the ids comma-joined", async () => {
    const spy = stubFetch(200, { bank_id: "bank", items: {} });
    await fetchLLMRowCosts("bank", ["a1", "b2"]);
    const url = requestedUrl(spy);
    expect(url.pathname).toBe("/api/banks/bank/llm-requests/cost/rows");
    expect(url.searchParams.get("ids")).toBe("a1,b2");
  });
});

describe("llmCostPanelEnabled", () => {
  it.each(["true", "1", "yes", "YES", " True "])("turns the panel on for %j", (raw) => {
    expect(llmCostPanelEnabled(raw)).toBe(true);
  });

  it.each([undefined, "", "false", "0", "no", "nope"])("leaves the panel off for %j", (raw) => {
    expect(llmCostPanelEnabled(raw)).toBe(false);
  });
});

describe("plan amounts are labelled as notional", () => {
  it("says the plan value is an API list price, not billed, in every locale", () => {
    for (const [locale, strings] of Object.entries(STRINGS)) {
      expect(strings.planNote.length, locale).toBeGreaterThan(0);
      expect(strings.planValue, locale).toContain(strings.planNote);
      expect(strings.planPrefix, locale).not.toContain("plan");
    }
    expect(STRINGS.en.planNote).toBe("API list price, not billed");
  });
});
