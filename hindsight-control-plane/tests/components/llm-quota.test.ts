// fork: cost-accounting
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  clampPercent,
  fetchLLMQuota,
  formatBalance,
  formatResetIn,
  refreshHitServerCache,
  type LLMQuotaResponse,
  type ProviderQuota,
} from "@/lib/fork/llm-quota";
import { QUOTA_LOAD_ERROR } from "@/lib/fork/fork-request";

const EN_RESET = {
  daysHours: "in {d}d {h}h",
  hoursMinutes: "in {h}h {m}m",
  minutes: "in {m}m",
  underMinute: "in <1m",
};
const ZH_RESET = {
  daysHours: "{d} 天 {h} 小時後",
  hoursMinutes: "{h} 小時 {m} 分後",
  minutes: "{m} 分鐘後",
  underMinute: "不到 1 分鐘",
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("clampPercent", () => {
  it("keeps values inside 0-100", () => {
    expect(clampPercent(42.5)).toBe(42.5);
    expect(clampPercent(0)).toBe(0);
    expect(clampPercent(100)).toBe(100);
  });

  it("clamps out-of-range values", () => {
    expect(clampPercent(-3)).toBe(0);
    expect(clampPercent(140)).toBe(100);
  });

  it("keeps null and non-finite values as null", () => {
    expect(clampPercent(null)).toBeNull();
    expect(clampPercent(Number.NaN)).toBeNull();
    expect(clampPercent(Number.POSITIVE_INFINITY)).toBeNull();
  });
});

describe("formatResetIn", () => {
  const now = new Date("2026-10-15T10:00:00Z");

  it("formats hours and minutes", () => {
    expect(formatResetIn("2026-10-15T13:20:00Z", now, EN_RESET)).toBe("in 3h 20m");
  });

  it("formats minutes only under an hour", () => {
    expect(formatResetIn("2026-10-15T10:45:00Z", now, EN_RESET)).toBe("in 45m");
  });

  it("formats days and hours", () => {
    expect(formatResetIn("2026-10-17T14:00:00Z", now, EN_RESET)).toBe("in 2d 4h");
  });

  it("shows under a minute when less than a minute is left", () => {
    expect(formatResetIn("2026-10-15T10:00:30Z", now, EN_RESET)).toBe("in <1m");
  });

  it("returns an empty string for null, past, or unparseable times", () => {
    expect(formatResetIn(null, now, EN_RESET)).toBe("");
    expect(formatResetIn("2026-10-15T09:00:00Z", now, EN_RESET)).toBe("");
    expect(formatResetIn("2026-10-15T10:00:00Z", now, EN_RESET)).toBe("");
    expect(formatResetIn("not-a-date", now, EN_RESET)).toBe("");
  });
});

describe("formatResetIn in another language", () => {
  it("uses the caller's templates", () => {
    const now = new Date("2026-10-15T10:00:00Z");
    expect(formatResetIn("2026-10-20T09:00:00Z", now, ZH_RESET)).toBe("4 天 23 小時後");
    expect(formatResetIn("2026-10-15T13:20:00Z", now, ZH_RESET)).toBe("3 小時 20 分後");
  });
});

describe("formatBalance", () => {
  it("rounds to two decimals and groups digits", () => {
    expect(formatBalance(62199.48052275, "en")).toBe("62,199.48");
    expect(formatBalance(5, "en")).toBe("5");
  });
});

describe("fetchLLMQuota", () => {
  it("returns null when the endpoint is absent", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("{}", { status: 404 }))
    );
    await expect(fetchLLMQuota("bank-a")).resolves.toBeNull();
  });

  it("requests the bank quota path", async () => {
    const spy = vi.fn(
      async (..._args: [RequestInfo | URL, RequestInit?]) =>
        new Response(JSON.stringify({ bank_id: "bank a", scope: "server", providers: [] }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
    );
    vi.stubGlobal("fetch", spy);
    const data = await fetchLLMQuota("bank a");
    expect(data?.providers).toEqual([]);
    expect(String(spy.mock.calls[0][0])).toContain("/api/banks/bank%20a/llm-requests/quota");
  });

  it("throws on other non-2xx responses", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("{}", { status: 500 }))
    );
    await expect(fetchLLMQuota("bank-a")).rejects.toThrow("status 500");
  });
});

describe("fetchLLMQuota errors", () => {
  it("keeps the proxy's error text", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () => new Response(JSON.stringify({ error: "bank not readable" }), { status: 403 })
      )
    );
    await expect(fetchLLMQuota("bank")).rejects.toMatchObject({
      status: 403,
      detail: "bank not readable",
    });
  });
});

describe("fetchLLMQuota generic proxy error", () => {
  it("drops the proxy's generic load-error text, which repeats the card's own message", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ error: QUOTA_LOAD_ERROR }), { status: 500 }))
    );
    await expect(fetchLLMQuota("bank")).rejects.toMatchObject({ status: 500, detail: null });
  });
});

function quota(...providers: Partial<ProviderQuota>[]): LLMQuotaResponse {
  return {
    bank_id: "bank",
    scope: "server",
    providers: providers.map((p) => ({
      provider: "openai-codex",
      lanes: ["default"],
      status: "ok",
      reason: null,
      detail: null,
      plan: null,
      windows: [],
      credits: null,
      budget: null,
      fetched_at: null,
      stale: false,
      ...p,
    })),
  };
}

describe("refreshHitServerCache", () => {
  const t1 = "2026-10-20T01:00:00Z";
  const t2 = "2026-10-20T01:01:30Z";

  it("is true when a refresh returns the reading already shown", () => {
    expect(refreshHitServerCache(t1, quota({ fetched_at: t1 }))).toBe(true);
  });

  it("is false when the reading is newer", () => {
    expect(refreshHitServerCache(t1, quota({ fetched_at: t2 }))).toBe(false);
  });

  it("is false with nothing to compare (first load or another bank)", () => {
    expect(refreshHitServerCache(null, quota({ fetched_at: t1 }))).toBe(false);
  });

  it("is false while a provider that never read ok repeats its error during backoff", () => {
    expect(
      refreshHitServerCache(t1, quota({ fetched_at: t1, status: "error", reason: "timeout" }))
    ).toBe(false);
    expect(
      refreshHitServerCache(
        t1,
        quota({ fetched_at: t1, status: "unavailable", reason: "token_scope" })
      )
    ).toBe(false);
  });

  it("ignores a provider the server never reads, which has no fetched_at", () => {
    expect(
      refreshHitServerCache(
        t1,
        quota({ fetched_at: t1 }, { status: "unsupported", reason: "no_usage_endpoint" })
      )
    ).toBe(true);
  });

  it("is false while a provider is stale: backoff can repeat the reading for 900 s", () => {
    expect(
      refreshHitServerCache(t1, quota({ fetched_at: t1, stale: true }, { fetched_at: t1 }))
    ).toBe(false);
  });

  it("is false when no provider has a reading", () => {
    expect(refreshHitServerCache("", quota({ fetched_at: null }))).toBe(false);
  });
});
