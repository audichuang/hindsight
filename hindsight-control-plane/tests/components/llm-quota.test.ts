// fork: cost-accounting
import { afterEach, describe, expect, it, vi } from "vitest";
import { clampPercent, fetchLLMQuota, formatResetIn } from "@/lib/fork/llm-quota";

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
    expect(formatResetIn("2026-10-15T13:20:00Z", now)).toBe("in 3h 20m");
  });

  it("formats minutes only under an hour", () => {
    expect(formatResetIn("2026-10-15T10:45:00Z", now)).toBe("in 45m");
  });

  it("formats days and hours", () => {
    expect(formatResetIn("2026-10-17T14:00:00Z", now)).toBe("in 2d 4h");
  });

  it("shows under a minute when less than a minute is left", () => {
    expect(formatResetIn("2026-10-15T10:00:30Z", now)).toBe("in <1m");
  });

  it("returns an empty string for null, past, or unparseable times", () => {
    expect(formatResetIn(null, now)).toBe("");
    expect(formatResetIn("2026-10-15T09:00:00Z", now)).toBe("");
    expect(formatResetIn("2026-10-15T10:00:00Z", now)).toBe("");
    expect(formatResetIn("not-a-date", now)).toBe("");
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
