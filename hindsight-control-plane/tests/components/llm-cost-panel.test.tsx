// @vitest-environment jsdom
// fork: cost-accounting
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";

let costLlmEnabled = true;
vi.mock("next-intl", () => ({ useLocale: () => "en" }));
vi.mock("@/lib/features-context", () => ({
  useFeatures: () => ({ features: { llm_cost: costLlmEnabled } }),
}));

import { LLMCostPanel } from "@/components/fork/llm-cost-panel";

const QUOTA_BODY = {
  bank_id: "bank-a",
  scope: "server",
  providers: [
    {
      provider: "claude-code",
      plan: "max",
      status: "ok",
      reason: null,
      detail: null,
      stale: false,
      fetched_at: null,
      lanes: [],
      windows: [],
      credits: null,
      budget: null,
    },
  ],
};

/** Answers the cost route with `costStatus` and the quota route with `quotaStatus`. */
function stubRoutes(costStatus: number, quotaStatus: number) {
  const spy = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) => {
    const url = String(input);
    if (url.includes("/llm-requests/quota")) {
      return new Response(JSON.stringify(QUOTA_BODY), {
        status: quotaStatus,
        headers: { "Content-Type": "application/json" },
      });
    }
    return new Response("{}", {
      status: costStatus,
      headers: { "Content-Type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", spy);
  return spy;
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  costLlmEnabled = true;
});

describe("LLMCostPanel quota mount", () => {
  it("shows the quota card when the cost endpoint is absent but quota answers", async () => {
    stubRoutes(404, 200);
    render(<LLMCostPanel bankId="bank-a" />);
    await waitFor(() => expect(screen.getByText("Subscription quota")).toBeTruthy());
    expect(screen.getByText("claude-code")).toBeTruthy();
    expect(screen.queryByText("LLM cost")).toBeNull();
  });

  it("hides both cards when the cost endpoint and quota endpoint are both absent", async () => {
    const spy = stubRoutes(404, 404);
    const { container } = render(<LLMCostPanel bankId="bank-a" />);
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(container.textContent).toBe(""));
  });

  it("renders nothing and sends no request when features.llm_cost is false", () => {
    costLlmEnabled = false;
    const spy = stubRoutes(200, 200);
    const { container } = render(<LLMCostPanel bankId="bank-a" />);
    expect(container.innerHTML).toBe("");
    expect(spy).not.toHaveBeenCalled();
  });
});
