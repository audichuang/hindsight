// fork: cost-accounting
// Client side of the fork-only cost endpoints. Mirrors the response models in
// hindsight_api/engine/llm_cost.py. Endpoints are hidden from OpenAPI, so these
// types are maintained by hand.

import { fetchJsonOrNull } from "@/lib/fork/fork-request";

export type CostPeriod = "1d" | "7d" | "30d";
export type BillingKind = "metered" | "plan" | "local";
export type UnpricedReason =
  | "subscription_plan"
  | "local"
  | "unmapped_provider"
  | "unknown_model"
  | "no_usage"
  | "invalid_usage";
export type CostStatus = "actual" | "estimated" | "included" | "unknown";
export type CostSource = "provider" | "genai-prices" | "custom" | "none";

export interface CostStatusCounts {
  actual: number;
  estimated: number;
  included: number;
  unknown: number;
}

export interface ModelCostTotals {
  provider: string | null;
  model: string | null;
  billing: BillingKind;
  calls: number;
  priced_calls: number;
  unpriced_calls: number;
  unpriced_reason: UnpricedReason | null;
  input_tokens: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
  output_tokens: number;
  thoughts_tokens: number;
  rows_without_breakdown: number;
  legacy_rows_repaired: number;
  tokens_estimated_rows: number;
  cost_usd: number | null;
  list_price_usd: number | null;
  provider_cost_usd: number | null;
  provider_cost_calls: number;
  status_counts: CostStatusCounts;
}

export interface DayCostTotals {
  day: string;
  cost_usd: number;
  plan_list_price_usd: number;
}

export interface LLMCostSummaryResponse {
  bank_id: string;
  period: string | null;
  start: string;
  end: string | null;
  currency: "USD";
  pricing_source: string;
  truncated: boolean;
  tz: string;
  models: ModelCostTotals[];
  by_day: DayCostTotals[];
  cost_usd: number;
  plan_list_price_usd: number;
  unpriced_calls: number;
  status_counts: CostStatusCounts;
}

export interface RowCost {
  billing: BillingKind;
  cost_usd: number | null;
  list_price_usd: number | null;
  reason: UnpricedReason | null;
  priced_as: string | null;
  cost_status: CostStatus;
  cost_source: CostSource;
  provider_cost_usd: number | null;
  legacy_input_repaired: boolean;
  has_cache_write_breakdown: boolean;
  tokens_estimated: boolean;
}

export interface LLMRowCostResponse {
  bank_id: string;
  items: Record<string, RowCost>;
}

export function fetchLLMCostSummary(
  bankId: string,
  options: { period: CostPeriod; tz: string }
): Promise<LLMCostSummaryResponse | null> {
  const params = new URLSearchParams({ period: options.period, tz: options.tz });
  return fetchJsonOrNull<LLMCostSummaryResponse>(
    `/api/banks/${encodeURIComponent(bankId)}/llm-requests/cost?${params.toString()}`,
    "cost"
  );
}

export function fetchLLMRowCosts(
  bankId: string,
  ids: string[]
): Promise<LLMRowCostResponse | null> {
  const params = new URLSearchParams({ ids: ids.join(",") });
  return fetchJsonOrNull<LLMRowCostResponse>(
    `/api/banks/${encodeURIComponent(bankId)}/llm-requests/cost/rows?${params.toString()}`,
    "cost"
  );
}

/**
 * The browser's IANA time zone, which sets the day buckets in the cost summary.
 * Falls back to UTC when the runtime cannot report one.
 */
export function browserTimeZone(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
  } catch {
    return "UTC";
  }
}

/**
 * True when some metered calls were provider-reported and some were not:
 * 0 < actual < calls. Plan and local rows never count as partly reported.
 */
export function isPartlyProviderReported(
  billing: BillingKind,
  calls: number,
  counts: CostStatusCounts
): boolean {
  return billing === "metered" && counts.actual > 0 && counts.actual < calls;
}

/**
 * Whether the control plane shows the cost panel, from its own `HINDSIGHT_CP_LLM_COST_ENABLED`.
 * It must agree with the API's `HINDSIGHT_API_LLM_COST_ENABLED`. Read by `/api/version` and sent in
 * the features payload the page already loads, so a disabled deployment makes no cost request.
 * Only true, 1 and yes (any case, surrounding space ignored) turn it on.
 */
export function llmCostPanelEnabled(raw: string | undefined): boolean {
  return ["true", "1", "yes"].includes((raw ?? "").trim().toLowerCase());
}

/**
 * USD for display. null shows an em dash, dust shows "<$0.0001", and everything
 * else gets 4 decimals below $1 and 2 above.
 */
export function formatUsd(n: number | null): string {
  if (n === null) {
    return "—";
  }
  if (n > 0 && n < 0.0001) {
    return "<$0.0001";
  }
  if (Math.abs(n) < 1) {
    return `$${n.toFixed(4)}`;
  }
  return `$${n.toFixed(2)}`;
}
