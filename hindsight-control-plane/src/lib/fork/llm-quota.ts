// fork: cost-accounting
// Client side of the fork subscription quota endpoint. Mirrors the response models in
// hindsight_api/engine/llm_quota.py. The endpoint is hidden from OpenAPI, so these types are
// maintained by hand.

import { withBasePath } from "@/lib/base-path";

export type QuotaStatus = "ok" | "unsupported" | "unavailable" | "error";
export type QuotaReason =
  | "no_credentials"
  | "token_scope"
  | "auth_failed"
  | "http_error"
  | "malformed_response"
  | "timeout"
  | "provider_error"
  | "not_a_plan_endpoint"
  | "no_usage_endpoint"
  | "internal_error";

export interface QuotaWindow {
  label: string;
  used_percent: number | null;
  resets_at: string | null;
  window_seconds: number | null;
  scope: "account" | "model";
}

export interface QuotaCredits {
  balance: number | null;
  unlimited: boolean;
  banked_resets: number;
}

export interface QuotaBudget {
  used: number;
  limit: number;
  currency: string;
}

export interface ProviderQuota {
  provider: string;
  lanes: string[];
  status: QuotaStatus;
  reason: QuotaReason | null;
  detail: string | null;
  plan: string | null;
  windows: QuotaWindow[];
  credits: QuotaCredits | null;
  budget: QuotaBudget | null;
  fetched_at: string | null;
  stale: boolean;
}

export interface LLMQuotaResponse {
  bank_id: string;
  scope: "server";
  providers: ProviderQuota[];
}

/** Returns null when the endpoint is absent (404), which means the quota feature is off. */
async function fetchJsonOrNull<T>(path: string): Promise<T | null> {
  const response = await fetch(withBasePath(path), { method: "GET" });
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(`quota request failed with status ${response.status}`);
  }
  return (await response.json()) as T;
}

export function fetchLLMQuota(bankId: string): Promise<LLMQuotaResponse | null> {
  return fetchJsonOrNull<LLMQuotaResponse>(
    `/api/banks/${encodeURIComponent(bankId)}/llm-requests/quota`
  );
}

/** Clamps a provider percentage to 0-100. Null and non-finite values stay null. */
export function clampPercent(n: number | null): number | null {
  if (n === null || !Number.isFinite(n)) {
    return null;
  }
  return Math.min(100, Math.max(0, n));
}

/**
 * Relative time until `resetsAtIso`, such as "in 3h 20m" or "in 2d 4h". Returns an empty string
 * for a null, unparseable or past time.
 */
export function formatResetIn(resetsAtIso: string | null, now: Date): string {
  if (resetsAtIso === null) {
    return "";
  }
  const resetMs = Date.parse(resetsAtIso);
  if (Number.isNaN(resetMs)) {
    return "";
  }
  const diffMs = resetMs - now.getTime();
  if (diffMs <= 0) {
    return "";
  }
  const totalMinutes = Math.floor(diffMs / 60_000);
  if (totalMinutes === 0) {
    return "in <1m";
  }
  const days = Math.floor(totalMinutes / 1440);
  const hours = Math.floor((totalMinutes % 1440) / 60);
  const minutes = totalMinutes % 60;
  if (days > 0) {
    return `in ${days}d ${hours}h`;
  }
  if (hours > 0) {
    return `in ${hours}h ${minutes}m`;
  }
  return `in ${minutes}m`;
}
