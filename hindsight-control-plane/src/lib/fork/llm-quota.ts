// fork: cost-accounting
// Client side of the fork subscription quota endpoint. Mirrors the response models in
// hindsight_api/engine/llm_quota.py. The endpoint is hidden from OpenAPI, so these types are
// maintained by hand.

import { withBasePath } from "@/lib/base-path";
import { errorDetail, ForkRequestError } from "@/lib/fork/fork-request";

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
    throw new ForkRequestError(response.status, await errorDetail(response));
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

/** Templates for the reset countdown; `{d}`, `{h}` and `{m}` are replaced with numbers. */
export interface ResetInLabels {
  daysHours: string;
  hoursMinutes: string;
  minutes: string;
  underMinute: string;
}

/**
 * Relative time until `resetsAtIso` in the caller's language, such as "in 3h 20m" or "3 小時 20 分後".
 * Returns an empty string for a null, unparseable or past time.
 */
export function formatResetIn(
  resetsAtIso: string | null,
  now: Date,
  labels: ResetInLabels
): string {
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
  const d = Math.floor(totalMinutes / 1440);
  const h = Math.floor((totalMinutes % 1440) / 60);
  const m = totalMinutes % 60;
  const template =
    totalMinutes === 0
      ? labels.underMinute
      : d > 0
        ? labels.daysHours
        : h > 0
          ? labels.hoursMinutes
          : labels.minutes;
  return template.replace("{d}", String(d)).replace("{h}", String(h)).replace("{m}", String(m));
}

/** A credit balance with at most two decimals, grouped for the locale ("62,199.48"). */
export function formatBalance(balance: number, locale: string): string {
  return balance.toLocaleString(locale, { maximumFractionDigits: 2 });
}

/** The newest `fetched_at` across providers, or an empty string when none has one. */
export function newestFetchedAt(data: LLMQuotaResponse): string {
  return data.providers.reduce(
    (max, p) => (p.fetched_at && p.fetched_at > max ? p.fetched_at : max),
    ""
  );
}

/**
 * True when a manual refresh brought back the reading already on screen because the server's
 * 60 s cache answered it. `previousNewest` is null when there is nothing to compare (the first
 * load, or another bank). False while any provider is stale: during backoff the server returns
 * the same last good reading for up to 900 s, so "once a minute" would be wrong, and the stale
 * badge already explains it.
 */
export function refreshHitServerCache(
  previousNewest: string | null,
  data: LLMQuotaResponse
): boolean {
  if (previousNewest === null || data.providers.some((p) => p.stale)) {
    return false;
  }
  const newest = newestFetchedAt(data);
  return newest !== "" && newest === previousNewest;
}
