// fork: cost-accounting
// Subscription quota card for the LLM requests view. Shows each plan-backed provider's usage
// windows, credits and extra-usage budget. It renders nothing when the feature is off (the
// endpoint answers 404), when no provider reports anything, or when the control plane has the
// cost feature disabled. Exported but not mounted yet; LLMCostPanel mounts it in a later step.

import { useEffect, useState } from "react";
import { useLocale } from "next-intl";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useFeatures } from "@/lib/features-context";
import { clampPercent, fetchLLMQuota, formatResetIn } from "@/lib/fork/llm-quota";
import type {
  LLMQuotaResponse,
  ProviderQuota,
  QuotaReason,
  QuotaStatus,
  QuotaWindow,
} from "@/lib/fork/llm-quota";

interface QuotaStrings {
  title: string;
  refresh: string;
  loading: string;
  loadError: string;
  plan: string;
  lanes: string;
  resets: string;
  updated: string;
  stale: string;
  staleHint: string;
  noWindows: string;
  credits: string;
  balance: string;
  unlimited: string;
  bankedResets: string;
  budget: string;
  modelScope: string;
  accountScope: string;
  detail: string;
  statuses: Record<QuotaStatus, string>;
  reasons: Record<QuotaReason, string>;
}

const EN: QuotaStrings = {
  title: "Subscription quota",
  refresh: "Refresh",
  loading: "Loading quota…",
  loadError: "Could not load quota data.",
  plan: "Plan",
  lanes: "Lanes",
  resets: "Resets",
  updated: "Updated",
  stale: "Stale",
  staleHint: "Showing the last good reading; the latest fetch failed.",
  noWindows: "No usage windows reported.",
  credits: "Credits",
  balance: "Balance",
  unlimited: "Unlimited",
  bankedResets: "Banked resets",
  budget: "Extra usage",
  modelScope: "model",
  accountScope: "account",
  detail: "Detail",
  statuses: { ok: "OK", unsupported: "Not supported", unavailable: "Unavailable", error: "Error" },
  reasons: {
    no_credentials: "No credentials configured for this provider",
    token_scope: "Token lacks the usage scope",
    auth_failed: "Authentication failed",
    http_error: "Provider returned an HTTP error",
    malformed_response: "Unexpected response from provider",
    timeout: "Provider did not respond in time",
    provider_error: "Provider reported an error",
    not_a_plan_endpoint: "Not a subscription plan endpoint",
    no_usage_endpoint: "This provider has no usage endpoint",
    internal_error: "Internal error",
  },
};

const ZH_TW: QuotaStrings = {
  title: "訂閱額度",
  refresh: "重新整理",
  loading: "載入額度中…",
  loadError: "無法載入額度資料。",
  plan: "方案",
  lanes: "通道",
  resets: "重置於",
  updated: "更新於",
  stale: "過期",
  staleHint: "顯示最近一次成功的數值；最新一次讀取失敗。",
  noWindows: "供應商未回報用量區間。",
  credits: "點數",
  balance: "餘額",
  unlimited: "無上限",
  bankedResets: "可用重置次數",
  budget: "額外用量",
  modelScope: "模型",
  accountScope: "帳號",
  detail: "詳細資訊",
  statuses: { ok: "正常", unsupported: "不支援", unavailable: "無法取得", error: "錯誤" },
  reasons: {
    no_credentials: "此供應商未設定憑證",
    token_scope: "權杖缺少用量查詢權限",
    auth_failed: "驗證失敗",
    http_error: "供應商回傳 HTTP 錯誤",
    malformed_response: "供應商回應格式異常",
    timeout: "供應商未在時限內回應",
    provider_error: "供應商回報錯誤",
    not_a_plan_endpoint: "非訂閱方案端點",
    no_usage_endpoint: "此供應商沒有用量端點",
    internal_error: "內部錯誤",
  },
};

/** Keyed by locale. Every locale must carry the same keys as `en`. */
const STRINGS: Record<string, QuotaStrings> = { en: EN, "zh-TW": ZH_TW };

function stringsFor(locale: string): QuotaStrings {
  return STRINGS[locale] ?? STRINGS.en;
}

type QuotaLoad =
  | { bankId: string; status: "ready"; data: LLMQuotaResponse }
  | { bankId: string; status: "off" | "error" };

/** Local date and time for an ISO timestamp, or an empty string when it does not parse. */
function formatLocalTime(iso: string | null, locale: string): string {
  if (iso === null) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString(locale, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function LLMQuotaCard({ bankId }: { bankId: string }) {
  const locale = useLocale();
  const s = stringsFor(locale);
  // Nothing is fetched unless the control plane reports the cost feature on.
  const enabled = useFeatures().features?.llm_cost === true;
  const [load, setLoad] = useState<QuotaLoad | null>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    fetchLLMQuota(bankId)
      .then((data) => {
        if (cancelled) return;
        setLoad(data ? { bankId, status: "ready", data } : { bankId, status: "off" });
      })
      .catch(() => {
        if (!cancelled) setLoad({ bankId, status: "error" });
      });
    return () => {
      cancelled = true;
    };
  }, [bankId, enabled, nonce]);

  const current = load?.bankId === bankId ? load : null;
  if (!enabled || current?.status === "off") return null;
  if (current?.status === "ready" && current.data.providers.length === 0) return null;

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between gap-3 flex-wrap">
        <CardTitle>{s.title}</CardTitle>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setNonce((n) => n + 1)}
          disabled={current === null}
        >
          {s.refresh}
        </Button>
      </CardHeader>
      <CardContent className="space-y-4">
        {current === null && <p className="text-sm text-muted-foreground">{s.loading}</p>}
        {current?.status === "error" && <p className="text-sm text-destructive">{s.loadError}</p>}
        {current?.status === "ready" &&
          current.data.providers.map((provider) => (
            <ProviderBlock
              key={`${provider.provider}|${provider.lanes.join(",")}`}
              provider={provider}
              locale={locale}
              s={s}
            />
          ))}
      </CardContent>
    </Card>
  );
}

function ProviderBlock({
  provider,
  locale,
  s,
}: {
  provider: ProviderQuota;
  locale: string;
  s: QuotaStrings;
}) {
  return (
    <div className="space-y-3 rounded-md border p-3">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="font-medium">{provider.provider}</span>
        {provider.plan && (
          <span className="text-muted-foreground">
            {s.plan}: <span className="font-mono text-foreground">{provider.plan}</span>
          </span>
        )}
        <span className="rounded border px-1.5 py-0.5 text-xs">{s.statuses[provider.status]}</span>
        {provider.stale && (
          <span className="rounded bg-amber-100 px-1.5 py-0.5 text-xs text-amber-900 dark:bg-amber-900/40 dark:text-amber-100">
            {s.stale}
          </span>
        )}
      </div>
      {provider.lanes.length > 0 && (
        <p className="text-xs text-muted-foreground">
          {s.lanes}: <span className="font-mono">{provider.lanes.join(", ")}</span>
        </p>
      )}
      {provider.stale && <p className="text-xs text-muted-foreground">{s.staleHint}</p>}
      {provider.reason && <p className="text-sm">{s.reasons[provider.reason]}</p>}
      {provider.detail && (
        <p className="text-xs text-muted-foreground">
          {s.detail}: <span className="font-mono">{provider.detail}</span>
        </p>
      )}
      {provider.windows.map((quotaWindow) => (
        <WindowRow
          key={`${quotaWindow.label}|${quotaWindow.scope}`}
          quotaWindow={quotaWindow}
          locale={locale}
          s={s}
        />
      ))}
      {provider.status === "ok" && provider.windows.length === 0 && (
        <p className="text-sm text-muted-foreground">{s.noWindows}</p>
      )}
      {provider.credits && (
        <p className="text-sm">
          <span className="text-muted-foreground">{s.credits}</span>{" "}
          <span className="font-mono">
            {s.balance}:{" "}
            {provider.credits.unlimited
              ? s.unlimited
              : (provider.credits.balance?.toString() ?? "—")}
          </span>
          {provider.credits.banked_resets > 0 && (
            <span className="font-mono">
              {" "}
              · {s.bankedResets}: {provider.credits.banked_resets}
            </span>
          )}
        </p>
      )}
      {provider.budget && (
        <p className="text-sm">
          <span className="text-muted-foreground">{s.budget}</span>{" "}
          <span className="font-mono">
            {provider.budget.used.toFixed(2)} / {provider.budget.limit.toFixed(2)}{" "}
            {provider.budget.currency}
          </span>
        </p>
      )}
      {provider.fetched_at && (
        <p className="text-xs text-muted-foreground">
          {s.updated} {formatLocalTime(provider.fetched_at, locale)}
        </p>
      )}
    </div>
  );
}

function WindowRow({
  quotaWindow,
  locale,
  s,
}: {
  quotaWindow: QuotaWindow;
  locale: string;
  s: QuotaStrings;
}) {
  const percent = clampPercent(quotaWindow.used_percent);
  const resetLocal = formatLocalTime(quotaWindow.resets_at, locale);
  const resetIn = formatResetIn(quotaWindow.resets_at, new Date());
  const resetText = [resetLocal && `${s.resets} ${resetLocal}`, resetIn]
    .filter(Boolean)
    .join(" · ");
  const scopeLabel = quotaWindow.scope === "model" ? s.modelScope : s.accountScope;
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between gap-2 text-sm">
        <span>
          {quotaWindow.label} <span className="text-xs text-muted-foreground">({scopeLabel})</span>
        </span>
        <span className="font-mono">{percent === null ? "—" : `${Math.round(percent)}%`}</span>
      </div>
      <div
        role="progressbar"
        aria-label={quotaWindow.label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent ?? undefined}
        className="h-2 w-full overflow-hidden rounded-full bg-muted"
      >
        <div
          className={`h-full ${percent !== null && percent >= 90 ? "bg-destructive" : "bg-primary"}`}
          style={{ width: `${percent ?? 0}%` }}
        />
      </div>
      {resetText && <p className="text-xs text-muted-foreground">{resetText}</p>}
    </div>
  );
}
