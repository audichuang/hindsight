// fork: cost-accounting
// Cost panel and per-call cost line for the LLM requests view. Renders nothing
// when the fork endpoints are absent (feature off), and logs nothing in that case.

import { useEffect, useState } from "react";
import { useLocale } from "next-intl";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { LLMRequestEntry } from "@/lib/api";
import { useFeatures } from "@/lib/features-context";
import {
  browserTimeZone,
  fetchLLMCostSummary,
  fetchLLMRowCosts,
  formatUsd,
  isPartlyProviderReported,
  type BillingKind,
  type CostPeriod,
  type CostSource,
  type CostStatus,
  type CostStatusCounts,
  type LLMCostSummaryResponse,
  type RowCost,
  type UnpricedReason,
} from "@/lib/fork/llm-cost";

interface CostStrings {
  title: string;
  periodLabel: string;
  period1d: string;
  period7d: string;
  period30d: string;
  loading: string;
  loadError: string;
  empty: string;
  truncated: string;
  meteredTotal: string;
  planValue: string;
  planNote: string;
  lowerBound: string;
  pricingSource: string;
  planPrefix: string;
  unpricedSuffix: string;
  statusCountsLabel: string;
  partlyProviderReported: string;
  providerReported: string;
  copilotNote: string;
  tokensEstimated: string;
  byDay: string;
  byDayTz: string;
  colDay: string;
  colMeteredCost: string;
  colPlanListPrice: string;
  colProvider: string;
  colModel: string;
  colBilling: string;
  colCalls: string;
  colInput: string;
  colCached: string;
  colWrite: string;
  colOutput: string;
  colThoughts: string;
  colCost: string;
  colUnpriced: string;
  rowCost: string;
  rowListPrice: string;
  rowBilling: string;
  rowReason: string;
  rowPricedAs: string;
  rowStatus: string;
  rowSource: string;
  rowProviderCost: string;
  rowTokensEstimated: string;
  billing: Record<BillingKind, string>;
  reasons: Record<UnpricedReason, string>;
  status: Record<CostStatus, string>;
  source: Record<CostSource, string>;
}

const EN: CostStrings = {
  title: "LLM cost",
  periodLabel: "Period",
  period1d: "Last 24 hours",
  period7d: "Last 7 days",
  period30d: "Last 30 days",
  loading: "Loading cost data…",
  loadError: "Could not load cost data.",
  empty: "No LLM calls in this period.",
  truncated: "Only the most recent calls were included; totals are partial.",
  meteredTotal: "Metered cost",
  planValue: "≈ API list price, not billed",
  planNote: "API list price, not billed",
  lowerBound:
    "Some calls were recorded before cache-write tracking, so their cost is a lower bound.",
  pricingSource: "Pricing source",
  planPrefix: "≈ ",
  unpricedSuffix: "unpriced",
  statusCountsLabel: "Calls by cost status",
  partlyProviderReported: "partly provider-reported",
  providerReported: "Provider-reported",
  copilotNote:
    "GitHub Copilot bills premium requests, not tokens, so its list price here is notional.",
  tokensEstimated:
    "Some token counts were estimated from text length because the provider did not report them.",
  byDay: "By day",
  byDayTz: "Time zone",
  colDay: "Day",
  colMeteredCost: "Metered cost",
  colPlanListPrice: "Plan list price",
  colProvider: "Provider",
  colModel: "Model",
  colBilling: "Billing",
  colCalls: "Calls",
  colInput: "Input",
  colCached: "Cache read",
  colWrite: "Cache write",
  colOutput: "Output",
  colThoughts: "Thoughts",
  colCost: "Cost",
  colUnpriced: "Unpriced",
  rowCost: "Cost",
  rowListPrice: "List price",
  rowBilling: "Billing",
  rowReason: "Reason",
  rowPricedAs: "Priced as",
  rowStatus: "Cost status",
  rowSource: "Cost source",
  rowProviderCost: "Provider-reported cost",
  rowTokensEstimated: "Token counts estimated",
  billing: { metered: "metered", plan: "plan", local: "local" },
  reasons: {
    subscription_plan: "subscription plan",
    local: "local model",
    unmapped_provider: "provider not mapped",
    unknown_model: "model not in price table",
    no_usage: "no token usage recorded",
    invalid_usage: "invalid token usage",
  },
  status: {
    actual: "provider-reported",
    estimated: "estimated",
    included: "included in plan",
    unknown: "unknown",
  },
  source: {
    provider: "provider",
    "genai-prices": "genai-prices table",
    custom: "custom price",
    none: "none",
  },
};

const ZH_TW: CostStrings = {
  title: "LLM 費用",
  periodLabel: "期間",
  period1d: "最近 24 小時",
  period7d: "最近 7 天",
  period30d: "最近 30 天",
  loading: "載入費用資料中…",
  loadError: "無法載入費用資料。",
  empty: "此期間沒有 LLM 呼叫。",
  truncated: "只納入最近的呼叫，總計為部分結果。",
  meteredTotal: "按量計費成本",
  planValue: "≈ API 標價，不計費",
  planNote: "API 標價，不計費",
  lowerBound: "部分呼叫記錄於快取寫入追蹤之前，因此費用為下限。",
  pricingSource: "價格來源",
  planPrefix: "≈ ",
  unpricedSuffix: "未計價",
  statusCountsLabel: "各費用狀態的呼叫數",
  partlyProviderReported: "部分為供應商回報",
  providerReported: "供應商回報",
  copilotNote:
    "GitHub Copilot 依 premium request 計費，而非 token，因此此處的 list 價格僅為參考值。",
  tokensEstimated: "部分 token 數因供應商未回報，依文字長度估算。",
  byDay: "每日明細",
  byDayTz: "時區",
  colDay: "日期",
  colMeteredCost: "按量計費成本",
  colPlanListPrice: "訂閱 list 價格",
  colProvider: "供應商",
  colModel: "模型",
  colBilling: "計費",
  colCalls: "呼叫次數",
  colInput: "輸入",
  colCached: "快取讀取",
  colWrite: "快取寫入",
  colOutput: "輸出",
  colThoughts: "推理",
  colCost: "成本",
  colUnpriced: "未計價",
  rowCost: "成本",
  rowListPrice: "list 價格",
  rowBilling: "計費",
  rowReason: "原因",
  rowPricedAs: "計價依據",
  rowStatus: "費用狀態",
  rowSource: "費用來源",
  rowProviderCost: "供應商回報成本",
  rowTokensEstimated: "token 數為估算",
  billing: { metered: "按量", plan: "訂閱", local: "本地" },
  reasons: {
    subscription_plan: "訂閱方案",
    local: "本地模型",
    unmapped_provider: "供應商未對應",
    unknown_model: "模型不在價格表中",
    no_usage: "未記錄 token 用量",
    invalid_usage: "token 用量無效",
  },
  status: {
    actual: "供應商回報",
    estimated: "估算",
    included: "已含於訂閱",
    unknown: "未知",
  },
  source: {
    provider: "供應商",
    "genai-prices": "genai-prices 價格表",
    custom: "自訂價格",
    none: "無",
  },
};

/** Keyed by locale. Every locale must carry the same keys as `en`. */
export const STRINGS: Record<string, CostStrings> = { en: EN, "zh-TW": ZH_TW };

function stringsFor(locale: string): CostStrings {
  return STRINGS[locale] ?? STRINGS.en;
}

type CostLoad =
  | { key: string; status: "ready"; data: LLMCostSummaryResponse }
  | { key: string; status: "off" | "error" };

export function LLMCostPanel({ bankId }: { bankId: string }) {
  const s = stringsFor(useLocale());
  // Nothing mounts and nothing is fetched unless the control plane reports the feature on.
  const enabled = useFeatures().features?.llm_cost === true;
  const [period, setPeriod] = useState<CostPeriod>("7d");
  const [load, setLoad] = useState<CostLoad | null>(null);
  const tz = browserTimeZone();
  const key = `${bankId}|${period}|${tz}`;

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    fetchLLMCostSummary(bankId, { period, tz })
      .then((data) => {
        if (cancelled) return;
        setLoad(data ? { key, status: "ready", data } : { key, status: "off" });
      })
      .catch(() => {
        if (!cancelled) setLoad({ key, status: "error" });
      });
    return () => {
      cancelled = true;
    };
  }, [bankId, period, tz, key, enabled]);

  const current = load?.key === key ? load : null;
  if (!enabled) return null;

  // The card hides itself when the cost endpoint answers 404.
  const costCard =
    current?.status === "off" ? null : (
      <Card>
        <CardHeader className="flex flex-row items-center justify-between gap-3 flex-wrap">
          <CardTitle>{s.title}</CardTitle>
          <div className="flex items-center gap-2 text-sm">
            <span className="text-muted-foreground">{s.periodLabel}</span>
            <Select value={period} onValueChange={(v) => setPeriod(v as CostPeriod)}>
              <SelectTrigger className="w-[160px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1d">{s.period1d}</SelectItem>
                <SelectItem value="7d">{s.period7d}</SelectItem>
                <SelectItem value="30d">{s.period30d}</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {current === null && <p className="text-sm text-muted-foreground">{s.loading}</p>}
          {current?.status === "error" && <p className="text-sm text-destructive">{s.loadError}</p>}
          {current?.status === "ready" && <CostSummaryBody data={current.data} s={s} />}
        </CardContent>
      </Card>
    );

  return costCard;
}

/** "3 provider-reported · 2 estimated · 4 included in plan · 0 unknown", in the locale's words. */
function StatusCountsLine({ counts, s }: { counts: CostStatusCounts; s: CostStrings }) {
  const statuses: CostStatus[] = ["actual", "estimated", "included", "unknown"];
  return (
    <div className="text-xs text-muted-foreground">
      <span>{s.statusCountsLabel}: </span>
      {statuses.map((status, index) => (
        <span key={status}>
          {index > 0 && " · "}
          <span className="font-mono">{counts[status]}</span> {s.status[status]}
        </span>
      ))}
    </div>
  );
}

function CostSummaryBody({ data, s }: { data: LLMCostSummaryResponse; s: CostStrings }) {
  const tokensEstimated = data.models.some((m) => m.tokens_estimated_rows > 0);
  const hasCopilot = data.models.some((m) => m.provider === "github-copilot");
  return (
    <>
      <div className="flex flex-wrap gap-x-8 gap-y-2 text-sm">
        <div>
          <span className="text-muted-foreground">{s.meteredTotal}</span>{" "}
          <span className="font-mono font-medium">{formatUsd(data.cost_usd)}</span>
        </div>
        <div>
          <span className="text-muted-foreground">{s.planValue}</span>{" "}
          <span className="font-mono">{formatUsd(data.plan_list_price_usd)}</span>
        </div>
        <div>
          <span className="text-muted-foreground">{s.pricingSource}</span>{" "}
          <span className="font-mono text-xs">{data.pricing_source}</span>
        </div>
      </div>
      <StatusCountsLine counts={data.status_counts} s={s} />
      {data.truncated && <p className="text-xs text-muted-foreground">{s.truncated}</p>}
      {data.models.some((m) => m.legacy_rows_repaired > 0 || m.rows_without_breakdown > 0) && (
        <p className="text-xs text-muted-foreground">{s.lowerBound}</p>
      )}
      {hasCopilot && <p className="text-xs text-muted-foreground">{s.copilotNote}</p>}
      {tokensEstimated && <p className="text-xs text-muted-foreground">{s.tokensEstimated}</p>}
      {data.models.length === 0 ? (
        <p className="text-sm text-muted-foreground">{s.empty}</p>
      ) : (
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{s.colProvider}</TableHead>
                <TableHead>{s.colModel}</TableHead>
                <TableHead>{s.colBilling}</TableHead>
                <TableHead className="text-right">{s.colCalls}</TableHead>
                <TableHead className="text-right">{s.colInput}</TableHead>
                <TableHead className="text-right">{s.colCached}</TableHead>
                <TableHead className="text-right">{s.colWrite}</TableHead>
                <TableHead className="text-right">{s.colOutput}</TableHead>
                <TableHead className="text-right">{s.colThoughts}</TableHead>
                <TableHead className="text-right">{s.colCost}</TableHead>
                <TableHead>{s.colUnpriced}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.models.map((m) => (
                <TableRow key={`${m.provider ?? ""}/${m.model ?? ""}/${m.billing}`}>
                  <TableCell className="font-mono text-xs">{m.provider ?? "—"}</TableCell>
                  <TableCell className="font-mono text-xs">{m.model ?? "—"}</TableCell>
                  <TableCell className="text-xs">{s.billing[m.billing]}</TableCell>
                  <TableCell className="text-right font-mono">{m.calls}</TableCell>
                  <TableCell className="text-right font-mono">
                    {m.input_tokens.toLocaleString()}
                  </TableCell>
                  <TableCell className="text-right font-mono">
                    {m.cache_read_tokens.toLocaleString()}
                  </TableCell>
                  <TableCell className="text-right font-mono">
                    {m.cache_write_tokens.toLocaleString()}
                  </TableCell>
                  <TableCell className="text-right font-mono">
                    {m.output_tokens.toLocaleString()}
                  </TableCell>
                  <TableCell className="text-right font-mono">
                    {m.thoughts_tokens.toLocaleString()}
                  </TableCell>
                  <TableCell className="text-right font-mono">
                    {m.billing === "plan" ? (
                      <>
                        {`${s.planPrefix}${formatUsd(m.list_price_usd)}`}
                        <span className="block text-[10px] font-sans text-muted-foreground">
                          {s.planNote}
                        </span>
                      </>
                    ) : (
                      formatUsd(m.cost_usd)
                    )}
                    {m.provider_cost_usd !== null && (
                      <span className="block text-[10px] font-sans text-muted-foreground">
                        {`${s.providerReported} ${formatUsd(m.provider_cost_usd)}`}
                      </span>
                    )}
                    {isPartlyProviderReported(m.billing, m.calls, m.status_counts) && (
                      <span className="block text-[10px] font-sans text-muted-foreground">
                        {s.partlyProviderReported}
                      </span>
                    )}
                  </TableCell>
                  <TableCell className="text-xs">
                    {m.unpriced_calls > 0 && m.unpriced_reason !== null && (
                      <span>
                        {m.unpriced_calls} {s.unpricedSuffix} ({s.reasons[m.unpriced_reason]})
                      </span>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
      {data.by_day.length > 0 && (
        <details className="text-sm">
          <summary className="cursor-pointer text-muted-foreground">
            {s.byDay} ({s.byDayTz}: <span className="font-mono text-xs">{data.tz}</span>)
          </summary>
          <div className="overflow-x-auto pt-2">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{s.colDay}</TableHead>
                  <TableHead className="text-right">{s.colMeteredCost}</TableHead>
                  <TableHead className="text-right">{s.colPlanListPrice}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.by_day.map((d) => (
                  <TableRow key={d.day}>
                    <TableCell className="font-mono text-xs">{d.day}</TableCell>
                    <TableCell className="text-right font-mono">{formatUsd(d.cost_usd)}</TableCell>
                    <TableCell className="text-right font-mono">
                      {formatUsd(d.plan_list_price_usd)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </details>
      )}
    </>
  );
}

/** One call's cost, shown inside the request detail dialog. */
export function LLMCallCost({ entry }: { entry: LLMRequestEntry }) {
  const s = stringsFor(useLocale());
  const [result, setResult] = useState<{ id: string; row: RowCost | null } | null>(null);
  const enabled = useFeatures().features?.llm_cost === true;
  const bankId = entry.bank_id;
  const entryId = entry.id;

  useEffect(() => {
    if (!enabled || !bankId) return;
    let cancelled = false;
    fetchLLMRowCosts(bankId, [entryId])
      .then((res) => {
        if (!cancelled) setResult({ id: entryId, row: res?.items[entryId] ?? null });
      })
      .catch(() => {
        if (!cancelled) setResult({ id: entryId, row: null });
      });
    return () => {
      cancelled = true;
    };
  }, [enabled, bankId, entryId]);

  const row = result?.id === entryId ? result.row : null;
  if (!enabled || !row) return null;

  return (
    <div className="border-t border-border pt-3 space-y-1 text-sm">
      <div>
        <span className="text-muted-foreground">{s.rowBilling}</span>{" "}
        <span className="font-mono">{s.billing[row.billing]}</span>
      </div>
      <div>
        <span className="text-muted-foreground">{s.rowCost}</span>{" "}
        <span className="font-mono font-medium">
          {row.billing === "plan"
            ? `${s.planPrefix}${formatUsd(row.list_price_usd)}`
            : formatUsd(row.cost_usd)}
        </span>
        {row.billing === "plan" && (
          <span className="ml-2 text-xs text-muted-foreground">{s.planNote}</span>
        )}
      </div>
      {row.provider_cost_usd !== null && (
        <div>
          <span className="text-muted-foreground">{s.rowProviderCost}</span>{" "}
          <span className="font-mono">{formatUsd(row.provider_cost_usd)}</span>
        </div>
      )}
      <div>
        <span className="text-muted-foreground">{s.rowStatus}</span>{" "}
        <span className="font-mono">{s.status[row.cost_status]}</span>
      </div>
      <div>
        <span className="text-muted-foreground">{s.rowSource}</span>{" "}
        <span className="font-mono">{s.source[row.cost_source]}</span>
      </div>
      {row.billing !== "plan" && row.list_price_usd !== null && (
        <div>
          <span className="text-muted-foreground">{s.rowListPrice}</span>{" "}
          <span className="font-mono">{formatUsd(row.list_price_usd)}</span>
        </div>
      )}
      {row.tokens_estimated && (
        <div>
          <span className="text-xs text-muted-foreground">{s.rowTokensEstimated}</span>
        </div>
      )}
      {row.reason !== null && (
        <div>
          <span className="text-muted-foreground">{s.rowReason}</span>{" "}
          <span className="font-mono">{s.reasons[row.reason]}</span>
        </div>
      )}
      {row.priced_as !== null && (
        <div>
          <span className="text-muted-foreground">{s.rowPricedAs}</span>{" "}
          <span className="font-mono text-xs">{row.priced_as}</span>
        </div>
      )}
    </div>
  );
}
