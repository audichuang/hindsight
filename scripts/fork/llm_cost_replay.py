"""Replay exported llm_requests rows through the cost-accounting pricer (fork: cost-accounting).

Input is a CSV exported read-only from llm_requests, with the header
``id,bank_id,operation,scope,provider,model,started_at,input_tokens,output_tokens,cached_tokens,thoughts_tokens,usage``.
``usage`` is ``llm_info->'usage'`` as JSON text, empty when the row has none. Empty cells are NULL.

Prints the totals, the cost status counts, the per-day table (day buckets in ``--tz``, UTC by default),
the legacy_rows_repaired and rows_without_breakdown counts, and one grouping table chosen with ``--by``.
Pricing settings come from the HINDSIGHT_API_LLM_COST_* environment. ``--billing``, ``--provider-map``,
``--model-aliases`` and ``--custom-prices`` override those variables for one run.

Run from the repository root:

    uv run --directory hindsight-api-slim python ../scripts/fork/llm_cost_replay.py <csv> [--by day] [--tz Asia/Taipei]
"""

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, tzinfo
from pathlib import Path

from hindsight_api.config import llm_cost_env
from hindsight_api.engine.llm_cost import (
    ENV_BILLING,
    ENV_CUSTOM_PRICES,
    ENV_MODEL_ALIASES,
    ENV_PROVIDER_MAP,
    CostSettings,
    CostStatusCounts,
    LLMCostSummaryResponse,
    RowUsage,
    StoredUsage,
    load_cost_settings,
    parse_timezone,
    read_stored_usage,
    summarize_costs,
)

MULTIPLE_BANKS = "multiple"


@dataclass(frozen=True)
class ReplayRow:
    bank_id: str
    usage: RowUsage


@dataclass(frozen=True)
class Window:
    start: datetime | None = None
    end: datetime | None = None

    def contains(self, moment: datetime) -> bool:
        if self.start is not None and moment < self.start:
            return False
        return self.end is None or moment < self.end


@dataclass(frozen=True)
class Zone:
    """The zone the per-day buckets are cut in, with the name as the operator typed it."""

    tz: tzinfo = UTC
    name: str = "UTC"


def parse_timestamp(text: str) -> datetime:
    """ISO timestamp as Postgres writes it (``2026-10-01 12:00:00.5+00``) or as typed (``2026-10-01T12:00Z``). Naive means UTC."""
    value = text.strip().replace(" ", "T", 1)
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    if "T" in value and len(value) >= 3 and value[-3] in "+-" and value[-2:].isdigit():
        value += ":00"
    moment = datetime.fromisoformat(value)
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _optional_text(text: str) -> str | None:
    return text.strip() or None


def _optional_int(text: str) -> int | None:
    stripped = text.strip()
    return int(stripped) if stripped else None


def parse_usage(text: str) -> StoredUsage | None:
    """The ``llm_info.usage`` column, read by the same reader the store uses. Empty means no marker (written before the fork)."""
    if not text.strip():
        return None
    return read_stored_usage(json.loads(text))


def to_replay_row(record: Mapping[str, str]) -> ReplayRow:
    usage = RowUsage(
        provider=_optional_text(record["provider"]),
        model=_optional_text(record["model"]),
        started_at=parse_timestamp(record["started_at"]),
        input_tokens=_optional_int(record["input_tokens"]),
        output_tokens=_optional_int(record["output_tokens"]),
        cached_tokens=_optional_int(record["cached_tokens"]),
        thoughts_tokens=_optional_int(record["thoughts_tokens"]),
        usage=parse_usage(record["usage"]),
    )
    return ReplayRow(bank_id=record["bank_id"], usage=usage)


def load_rows(path: Path) -> list[ReplayRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [to_replay_row(record) for record in csv.DictReader(handle)]


def resolve_settings(overrides: Mapping[str, str | None]) -> CostSettings:
    """Environment settings, with any non-empty flag value replacing its variable for this run."""
    given = {name: value for name, value in overrides.items() if value is not None}
    if not given:
        return load_cost_settings()
    return CostSettings.from_env({**llm_cost_env(), **given})


def replay_summary(
    rows: Sequence[ReplayRow], settings: CostSettings, window: Window, zone: Zone = Zone()
) -> LLMCostSummaryResponse:
    banks = {row.bank_id for row in rows}
    bank_id = banks.pop() if len(banks) == 1 else MULTIPLE_BANKS
    return summarize_costs(
        [row.usage for row in rows],
        settings,
        bank_id=bank_id,
        period=None,
        start=window.start.isoformat() if window.start else "",
        end=window.end.isoformat() if window.end else None,
        truncated=False,
        tz=zone.tz,
        tz_name=zone.name,
    )


def _usd(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


def _status_text(counts: CostStatusCounts) -> str:
    return f"actual={counts.actual} estimated={counts.estimated} included={counts.included} unknown={counts.unknown}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [
        max(len(header), *(len(row[i]) for row in rows)) if rows else len(header) for i, header in enumerate(headers)
    ]
    lines = ["  ".join(header.ljust(width) for header, width in zip(headers, widths, strict=True)).rstrip()]
    lines.extend("  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip() for row in rows)
    return "\n".join(lines)


def _model_table(summary: LLMCostSummaryResponse) -> str:
    rows = [
        [
            totals.provider or "-",
            totals.model or "-",
            totals.billing,
            str(totals.calls),
            str(totals.unpriced_calls),
            totals.unpriced_reason or "-",
            str(totals.input_tokens),
            str(totals.cache_read_tokens),
            str(totals.cache_write_tokens),
            str(totals.output_tokens),
            _usd(totals.cost_usd),
            _usd(totals.list_price_usd),
            _usd(totals.provider_cost_usd),
            _status_text(totals.status_counts),
        ]
        for totals in summary.models
    ]
    headers = [
        "provider",
        "model",
        "billing",
        "calls",
        "unpriced",
        "reason",
        "input",
        "cache_read",
        "cache_write",
        "output",
        "cost_usd",
        "list_usd",
        "provider_usd",
        "status",
    ]
    return _table(headers, rows)


def _day_table(summary: LLMCostSummaryResponse) -> str:
    rows = [[day.day, _usd(day.cost_usd), _usd(day.plan_list_price_usd)] for day in summary.by_day]
    return _table([f"day ({summary.tz})", "cost_usd", "plan_list_usd"], rows)


def _bank_table(rows: Sequence[ReplayRow], settings: CostSettings, window: Window, zone: Zone) -> str:
    by_bank: dict[str, list[ReplayRow]] = {}
    for row in rows:
        by_bank.setdefault(row.bank_id, []).append(row)
    table_rows = []
    for bank_id in sorted(by_bank):
        summary = replay_summary(by_bank[bank_id], settings, window, zone)
        table_rows.append(
            [
                bank_id,
                str(sum(m.calls for m in summary.models)),
                _usd(summary.cost_usd),
                _usd(summary.plan_list_price_usd),
                str(summary.unpriced_calls),
            ]
        )
    return _table(["bank", "calls", "cost_usd", "plan_list_usd", "unpriced"], table_rows)


def render_report(
    rows: Sequence[ReplayRow], settings: CostSettings, window: Window, by: str, zone: Zone = Zone()
) -> str:
    summary = replay_summary(rows, settings, window, zone)
    calls = sum(model.calls for model in summary.models)
    legacy = sum(model.legacy_rows_repaired for model in summary.models)
    no_breakdown = sum(model.rows_without_breakdown for model in summary.models)
    provider_costs = [model.provider_cost_usd for model in summary.models if model.provider_cost_usd is not None]
    sections = [
        f"pricing source: {summary.pricing_source}",
        f"rows: {calls}",
        f"status_counts: {_status_text(summary.status_counts)}",
        f"cost_usd (metered + local): {_usd(summary.cost_usd)}",
        f"provider-reported cost (USD): {_usd(sum(provider_costs) if provider_costs else None)}",
        f"plan list price (notional): {_usd(summary.plan_list_price_usd)}",
        f"unpriced calls: {summary.unpriced_calls}",
        f"legacy_rows_repaired: {legacy}",
        f"rows_without_breakdown: {no_breakdown}",
        "",
        f"per day ({summary.tz}):",
        _day_table(summary),
    ]
    if by == "model":
        sections += ["", "per model:", _model_table(summary)]
    elif by == "bank":
        sections += ["", "per bank:", _bank_table(rows, settings, window, zone)]
    return "\n".join(sections)


def _parse_window_bound(text: str | None) -> datetime | None:
    return None if text is None else parse_timestamp(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay llm_requests rows through the cost pricer.")
    parser.add_argument("csv", type=Path, help="CSV exported from llm_requests")
    parser.add_argument("--window-start", help="inclusive lower bound, ISO, UTC if no offset")
    parser.add_argument("--window-end", help="exclusive upper bound, ISO, UTC if no offset")
    parser.add_argument("--by", choices=("model", "day", "bank"), default="model", help="grouping table to print")
    parser.add_argument("--tz", default="UTC", help="IANA time zone for the per-day buckets (default UTC)")
    parser.add_argument("--billing", dest="billing", help=f"overrides {ENV_BILLING}")
    parser.add_argument("--provider-map", dest="provider_map", help=f"overrides {ENV_PROVIDER_MAP}")
    parser.add_argument("--model-aliases", dest="model_aliases", help=f"overrides {ENV_MODEL_ALIASES}")
    parser.add_argument("--custom-prices", dest="custom_prices", help=f"overrides {ENV_CUSTOM_PRICES}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        zone = Zone(tz=parse_timezone(args.tz), name=args.tz)
    except ValueError as exc:
        parser.error(str(exc))
    settings = resolve_settings(
        {
            ENV_BILLING: args.billing,
            ENV_PROVIDER_MAP: args.provider_map,
            ENV_MODEL_ALIASES: args.model_aliases,
            ENV_CUSTOM_PRICES: args.custom_prices,
        }
    )
    window = Window(start=_parse_window_bound(args.window_start), end=_parse_window_bound(args.window_end))
    rows = [row for row in load_rows(args.csv) if window.contains(row.usage.started_at)]
    if not rows:
        print("no rows in the window")
        return 0
    print(render_report(rows, settings, window, args.by, zone))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
