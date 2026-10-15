"""The historical replay script prices a CSV export exactly as ``summarize_costs`` does (fork: cost-accounting).

The script lives outside the package, so it is loaded by path. The fixture is written to a tmp dir and
the expected totals are built directly as ``RowUsage`` values, so the CSV parsing is checked too.
"""

import importlib.util
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from zoneinfo import ZoneInfo

import pytest

from hindsight_api.engine.llm_cost import CostSettings, RowUsage, StoredUsage, summarize_costs

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "fork" / "llm_cost_replay.py"

_CSV = """id,bank_id,operation,scope,provider,model,status,started_at,input_tokens,output_tokens,cached_tokens,thoughts_tokens,usage
1,bank-a,retain,memory,anthropic,claude-haiku-5-5,success,2026-10-01 10:00:00.000000+00,2010,300,0,,"{""v"":1,""cache_write_tokens"":2000,""cache_write_5m_tokens"":2000,""cache_write_1h_tokens"":0}"
2,bank-a,reflect,tools,anthropic,claude-haiku-5-5,success,2026-10-01 23:30:00+00,12,250,9000,,
3,bank-b,retain,memory,openai-codex,gpt-6.1-sol,success,2026-10-02 01:00:00+00,1200,30,1000,60,
4,bank-b,retain,memory,xai-oauth,grok-4.7,success,2026-10-02 02:00:00+00,500,50,,,
"""

# One failed call at 17:30Z on Oct 1: in Asia/Taipei (UTC+8) it falls on Oct 2.
_TAIPEI_CSV = """id,bank_id,operation,scope,provider,model,status,started_at,input_tokens,output_tokens,cached_tokens,thoughts_tokens,usage
5,bank-c,retain,memory,anthropic,claude-haiku-5-5,error,2026-10-01T17:30:00Z,10,1,0,,
"""


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("llm_cost_replay", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


replay = _load_script()


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    path = tmp_path / "llm_requests.csv"
    path.write_text(_CSV, encoding="utf-8")
    return path


def _expected_usage() -> list[RowUsage]:
    day_one = datetime(2026, 10, 1, 10, 0, tzinfo=UTC)
    return [
        RowUsage(
            provider="anthropic",
            model="claude-haiku-5-5",
            started_at=day_one,
            input_tokens=2010,
            output_tokens=300,
            cached_tokens=0,
            thoughts_tokens=None,
            usage=StoredUsage(v=1, cache_write_tokens=2000, cache_write_5m_tokens=2000, cache_write_1h_tokens=0),
        ),
        RowUsage(
            provider="anthropic",
            model="claude-haiku-5-5",
            started_at=datetime(2026, 10, 1, 23, 30, tzinfo=UTC),
            input_tokens=12,
            output_tokens=250,
            cached_tokens=9000,
            thoughts_tokens=None,
        ),
        RowUsage(
            provider="openai-codex",
            model="gpt-6.1-sol",
            started_at=datetime(2026, 10, 2, 1, 0, tzinfo=UTC),
            input_tokens=1200,
            output_tokens=30,
            cached_tokens=1000,
            thoughts_tokens=60,
        ),
        RowUsage(
            provider="xai-oauth",
            model="grok-4.7",
            started_at=datetime(2026, 10, 2, 2, 0, tzinfo=UTC),
            input_tokens=500,
            output_tokens=50,
            cached_tokens=None,
            thoughts_tokens=None,
        ),
    ]


def test_replay_totals_match_summarize_costs(csv_path: Path) -> None:
    settings = CostSettings.from_env({})
    window = replay.Window()
    rows = replay.load_rows(csv_path)

    expected = summarize_costs(
        _expected_usage(), settings, bank_id="multiple", period=None, start="", end=None, truncated=False
    )
    actual = replay.replay_summary(rows, settings, window)

    assert actual.model_dump(exclude={"pricing_source"}) == expected.model_dump(exclude={"pricing_source"})
    # Hand check: row 1 = (10*0.10 + 2000*0.125 + 300*0.50)/1e6; row 2 = (12*0.10 + 9000*0.01 + 250*0.50)/1e6;
    # the legacy row's input is folded to 9012 before pricing.
    assert actual.cost_usd == pytest.approx(0.000401 + 0.0002162, abs=1e-9)
    assert actual.plan_list_price_usd == pytest.approx(0.0014 + 0.0013, abs=1e-9)


def test_replay_reads_the_usage_marker_per_row(csv_path: Path) -> None:
    rows = replay.load_rows(csv_path)

    assert rows[0].usage.usage == StoredUsage(
        v=1, cache_write_tokens=2000, cache_write_5m_tokens=2000, cache_write_1h_tokens=0
    )
    assert rows[1].usage.usage is None  # no llm_info.usage: no marker


def test_replay_window_keeps_rows_at_or_after_start_and_before_end(csv_path: Path) -> None:
    rows = replay.load_rows(csv_path)
    window = replay.Window(
        start=replay.parse_timestamp("2026-10-02T00:00:00Z"),
        end=replay.parse_timestamp("2026-10-02 02:00:00+00"),
    )
    kept = [row for row in rows if window.contains(row.usage.started_at)]
    assert [row.usage.model for row in kept] == ["gpt-6.1-sol"]


def test_replay_report_prints_counts(csv_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert replay.main([str(csv_path), "--by", "bank"]) == 0
    out = capsys.readouterr().out
    assert "rows: 4" in out
    assert "legacy_rows_repaired: 1" in out
    # Only cache-write providers count: the anthropic and codex rows lack a breakdown; the xai row is not one of them.
    assert "rows_without_breakdown: 2" in out
    assert "per bank:" in out


def test_replay_report_prints_status_counts_and_provider_cost(
    csv_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert replay.main([str(csv_path), "--by", "model"]) == 0
    out = capsys.readouterr().out
    assert "status_counts: actual=0 estimated=" in out
    assert "provider-reported cost (USD): -" in out  # no row in the fixture carries a provider cost
    assert "provider_usd" in out
    assert "per day (UTC):" in out


def test_replay_tz_moves_the_late_evening_row_to_the_next_local_day(tmp_path: Path) -> None:
    path = tmp_path / "taipei.csv"
    path.write_text(_TAIPEI_CSV, encoding="utf-8")
    rows = replay.load_rows(path)
    settings = CostSettings.from_env({})

    utc_days = [day.day for day in replay.replay_summary(rows, settings, replay.Window()).by_day]
    taipei = replay.replay_summary(rows, settings, replay.Window(), replay.Zone(ZoneInfo("Asia/Taipei"), "Asia/Taipei"))

    assert utc_days == ["2026-10-01"]
    assert [day.day for day in taipei.by_day] == ["2026-10-02"]
    assert taipei.tz == "Asia/Taipei"
    assert taipei.status_counts.estimated == 1  # priced from the genai-prices table, not provider-reported


def test_replay_tz_flag_names_the_day_table(csv_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert replay.main([str(csv_path), "--tz", "Asia/Taipei", "--by", "day"]) == 0
    out = capsys.readouterr().out
    assert "per day (Asia/Taipei):" in out
    assert "day (Asia/Taipei)" in out


def test_replay_tz_flag_defaults_to_utc(csv_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert replay.main([str(csv_path)]) == 0
    assert "per day (UTC):" in capsys.readouterr().out


def test_replay_rejects_an_unknown_tz(csv_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        replay.main([str(csv_path), "--tz", "Mars/Base"])
    assert raised.value.code == 2
    assert "IANA time zone" in capsys.readouterr().err


def test_replay_reports_empty_window(csv_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert replay.main([str(csv_path), "--window-start", "2030-01-01"]) == 0
    assert "no rows in the window" in capsys.readouterr().out


def test_parse_timestamp_accepts_postgres_and_typed_forms() -> None:
    expected = datetime(2026, 10, 1, 12, 0, 0, 500000, tzinfo=UTC)
    assert replay.parse_timestamp("2026-10-01 12:00:00.5+00") == expected
    assert replay.parse_timestamp("2026-10-01T12:00:00.500000Z") == expected
    assert replay.parse_timestamp("2026-10-01 12:00:00.5") == expected


def test_empty_usage_column_has_no_marker() -> None:
    assert replay.parse_usage("") is None
    assert replay.parse_usage("   ") is None


def test_usage_column_that_is_not_an_object_has_no_marker() -> None:
    assert replay.parse_usage("[1, 2]") is None
