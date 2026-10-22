"""A reprocess of a document retained without a date keeps the date it was written.

A retain with no ``timestamp`` stores no ``event_date`` in ``retain_params``, and
``_build_contents`` dates such an item ``utcnow()``. The original retain therefore
dated every fact at the moment it was written -- but a reprocess replays the same
missing key and dated the re-extracted facts at the moment of the REPROCESS, so a
year-old document recalled as the newest material in the bank, and the extraction
LLM resolved "tomorrow" against the reprocess day. ``documents.created_at`` is the
write time (``delete_document_for_replace`` hands it back so the replace keeps it),
so replaying it is the original retain's own behaviour, replayed faithfully.
"""

import inspect
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.retain import orchestrator as orch
from hindsight_api.engine.retain.fact_extraction import SECONDS_PER_FACT
from hindsight_api.engine.schema import fq_store_table_explicit as fq_table

_CREATED_AT = "2024-03-01T09:30:00+00:00"


class _StubEngine:
    """Just enough MemoryEngine for reprocess_document: it reads a document and
    submits a retain."""

    _operation_validator = None

    def __init__(self, doc: dict[str, Any]) -> None:
        self._doc = doc
        self.submitted: list[dict[str, Any]] = []

    async def _authenticate_tenant(self, request_context) -> None:
        return None

    async def _require_writable(self, bank_id, request_context, **kwargs) -> None:
        return None

    async def get_document(self, document_id, bank_id, request_context=None):
        return dict(self._doc)

    async def submit_async_retain(self, bank_id, contents, strategy=None, request_context=None, **kwargs):
        self.submitted.append(contents[0])
        return {"operation_id": "op-1"}


async def _replayed(retain_params: dict[str, Any], created_at: str | None = _CREATED_AT) -> dict[str, Any]:
    engine = _StubEngine(
        {"original_text": "Alice works at Google.", "retain_params": retain_params, "created_at": created_at}
    )
    await MemoryEngine.reprocess_document(engine, "bank-1", "doc-1", request_context=None)
    return engine.submitted[0]


def _params(**item) -> dict[str, Any]:
    retain_params, _tags = orch._build_retain_params([item])
    return retain_params


@pytest.mark.asyncio
async def test_an_undated_document_replays_its_write_time():
    item = await _replayed(_params(content="Alice works at Google."))
    assert item["event_date"] == _CREATED_AT


@pytest.mark.asyncio
async def test_a_stored_date_still_wins():
    item = await _replayed(_params(content="x", event_date="2023-05-06T07:08:09+00:00"))
    assert item["event_date"] == "2023-05-06T07:08:09+00:00"


@pytest.mark.asyncio
async def test_an_explicitly_undated_document_stays_undated():
    """timestamp "unset" stores event_date=None: the caller said "no date", which is
    not the same as saying nothing (#5473)."""
    item = await _replayed(_params(content="x", event_date=None))
    assert item["event_date"] is None


@pytest.mark.asyncio
async def test_a_placeholder_date_still_replays_as_undated():
    item = await _replayed({"event_date": "9999-12-31T23:59:59.999999+00:00"})
    assert item["event_date"] is None


@pytest.mark.asyncio
async def test_a_document_without_a_write_time_keeps_upstream_behaviour():
    """A store-owned record written before it carried created_at: nothing to replay,
    so the key stays absent and retain dates it now, as before."""
    item = await _replayed({}, created_at=None)
    assert "event_date" not in item


@pytest.mark.asyncio
async def test_the_write_time_survives_repeated_reprocesses():
    """The replayed date is captured into retain_params, so every later reprocess
    replays it from there: created_at is gone after the first one, and the date stays."""
    stored = _params(content="Alice works at Google.")
    created_at: str | None = _CREATED_AT
    for _ in range(3):
        item = await _replayed(stored, created_at=created_at)
        assert item["event_date"] == _CREATED_AT
        stored = _params(**item)
        created_at = None
    assert stored["event_date"] == _CREATED_AT


def test_memory_engine_keeps_the_reprocess_event_date_hook():
    assert "# fork: reprocess-event-date" in inspect.getsource(MemoryEngine.reprocess_document)


async def _source_facts(memory, bank_id: str, request_context) -> list[dict[str, Any]]:
    page = await memory.list_memory_units(
        bank_id, document_id="doc", fact_type=["world", "experience"], request_context=request_context
    )
    return page["items"]


@pytest.mark.asyncio
async def test_reprocessed_facts_keep_the_original_date(memory, request_context):
    """End to end: retain undated, age the document, reprocess, and every re-extracted
    fact carries the document's write time rather than the reprocess time."""
    bank_id = f"test-fork-reprocess-date-{uuid.uuid4().hex[:8]}"
    written = datetime(2024, 3, 1, 9, 30, tzinfo=UTC)
    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Alice works at Google. Bob works at Meta.", "document_id": "doc"}],
            request_context=request_context,
        )
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            # What an undated document retained back then looks like now; no API can set it.
            await conn.execute(
                f"UPDATE {fq_table('documents')} SET created_at = $1 WHERE bank_id = $2 AND id = 'doc'",
                written,
                bank_id,
            )
        before = {u["id"] for u in await _source_facts(memory, bank_id, request_context)}
        assert before

        result = await memory.reprocess_document(bank_id, "doc", request_context=request_context)
        assert result is not None

        units = await _source_facts(memory, bank_id, request_context)
        doc = await memory.get_document("doc", bank_id, request_context=request_context)
        assert units
        assert not ({u["id"] for u in units} & before), "reprocess did not re-extract"
        assert datetime.fromisoformat(doc["created_at"]) == written, "the replace no longer preserves created_at"
        # Retain spaces a document's facts SECONDS_PER_FACT apart to keep their order.
        window = timedelta(seconds=len(units) * SECONDS_PER_FACT)
        for unit in units:
            assert written <= datetime.fromisoformat(unit["date"]) <= written + window
            assert written <= datetime.fromisoformat(unit["mentioned_at"]) <= written + window
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
