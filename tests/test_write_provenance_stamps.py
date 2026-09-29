"""Regression tests for the write-provenance tranche (G5, 2026-09-29).

Three law-to-column repairs, each with a measured pre-state from the
six-bank census (journal notice sf_fe48aa91df948f7931a7c824):

1. VERACITY-BY-CHANNEL: before, every auto-captured row (source
   'conversation') was born veracity='unknown' — 88-90% of private banks.
   Now an 'unknown' verdict derives from the write channel:
   conversation->stated, tool/verification->tool, document->imported.
   Never inflates: unmapped sources keep 'unknown'; an explicit label
   from the caller is never overwritten.

2. AUTHOR KWARGS ON remember(): the surface writer stamped
   metadata.writer_profile on 152/209 rows while author_id stayed NULL
   on all of them. remember() now accepts author_id/author_type kwargs
   (caller-authoritative, instance-default fallback) so integrations
   that know the writing profile can feed the typed column.

3. TRIM TOMBSTONES: _trim_working_memory deleted rows silently — a
   cited-then-trimmed handle was indistinguishable from a never-minted
   id (33 dangling, zero audit traces). Every evicted id now leaves an
   audit_log row (action='trim') before the DELETE, and the
   consolidated/pinned exemptions keep their (no-tombstone, no-delete)
   behavior.
"""

import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

import mnemosyne.core.beam as beam_mod
from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


def _col(db_path, memory_id, column):
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            f"SELECT {column} FROM working_memory WHERE id = ?", (memory_id,)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


class TestVeracityByChannel:
    def test_conversation_capture_is_stated(self, temp_db):
        mem = BeamMemory(session_id="s1", db_path=temp_db)
        mid = mem.remember("the operator said the deploy shipped", source="conversation")
        assert _col(temp_db, mid, "veracity") == "stated"

    def test_tool_and_verification_are_tool(self, temp_db):
        mem = BeamMemory(session_id="s1", db_path=temp_db)
        m1 = mem.remember("census output: 209 rows", source="tool")
        m2 = mem.remember("verdict row read-back ok", source="verification")
        assert _col(temp_db, m1, "veracity") == "tool"
        assert _col(temp_db, m2, "veracity") == "tool"

    def test_explicit_verdict_never_overwritten(self, temp_db):
        mem = BeamMemory(session_id="s1", db_path=temp_db)
        mid = mem.remember("a claim labeled inferred by its author",
                           source="conversation", veracity="inferred")
        assert _col(temp_db, mid, "veracity") == "inferred"

    def test_unmapped_channel_stays_unknown(self, temp_db):
        mem = BeamMemory(session_id="s1", db_path=temp_db)
        mid = mem.remember("some insight without a mapped channel", source="insight")
        assert _col(temp_db, mid, "veracity") == "unknown"


class TestAuthorKwargs:
    def test_kwargs_land_in_typed_columns(self, temp_db):
        mem = BeamMemory(session_id="s1", db_path=temp_db)
        mid = mem.remember("a surface row", source="surface_manual",
                           author_id="keeper", author_type="profile")
        conn = sqlite3.connect(str(temp_db))
        try:
            row = conn.execute(
                "SELECT author_id, author_type FROM working_memory WHERE id = ?", (mid,)
            ).fetchone()
        finally:
            conn.close()
        assert row == ("keeper", "profile")

    def test_instance_author_is_the_fallback(self, temp_db):
        mem = BeamMemory(session_id="s1", db_path=temp_db,
                         author_id="senses", author_type="profile")
        mid = mem.remember("instance-default author", source="observation")
        assert _col(temp_db, mid, "author_id") == "senses"

    def test_explicit_kwargs_win_over_instance(self, temp_db):
        mem = BeamMemory(session_id="s1", db_path=temp_db,
                         author_id="senses", author_type="profile")
        mid = mem.remember("delegated write", source="surface_manual",
                           author_id="why", author_type="profile")
        assert _col(temp_db, mid, "author_id") == "why"


class TestTrimTombstones:
    # NOTE (found by this test, pre-existing behavior): the trim's
    # keep-newest-N chrono predicate resolves at SECOND granularity —
    # writes inside one second tie and the eviction set is arbitrary
    # among them. Tests space writes >1s apart to be deterministic;
    # production consequence: keep-newest-N is a best-effort bound
    # under bursty writes, never a guarantee of "oldest gone first".

    def test_evicted_ids_leave_tombstones_and_exempt_survive(self, temp_db, monkeypatch):
        monkeypatch.setattr(beam_mod, "WORKING_MEMORY_MAX_ITEMS", 3)
        mem = BeamMemory(session_id="s1", db_path=temp_db)
        written = []
        for i in range(6):
            written.append(mem.remember(f"census row {i} of 6", source="conversation"))
            time.sleep(1.05)  # distinct chrono seconds (see NOTE above)
        conn = sqlite3.connect(str(temp_db))
        try:
            live = [r[0] for r in conn.execute(
                "SELECT id FROM working_memory ORDER BY timestamp DESC")]
            tombs = [r[0] for r in conn.execute(
                "SELECT memory_id FROM audit_log WHERE action = 'trim'")]
            # tombstones carry the bank classification
            banks = {r[0] for r in conn.execute(
                "SELECT DISTINCT bank FROM audit_log WHERE action = 'trim'")}
        finally:
            conn.close()
        assert len(live) == 3                       # keep-newest-3 intact
        assert set(tombs) == set(written[:3])       # exactly the doomed ones
        assert banks == {"private"}                 # tmp path has no 'shared'

    def test_pinned_and_consolidated_never_tombstoned_or_trimmed(self, temp_db, monkeypatch):
        monkeypatch.setattr(beam_mod, "WORKING_MEMORY_MAX_ITEMS", 3)
        mem = BeamMemory(session_id="s1", db_path=temp_db)
        p1 = mem.remember("pinned survivor", source="conversation")
        c1 = mem.remember("consolidated survivor", source="conversation")
        # exempt them BEFORE fillers can evict them (they start unpinned)
        mem.conn.execute("UPDATE working_memory SET pinned = 1 WHERE id = ?", (p1,))
        mem.conn.execute(
            "UPDATE working_memory SET consolidated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (c1,))
        mem.conn.commit()
        fillers = []
        for i in range(4):  # 4 unique candidates > max 3 → one eviction
            fillers.append(mem.remember(f"filler eviction candidate {i}",
                                         source="conversation"))
            time.sleep(1.05)
        conn = sqlite3.connect(str(temp_db))
        try:
            still = {r[0] for r in conn.execute("SELECT id FROM working_memory")}
            tombs = {r[0] for r in conn.execute(
                "SELECT memory_id FROM audit_log WHERE action = 'trim'")}
        finally:
            conn.close()
        assert tombs, "expected at least one filler tombstone"
        assert p1 in still and c1 in still          # survivors intact
        assert p1 not in tombs and c1 not in tombs  # exemptions never tombstoned
        assert tombs <= set(fillers)                # only fillers were evicted
