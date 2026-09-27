"""Regression test for the conflict pair key (E8).

The `conflicts` table records detected contradictions as rows, and the
detector does not canonicalize orientation: measured 2026-09-26, 15 of 32
persisted rows violated `fact_a_id < fact_b_id`. A plain UNIQUE on
(fact_a_id, fact_b_id) therefore cannot see the swapped re-land `(b, a)`,
and the same contradiction accumulates rows.

Two halves must ship together:

1. An order-normalized unique index on the pair,
   `min(fact_a_id, fact_b_id), max(fact_a_id, fact_b_id)` — created by the
   canonical DDL in `VeracityConsolidator._init_tables` for fresh banks and
   by `mnemosyne.migrations.e8_conflict_pair_key` for existing banks
   (`CREATE TABLE IF NOT EXISTS` never retrofits a DDL change).
2. `_record_conflict`'s insert gaining `ON CONFLICT DO NOTHING`. Without
   it, the index converts silent re-growth into an IntegrityError inside a
   caller's `_serialized_write` scope — the partial-state class the method's
   own docstring guards.

The test asserts the relationship (a re-detected pair cannot become a second
row), never a row count of any live bank.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mnemosyne.core.veracity_consolidation import VeracityConsolidator
from mnemosyne.migrations.e8_conflict_pair_key import migrate_conflict_pair_key

INDEX_NAME = "idx_conflicts_pair_norm"

CONFLICTS_DDL = """
CREATE TABLE IF NOT EXISTS conflicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_a_id TEXT NOT NULL,
    fact_b_id TEXT NOT NULL,
    conflict_type TEXT,
    resolution TEXT,
    resolved_at TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


def _legacy_bank(path: Path, rows) -> Path:
    """A bank whose conflicts table predates the index (the migration's input)."""
    con = sqlite3.connect(str(path))
    try:
        con.execute(CONFLICTS_DDL)
        con.executemany(
            "INSERT INTO conflicts (fact_a_id, fact_b_id, conflict_type) VALUES (?, ?, ?)",
            rows,
        )
        con.commit()
    finally:
        con.close()
    return path


def _index_present(db_path: Path) -> bool:
    con = sqlite3.connect(str(db_path))
    try:
        row = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?", (INDEX_NAME,)
        ).fetchone()
    finally:
        con.close()
    return row is not None


def _row_count(db_path: Path) -> int:
    con = sqlite3.connect(str(db_path))
    try:
        return con.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0]
    finally:
        con.close()


def _ids(db_path: Path) -> set:
    con = sqlite3.connect(str(db_path))
    try:
        return {r[0] for r in con.execute("SELECT id FROM conflicts").fetchall()}
    finally:
        con.close()


def test_migration_adds_index_and_preserves_rows(tmp_path):
    # Third row is orientation-swapped against the second — the shape a
    # two-column unique key cannot normalize.
    bank = _legacy_bank(
        tmp_path / "bank.db",
        [
            ("cf_a", "cf_b", "contradiction"),
            ("cf_c", "cf_d", "contradiction"),
            ("cf_z", "cf_c", "contradiction"),
        ],
    )
    before_ids = _ids(bank)

    report = migrate_conflict_pair_key(bank)

    assert report["applied"] is True
    assert report["index_added"] is True
    assert report["duplicate_pairs"] == []
    assert _index_present(bank) is True
    assert _ids(bank) == before_ids  # index-only: no row written, none deleted


def test_migration_is_idempotent(tmp_path):
    bank = _legacy_bank(tmp_path / "bank.db", [("cf_a", "cf_b", "contradiction")])

    first = migrate_conflict_pair_key(bank)
    second = migrate_conflict_pair_key(bank)

    assert first["index_added"] is True
    assert second["index_already_present"] is True
    assert second["index_added"] is False
    assert _row_count(bank) == 1


def test_migration_refuses_duplicate_laden_bank_without_writing(tmp_path):
    # Same normalized pair under both orderings: a unique index cannot be
    # created over this, and choosing a winner is adjudication, not migration.
    bank = _legacy_bank(
        tmp_path / "bank.db",
        [("cf_x", "cf_y", "contradiction"), ("cf_y", "cf_x", "contradiction")],
    )

    report = migrate_conflict_pair_key(bank)

    assert report["applied"] is False
    assert report["index_added"] is False
    assert report["duplicate_pairs"] == ["cf_x/cf_y"]
    assert _index_present(bank) is False
    assert _row_count(bank) == 2  # nothing rewritten, nothing deleted


def test_migration_reports_missing_conflicts_table(tmp_path):
    bank = tmp_path / "other.db"
    con = sqlite3.connect(str(bank))
    con.execute("CREATE TABLE facts (id TEXT)")
    con.commit()
    con.close()

    report = migrate_conflict_pair_key(bank)

    assert report["conflicts_table_missing"] is True
    assert report["applied"] is False


def test_dry_run_creates_nothing(tmp_path):
    bank = _legacy_bank(tmp_path / "bank.db", [("cf_a", "cf_b", "contradiction")])

    report = migrate_conflict_pair_key(bank, dry_run=True)

    assert report["index_added"] is True  # would add
    assert report["dry_run"] is True
    assert _index_present(bank) is False  # but did not


def test_canonical_init_creates_index_on_fresh_bank(tmp_path):
    """The DDL path and the migration path converge: a fresh bank is born indexed."""
    db_path = tmp_path / "fresh.db"
    consolidator = VeracityConsolidator(db_path=db_path)
    try:
        consolidator._init_tables()
    finally:
        consolidator.conn.close()

    assert _index_present(db_path) is True
    # And the migration is then a no-op.
    report = migrate_conflict_pair_key(db_path)
    assert report["index_already_present"] is True


def test_record_conflict_swapped_pair_does_not_grow_the_ledger(tmp_path):
    """The regression this ships against: a re-detected pair, either orientation.

    First insert lands. The swapped re-detection must be a no-op and must not
    raise — a raise here lands inside a caller's `_serialized_write` scope and
    leaks partial state (fact row durable, conflict record lost).
    """
    db_path = tmp_path / "bank.db"
    consolidator = VeracityConsolidator(db_path=db_path)
    try:
        consolidator._init_tables()
        consolidator._record_conflict("cf_a", "cf_b", "contradiction")

        # Same pair, inverted orientation, and the original orientation again.
        consolidator._record_conflict("cf_b", "cf_a", "contradiction")
        consolidator._record_conflict("cf_a", "cf_b", "contradiction")

        assert _row_count(db_path) == 1
    finally:
        consolidator.conn.close()


def test_index_is_what_makes_the_bare_insert_unsafe(tmp_path):
    """Why the index must not ship alone: without the upsert, it raises.

    This pins the pairing requirement from the other side — the constraint is
    real, so an index-only PR would present as IntegrityError on the ambient
    detection path rather than as silent re-growth.
    """
    db_path = tmp_path / "bank.db"
    consolidator = VeracityConsolidator(db_path=db_path)
    try:
        consolidator._init_tables()
        consolidator._record_conflict("cf_a", "cf_b", "contradiction")

        with pytest.raises(sqlite3.IntegrityError):
            consolidator.conn.execute(
                "INSERT INTO conflicts (fact_a_id, fact_b_id, conflict_type) VALUES (?, ?, ?)",
                ("cf_b", "cf_a", "contradiction"),
            )
        consolidator.conn.rollback()
        assert _row_count(db_path) == 1
    finally:
        consolidator.conn.close()
