"""
Mnemosyne E8 Migration — order-normalized unique key on ``conflicts``
=====================================================================

Idempotent migration that adds ONE expression index to an existing bank:

  - ``idx_conflicts_pair_norm`` — UNIQUE on (min(fact_a_id, fact_b_id),
    max(fact_a_id, fact_b_id))

Why an expression index and not a two-column one: the detector does not
canonicalize orientation. 15 of 32 persisted conflict rows measured
2026-09-26 violate ``fact_a_id < fact_b_id``, so a plain UNIQUE on
(fact_a_id, fact_b_id) cannot see the swapped pair ``(b, a)`` and the same
contradiction re-lands as a third row.

Canonical DDL source: ``mnemosyne/core/veracity_consolidation.py``
(``_init_conflicts_table``, ``CREATE TABLE IF NOT EXISTS conflicts``). The
index DDL below belongs beside that statement upstream — and because the
table DDL is ``IF NOT EXISTS``, changing it reaches fresh installs only,
which is exactly why existing banks need this migration. Per the E7
precedent we do NOT invent DDL here; if upstream's canonical index
definition changes, this migration is updated to match.

PAIRING REQUIREMENT (do not ship this migration alone): core's
``_record_conflict`` currently executes a bare
``INSERT INTO conflicts (fact_a_id, fact_b_id, conflict_type) VALUES (?,?,?)``.
With the unique index in place that statement raises IntegrityError inside
the consolidation loop — the partial-state class the ``_record_conflict``
docstring already flags (fact INSERT durable, later conflict-record failure
leaks partial state). The insert must gain
``ON CONFLICT DO NOTHING`` in the same change. A bare upsert target is
deliberate: it needs no expression repeated verbatim and covers every
uniqueness violation on the table.

Safety: creates an index only. No table is dropped, no row is rewritten,
and no row is deleted. If pre-existing duplicate pairs are found the index
is NOT created (SQLite would fail) — the migration reports them and leaves
the bank untouched for adjudication instead of guessing a winner.

Safe to re-run (``CREATE UNIQUE INDEX IF NOT EXISTS`` + existence check).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Literal, TypedDict, Union, overload


# Canonical DDL — mirrors the index that belongs beside
# ``CREATE TABLE IF NOT EXISTS conflicts`` in
# ``mnemosyne/core/veracity_consolidation.py``.
_INDEX_NAME = "idx_conflicts_pair_norm"
_INDEX_DDL = (
    f"CREATE UNIQUE INDEX IF NOT EXISTS {_INDEX_NAME} "
    "ON conflicts (min(fact_a_id, fact_b_id), max(fact_a_id, fact_b_id))"
)

# Normalized-pair predicate, reused for the pre-flight duplicate sweep.
_DUPLICATE_PAIRS_SQL = """
    SELECT CASE WHEN fact_a_id < fact_b_id
                THEN fact_a_id || '/' || fact_b_id
                ELSE fact_b_id || '/' || fact_a_id END AS pair,
           COUNT(*) AS n
      FROM conflicts
     GROUP BY pair
    HAVING n > 1
"""


class MigrationReport(TypedDict):
    """Outcome of a real (non-dry-run) E8 pass."""

    applied: bool
    conflicts_table_missing: bool
    index_already_present: bool
    index_added: bool
    duplicate_pairs: List[str]


class MigrationDryRunReport(MigrationReport):
    """Outcome of a dry run: reports what a real pass would do."""

    dry_run: Literal[True]


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _has_index(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _duplicate_pairs(conn: sqlite3.Connection) -> List[str]:
    try:
        rows = conn.execute(_DUPLICATE_PAIRS_SQL).fetchall()
    except sqlite3.OperationalError:
        # No conflicts table (or unreadable shape) — nothing to sweep.
        return []
    return [str(r[0]) for r in rows]


@overload
def migrate_conflict_pair_key(db_path: Path, dry_run: Literal[True]) -> MigrationDryRunReport: ...


@overload
def migrate_conflict_pair_key(
    db_path: Path, dry_run: Literal[False] = False
) -> MigrationReport: ...


def migrate_conflict_pair_key(
    db_path: Path, dry_run: bool = False
) -> Union[MigrationReport, MigrationDryRunReport]:
    """Add the order-normalized unique index to ``conflicts``.

    Idempotent; index-only; never writes or deletes a row. Returns a report
    instead of raising when the bank is not yet at the conflicts schema or
    when duplicate pairs make a unique index impossible today.
    """
    report: MigrationReport = {
        "applied": False,
        "conflicts_table_missing": False,
        "index_already_present": False,
        "index_added": False,
        "duplicate_pairs": [],
    }

    if not db_path.exists():
        report["conflicts_table_missing"] = True
        if dry_run:
            report["dry_run"] = True  # type: ignore[typeddict-item]
        return report

    conn = sqlite3.connect(str(db_path))
    try:
        if not _has_table(conn, "conflicts"):
            report["conflicts_table_missing"] = True
            return report  # type: ignore[return-value]

        if _has_index(conn, _INDEX_NAME):
            report["index_already_present"] = True
            report["applied"] = True
            return report  # type: ignore[return-value]

        duplicates = _duplicate_pairs(conn)
        if duplicates:
            # A unique index cannot be created over duplicates. Report and
            # leave the bank untouched: choosing a winner is adjudication,
            # not migration.
            report["duplicate_pairs"] = duplicates
            return report  # type: ignore[return-value]

        if dry_run:
            report["index_added"] = True
            report["applied"] = True
            report["dry_run"] = True  # type: ignore[typeddict-item]
            return report  # type: ignore[return-value]

        conn.execute(_INDEX_DDL)
        conn.commit()
        report["index_added"] = True
        report["applied"] = True
    finally:
        conn.close()
    return report
