"""Regression: provider construction must never trip its own fail-closed guard.

FAMSERVER 2026-10-10: every real Hermes launch pins HERMES_HOME, so
_p1b_current_key() answers with the home's key from the very first line of
__init__. The old code wrote self._beam (a binding property) before the
_bindings registry existed; _write_slot() fail-closed on the unregistered
home key and __init__ raised "no binding for the current home" — the
provider could never even be constructed on a real deployment. The guard is
correct; construction was the bug. Seed the registry, including the
construction-time home key, before the first property write.
"""

from __future__ import annotations

import mnemosyne_hermes
from mnemosyne_hermes import MnemosyneMemoryProvider


def test_construction_under_turn_scope_home_key(monkeypatch):
    monkeypatch.setattr(mnemosyne_hermes, "_p1b_current_key", lambda: "/real/HERMES/HOME")
    monkeypatch.setattr(mnemosyne_hermes, "_p1b_home_key", lambda h=None: h or "/real/HERMES/HOME")
    p = MnemosyneMemoryProvider()  # must not raise
    assert p._beam is None  # reads resolve through the seeded slot


def test_construction_out_of_turn_unchanged(monkeypatch):
    monkeypatch.setattr(mnemosyne_hermes, "_p1b_current_key", lambda: None)
    monkeypatch.setattr(mnemosyne_hermes, "_p1b_home_key", lambda h=None: h or "default")
    p = MnemosyneMemoryProvider()
    assert p._beam is None


def test_first_write_in_turn_lands_in_own_home_slot(monkeypatch):
    monkeypatch.setattr(mnemosyne_hermes, "_p1b_current_key", lambda: "/home/a")
    monkeypatch.setattr(mnemosyne_hermes, "_p1b_home_key", lambda h=None: h or "/home/a")
    p = MnemosyneMemoryProvider()
    p._beam = object()
    assert p.__dict__["_bindings"]["/home/a"]["beam"] is p._beam
