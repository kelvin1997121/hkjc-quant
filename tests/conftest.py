"""Shared fixtures.

Every test that touches files runs in a temporary data/ and docs/ folder (the `sandbox`
fixture). No test reads the network, and none writes the repo's own data/ or docs/.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from hkjc import horses, model, reports, selfcheck, store, trackwork

FIXTURES = Path(__file__).parent / "fixtures"

# every module-level path that points into data/ or docs/ (test_sandbox.py checks none is missed)
REDIRECTS = [
    (store, "DATA", "data"), (store, "RESULTS", "data/results"), (store, "CARDS", "data/racecards"),
    (store, "AGENT_CACHE", "data/agent_cache"), (store, "DOCS", "docs"),
    (horses, "PATH", "data/horses.json"),
    (reports, "TRIALS", "data/trials"), (reports, "EMPTY", "data/trials/_empty.json"),
    (reports, "CHECKED", "data/trials/_checked.json"),
    (trackwork, "FOLDER", "data/trackwork"),
    (selfcheck, "LEARN_PATH", "data/learning.json"),
]


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """An empty copy of the repo's folder layout; the hkjc modules are pointed at it."""
    for sub in ("data/results", "data/racecards", "data/odds", "data/trials", "data/trackwork", "docs/data"):
        (tmp_path / sub).mkdir(parents=True)
    for module, name, rel in REDIRECTS:
        monkeypatch.setattr(module, name, tmp_path / rel)
    # model.train and the self-check swap this list; start every test from the base factors
    monkeypatch.setattr(model, "FEATURES", list(model.BASE_FEATURES))
    return tmp_path


@pytest.fixture
def meeting_1111(sandbox):
    """The 11 Nov 2026 Happy Valley meeting as it was saved: frozen analysis, results, race
    card, and the odds snapshots read before each post time."""
    stem, src = "2026-11-11_HV", FIXTURES / "2026-11-11_HV"
    for rel, name in (("docs/data", "analysis.json"), ("data/results", "results.json"),
                      ("data/racecards", "card.json"), ("data/odds", "odds.json")):
        shutil.copy(src / name, sandbox / rel / f"{stem}.json")
    return sandbox
