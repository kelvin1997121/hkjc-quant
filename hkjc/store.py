"""Flat-file storage: one JSON file per meeting, committed to git by the workflow."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# HKJC_DATA_DIR points everything at another data folder (the old-season test bank, see
# hkjc/oldbank.py). Unset in every live workflow, so live runs always use data/.
DATA = Path(os.environ["HKJC_DATA_DIR"]).resolve() if os.environ.get("HKJC_DATA_DIR") else ROOT / "data"
RESULTS = DATA / "results"
CARDS = DATA / "racecards"
AGENT_CACHE = DATA / "agent_cache"
CANCELLED = DATA / "cancelled.json"   # {"2026-11-14_ST": [6]}: races HKJC declared cancelled
DOCS = ROOT / "docs"


def _write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def _read(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def result_path(date_iso: str, venue: str) -> Path:
    return RESULTS / f"{date_iso}_{venue}.json"


def save_results(races: list[dict]):
    if races:
        _write(result_path(races[0]["date"], races[0]["venue"]), races)


def have_results(date_iso: str) -> bool:
    return any(RESULTS.glob(f"{date_iso}_*.json"))


def load_all_results() -> list[dict]:
    """Every stored race, chronologically."""
    races = []
    for p in sorted(RESULTS.glob("*.json")):
        races.extend(_read(p, []))
    races.sort(key=lambda r: (r["date"], r["race_no"]))
    return races


def cancelled(date_iso: str, venue: str) -> set[int]:
    """Race numbers of a meeting that HKJC declared cancelled (every single-race pool refunded).

    Such a race has no result and never will. It is kept apart from data/results, which holds
    races that were run, so nothing that learns from history ever sees it."""
    return set((_read(CANCELLED, {}) or {}).get(f"{date_iso}_{venue}", []))


def set_cancelled(date_iso: str, venue: str, nos) -> bool:
    """Record a meeting's cancelled races (an empty set forgets the meeting). True if changed."""
    book = _read(CANCELLED, {}) or {}
    key, new = f"{date_iso}_{venue}", sorted(set(nos))
    if book.get(key, []) == new:
        return False
    if new:
        book[key] = new
    else:
        book.pop(key, None)
    _write(CANCELLED, dict(sorted(book.items())))
    return True


def save_card(card: dict):
    path = CARDS / f"{card['date']}_{card['venue']}.json"
    old = _read(path)
    if old and old.get("races") == json.loads(json.dumps(card["races"], ensure_ascii=False)):
        return  # unchanged apart from fetch time: keep git quiet
    _write(path, card)


def load_card(date_iso: str, venue: str):
    return _read(CARDS / f"{date_iso}_{venue}.json")


def write_json(path: Path, obj):
    _write(path, obj)


def read_json(path: Path, default=None):
    return _read(path, default)
