"""Trackwork and vet records of every horse on a race card, saved before the race.

    python -m hkjc.trackwork    # upcoming cards -> data/trackwork/<date>_<venue>.json

Not a model factor. HKJC only keeps these records for active horses, so testing them
on past races would flatter the model with survivors. Each file here holds what was
public before that race for all its runners, so in a few months they can be tested
fairly. HKJC's per-race pages show about the last 18 days of work and every vet record.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date, datetime, timezone

from bs4 import BeautifulSoup

from . import scraper, store
from .backfill import HKT, today_hkt
from .reports import _clean, _matches

FOLDER = store.DATA / "trackwork"
WORK_PAGE = "/zh-hk/local/information/localtrackwork"
VET_PAGE = "/zh-hk/local/information/veterinaryrecord"
HORSE_CODE = re.compile(r"horseid=[A-Za-z]+_\d+_([A-Z]\d{3})")


def _head_cells(tr) -> list[str]:
    return [re.sub(r"\s+", "", c.get_text()) for c in tr.find_all(["td", "th"])]


def _plain_tables(html: str):
    """Data tables only: a layout table holding other tables would repeat their rows."""
    return [t for t in BeautifulSoup(html, "html.parser").find_all("table") if not t.find("table")]


def _full_date(dm: str, on: date) -> str:
    """'05/11' has no year: it is the latest such day up to the day we fetched it."""
    day, month = int(dm[:2]), int(dm[3:])
    x = date(on.year, month, day)
    return (x if x <= on else date(on.year - 1, month, day)).isoformat()


def _entries(td, on: date) -> list[list[str]]:
    """'05/11: 沙田 全天候 29.9 26.8 (56.7) (周俊樂)' lines -> [[iso date, text], ...]"""
    parts = re.split(r"(\d{2}/\d{2}):", _clean(td))
    return [[_full_date(d, on), text.strip()] for d, text in zip(parts[1::2], parts[2::2])]


def parse_work(html: str, on: date) -> dict:
    """{horse code: {...}} from a race's trackwork page, reserves included."""
    horses = {}
    for table in _plain_tables(html):
        rows = table.find_all("tr")
        at = next((i for i, tr in enumerate(rows[:3]) if "快操" in _head_cells(tr)), None)
        if at is None:
            continue
        heads = _head_cells(rows[at])
        reserve = "後備" in re.sub(r"\s+", "", _clean(rows[0]))
        for tr in rows[at + 1:]:
            tds = tr.find_all("td")
            code = HORSE_CODE.search(str(tr))
            if not code or len(tds) < len(heads):
                continue
            work = {h: _entries(td, on) for h, td in zip(heads[2:-1], tds[2:-1])}
            horses[code.group(1)] = {"no": _clean(tds[0]), "horse": _clean(tr.find("a")),
                                     "reserve": reserve, "work": {h: e for h, e in work.items() if e},
                                     "vet": []}
    return horses


def parse_vet(html: str) -> dict:
    """{horse code: [[date, detail, date passed], ...]}. Horses with no record aren't listed."""
    out: dict = {}
    for table in _plain_tables(html):
        if not any("通過日期" in _head_cells(tr) for tr in table.find_all("tr")[:3]):
            continue
        code = None
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            found = HORSE_CODE.search(str(tr))
            code = found.group(1) if found else code   # later rows of a horse leave these cells blank
            if len(tds) != 5 or not code or not re.fullmatch(r"\d{2}/\d{2}/\d{4}", _clean(tds[2])):
                continue
            iso = lambda s: datetime.strptime(s, "%d/%m/%Y").date().isoformat() if s else ""
            out.setdefault(code, []).append([iso(_clean(tds[2])), _clean(tds[3]), iso(_clean(tds[4]))])
    return out


def fetch_race(day: str, venue: str, race: dict, on: date, log=print) -> dict | None:
    params = {"RaceDate": day.replace("-", "/"), "Racecourse": venue, "RaceNo": race["race_no"]}
    horses = parse_work(scraper.get(WORK_PAGE, **params), on)
    if not _matches(horses, race["runners"]):   # HKJC falls back to another race on a bad request
        log(f"  ! R{race['race_no']}: trackwork page is about other horses")
        return None
    vet = parse_vet(scraper.get(VET_PAGE, **params))
    vet_ok = not set(vet) - set(horses)
    if vet_ok:
        for code, rows in vet.items():
            horses[code]["vet"] = rows
    else:
        log(f"  ! R{race['race_no']}: vet page is about other horses")
    return {"fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "vet_ok": vet_ok, "horses": horses}


def update_cards(log=print) -> tuple[int, int]:
    """Save every race not yet run on the stored cards; returns (saved, failed)."""
    FOLDER.mkdir(parents=True, exist_ok=True)
    today = today_hkt()
    clock = datetime.now(HKT).strftime("%H:%M")
    saved = failed = 0
    for p in sorted(store.CARDS.glob("20*_*.json")):
        card = store.read_json(p)
        if card["date"] < today.isoformat():
            continue
        path = FOLDER / f"{card['date']}_{card['venue']}.json"
        doc = store.read_json(path) or {"date": card["date"], "venue": card["venue"], "races": {}}
        for race in card["races"]:
            # once a race has run, the page may show what came after it, so its last copy stays
            if card["date"] == today.isoformat() and (race.get("post_time") or "99:99") <= clock:
                continue
            try:
                got = fetch_race(card["date"], card["venue"], race, today, log)
            except (RuntimeError, ValueError) as e:
                log(f"  ! R{race['race_no']}: {e}")
                got = None
            if got:
                doc["races"][str(race["race_no"])] = got
                saved += 1
            else:
                failed += 1
        # one meeting is about 500 KB with the usual indenting, too much to add every few days
        path.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        log(f"  {path.stem}: {len(doc['races'])}/{len(card['races'])} races, "
            f"{sum(len(r['horses']) for r in doc['races'].values())} horses")
    return saved, failed


if __name__ == "__main__":
    ok, bad = update_cards()
    print(f"saved {ok} races, {bad} failed")
    sys.exit(1 if bad and not ok else 0)
