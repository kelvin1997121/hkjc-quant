"""Official rating and gear for every horse, from its public profile page.

Result pages don't carry ratings or gear, so past races get them from here:

    python -m hkjc.horses            # fetch horses not stored yet, write into data/results

New races get them from the race card instead (see attach_card_fields), so this only
needs to run once. Pedigree per horse is kept in data/horses.json: not a model factor
(sire record showed no walk-forward gain), but it was costly to collect.
"""
from __future__ import annotations

import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from bs4 import BeautifulSoup

from . import scraper, store
from .backfill import _out_of_time, today_hkt

PATH = store.DATA / "horses.json"
CARD_PROFILE = {"sex": "sex", "age": "age", "sire": "sire", "dam": "dam", "import_type": "import_type"}


def parse_horse(html: str, code: str) -> dict | None:
    soup = BeautifulSoup(html, "html.parser")
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    if "父系" not in text:
        return None

    def find(pattern):
        m = re.search(pattern, text)
        return m.group(1).strip() if m else None

    age = find(r"馬齡\s*:\s*[^/]+/\s*(\d+)")
    profile = {
        "country": find(r"出生地(?:\s*/\s*馬齡)?\s*:\s*([^/\s]+)"),
        "age": int(age) if age else None,
        "age_on": today_hkt().isoformat(),
        "sex": find(r"毛色\s*/\s*性別\s*:\s*\S+\s*/\s*(\S+)"),
        "import_type": find(r"進口類別\s*:\s*(\S+)"),
        "sire": find(r"父系\s*:\s*(.+?)\s*母系\s*:"),
        "dam": find(r"母系\s*:\s*(.+?)\s*外祖父\s*:"),
        "damsire": find(r"外祖父\s*:\s*(.+?)\s*同父系馬"),
    }
    runs = {}
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue
        heads = [re.sub(r"\s+", "", c.get_text()) for c in rows[0].find_all(["td", "th"])]
        if "配備" not in heads or "評分" not in heads or "日期" not in heads:
            continue
        col = {h: i for i, h in enumerate(heads)}
        for tr in rows[1:]:
            cells = [re.sub(r"\s+", " ", c.get_text(" ", strip=True)) for c in tr.find_all("td")]
            if len(cells) < len(heads) - 1:
                continue    # season header rows
            raw = cells[col["日期"]]
            try:   # active horses print 2-digit years, retired horses 4-digit
                d = datetime.strptime(raw, "%d/%m/%Y" if len(raw) == 10 else "%d/%m/%y").date().isoformat()
            except ValueError:
                continue
            rating = cells[col["評分"]]
            gear = cells[col["配備"]]
            runs[d] = {"rating": int(rating) if rating.isdigit() else None,
                       "gear": "" if gear in ("--", "-") else gear}
        break
    return {"code": code, "profile": profile, "runs": runs}


def fetch_horse(code: str) -> dict | None:
    try:
        # Option=1: all past runs (the default view only goes back a few seasons)
        return parse_horse(scraper.get("/zh-hk/local/information/horse", HorseNo=code, Option=1), code)
    except RuntimeError as e:
        print(f"  ! {code}: {e}")
        return None


def apply_runs(fetched: list[dict]) -> int:
    """Write official rating and gear into the stored results of these horses."""
    by_code = {h["code"]: h["runs"] for h in fetched}
    changed = 0
    for p in sorted(store.RESULTS.glob("20*_*.json")):
        races = store.read_json(p, []) or []
        dirty = False
        for r in races:
            for x in r["runners"]:
                run = by_code.get(x.get("horse_code"), {}).get(r["date"])
                if run:
                    x["rating"], x["gear"] = run["rating"], run["gear"]
                    dirty = True
        if dirty:
            store.write_json(p, races)
            changed += 1
    return changed


def backfill_horses(log=print) -> int:
    known = store.read_json(PATH, {}) or {}
    codes = sorted({x["horse_code"] for r in store.load_all_results() for x in r["runners"]
                    if x.get("horse_code")} - set(known))
    log(f"  {len(codes)} horses to fetch ({len(known)} stored)")
    done = 0
    for i in range(0, len(codes), 100):
        if _out_of_time():
            break
        with ThreadPoolExecutor(max_workers=scraper.CONCURRENCY) as pool:
            batch = [h for h in pool.map(fetch_horse, codes[i:i + 100]) if h]
        apply_runs(batch)
        for h in batch:
            known[h["code"]] = h["profile"]
        store.write_json(PATH, known)
        done += len(batch)
        log(f"  {done}/{len(codes)} horses")
    return done


def attach_card_fields(races: list[dict], card: dict) -> None:
    """Carry official rating and gear from the race card onto results (no extra requests)."""
    cards = {r["race_no"]: {x["no"]: x for x in r["runners"]} for r in card.get("races", [])}
    for r in races:
        runners = cards.get(r["race_no"], {})
        for x in r["runners"]:
            c = runners.get(x.get("no"))
            if c and c.get("horse_code") == x.get("horse_code"):
                x["rating"], x["gear"] = c.get("rating"), c.get("gear") or ""


def remember_card_horses(card: dict) -> None:
    """Keep pedigree of horses first seen on a race card (new horses need no page fetch)."""
    known = store.read_json(PATH, {}) or {}
    changed = False
    for r in card.get("races", []):
        for x in r["runners"]:
            code = x.get("horse_code")
            if code and code not in known:
                known[code] = {**{k: x.get(v) for k, v in CARD_PROFILE.items()},
                               "age_on": card["date"]}
                changed = True
    if changed:
        store.write_json(PATH, known)


if __name__ == "__main__":
    print("fetched", backfill_horses(), "horses")
    sys.exit()
