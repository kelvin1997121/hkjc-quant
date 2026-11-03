"""Download historical results that are not stored yet."""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

from . import scraper, store

HKT = timezone(timedelta(hours=8))


def today_hkt() -> date:
    return datetime.now(HKT).date()


def backfill(max_meetings: int = 120, log=print) -> int:
    """Fetch up to `max_meetings` most recent meetings that are missing locally."""
    dates = [d for d in scraper.list_result_dates() if d <= today_hkt()]
    todo = [d for d in dates[:max_meetings] if not store.have_results(d.isoformat())]
    added = 0
    for i, d in enumerate(todo, 1):
        try:
            races = scraper.fetch_meeting_results(d)
        except RuntimeError as e:
            log(f"  ! {d}: {e}")
            continue
        if races:
            store.save_results(races)
            added += 1
        elif d < today_hkt():  # overseas simulcast date: remember it so we don't refetch
            store.write_json(store.result_path(d.isoformat(), "NONE"), [])
        log(f"  [{i}/{len(todo)}] {d} {races[0]['venue'] if races else '-'} {len(races)} races")
    return added


# optional wall-clock budget so cloud jobs can stop cleanly and commit what they have
_DEADLINE = time.time() + float(os.environ["BACKFILL_MINUTES"]) * 60 \
    if os.environ.get("BACKFILL_MINUTES") else None


def _out_of_time() -> bool:
    return _DEADLINE is not None and time.time() > _DEADLINE


def enrich_sectionals(log=print, newest_first=True) -> int:
    """Add per-horse sectional times to stored races that don't have them yet."""
    files = sorted((p for p in store.RESULTS.glob("20*_*.json") if "NONE" not in p.name),
                   reverse=newest_first)
    done = 0
    with ThreadPoolExecutor(max_workers=scraper.CONCURRENCY) as pool:
        for p in files:
            races = store.read_json(p, [])
            todo = [r for r in races if "has_sect" not in r]
            if not todo:
                continue
            if _out_of_time():
                break
            list(pool.map(scraper.attach_sectionals, todo))  # races of a meeting in parallel
            store.write_json(p, races)
            done += 1
            log(f"  sectionals {p.stem}: {sum(r.get('has_sect', False) for r in races)}/{len(races)}")
    return done


def _probed_files():
    return sorted(store.DATA.glob("probed_dates*.json"))


def backfill_range(start: date, end: date, log=print) -> int:
    """Probe older dates that the results selector no longer lists.

    HK meetings are almost always on Wed / Sat / Sun, with a summer break
    from late July to early September, so only those days are probed.
    Dates are probed several at a time (see HKJC_CONCURRENCY / HKJC_RATE).
    """
    probed = set()
    for f in _probed_files():  # shared knowledge from every job
        probed.update(store.read_json(f, []))
    # each range writes its own file so parallel cloud jobs never overwrite each other
    mine_path = store.DATA / f"probed_dates_{start.isoformat()}.json"
    mine = set(store.read_json(mine_path, []))
    todo = []
    d = end
    while d >= start:
        iso = d.isoformat()
        summer = (d.month == 7 and d.day > 20) or d.month == 8
        if d.weekday() in (2, 5, 6) and not summer and iso not in probed \
                and not store.have_results(iso):
            todo.append(d)
        d -= timedelta(days=1)
    log(f"  {len(todo)} dates to probe in {start}..{end}")

    def probe(day):
        if _out_of_time():
            return day, None
        try:
            return day, scraper.fetch_meeting_results(day)
        except RuntimeError as e:
            log(f"  ! {day}: {e}")
            return day, None

    added = 0
    with ThreadPoolExecutor(max_workers=scraper.CONCURRENCY) as pool:
        for day, races in pool.map(probe, todo):
            if races:
                store.save_results(races)
                added += 1
                log(f"  {day} {races[0]['venue']} {len(races)} races")
            elif races is not None:
                mine.add(day.isoformat())
                store.write_json(mine_path, sorted(mine))
    return added


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sectionals":
        print("enriched", enrich_sectionals(), "meetings")
        sys.exit()
    if len(sys.argv) > 2 and sys.argv[1] == "range":
        s, e = (date.fromisoformat(x) for x in sys.argv[2:4])
        print("added", backfill_range(s, e), "meetings")
        sys.exit()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    print("added", backfill(n), "meetings")
