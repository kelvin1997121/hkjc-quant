"""Download historical results that are not stored yet."""
from __future__ import annotations

import sys
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


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    print("added", backfill(n), "meetings")
