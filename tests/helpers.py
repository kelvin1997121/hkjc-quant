"""Small builders for made-up race cards, results, odds and analysis files."""
from __future__ import annotations

import os
import shutil
from datetime import datetime, timedelta, timezone

import pytest

HKT = timezone(timedelta(hours=8))


def hkt(day: str, clock: str) -> datetime:
    """'2026-11-14', '12:29:59' -> that moment in Hong Kong."""
    return datetime.fromisoformat(f"{day}T{clock}").replace(tzinfo=HKT)


def utc_iso(day: str, clock: str) -> str:
    """The same moment as the UTC timestamp string the system stores."""
    return hkt(day, clock).astimezone(timezone.utc).isoformat(timespec="seconds")


def code(race_no: int, no: int) -> str:
    return f"{chr(64 + race_no)}{no:03d}"       # race 1 horse 3 -> A003


def card_race(day, venue, race_no, post, n=8):
    runners = [{"no": i, "horse": f"馬{race_no}-{i}", "horse_code": code(race_no, i), "draw": i,
                "weight": 118 + i, "handicap_weight": 118 + i, "allowance": 0,
                "jockey": f"騎師{i}", "jockey_id": f"J{i:02d}", "trainer": f"練馬師{i}", "trainer_id": f"T{i:02d}",
                "rating": 60 - i, "body_weight": 1100 + i, "last6": "", "gear": "", "scratched": False}
               for i in range(1, n + 1)]
    return {"date": day, "venue": venue, "race_no": race_no, "race_name": f"第{race_no}場測試讓賽",
            "post_time": post, "distance": 1200, "surface": "TURF", "course": "A", "going": "好地",
            "rating_band": "60-40", "class": "第四班", "runners": runners}


def make_card(day, venue, posts: dict, n=8):
    """posts: {race_no: 'HH:MM'}"""
    return {"date": day, "venue": venue, "fetched_at": f"{day}T00:00:00Z",
            "races": [card_race(day, venue, no, post, n) for no, post in sorted(posts.items())]}


def make_result(race, order, odds=None, codes=None, dividends="auto"):
    """A result for a card race. order: horse numbers in finishing order; codes: {no: 'WV'} for
    horses with a code instead of a place; odds: {no: final win odds} (default 3 + number)."""
    codes = codes or {}
    odds = odds or {x["no"]: 3.0 + x["no"] for x in race["runners"]}
    place = {no: i for i, no in enumerate(order, 1)}
    runners = []
    for x in race["runners"]:
        no = x["no"]
        p = place.get(no)
        runners.append({"place_raw": str(p) if p else codes.get(no, ""), "place": p, "no": no,
                        "horse": x["horse"], "horse_code": x["horse_code"], "jockey": x["jockey"],
                        "jockey_id": x["jockey_id"], "trainer": x["trainer"], "trainer_id": x["trainer_id"],
                        "weight": x["weight"], "body_weight": x["body_weight"], "draw": x["draw"], "lbw": "",
                        "running": [p, p, p] if p else [], "time": round(69.5 + 0.12 * p, 2) if p else None,
                        "win_odds": odds.get(no) if p or codes.get(no, "").upper() in ("PU", "DNF", "UR", "FE") else None})
    runners.sort(key=lambda r: r["place"] or 99)
    if dividends == "auto":
        a, b, c = order[:3]
        pair = lambda x, y: f"{min(x, y)},{max(x, y)}"
        dividends = {"獨贏": [[str(a), round(odds[a] * 10, 1)]],
                     "位置": [[str(a), 15.0], [str(b), 22.0], [str(c), 31.0]],
                     "連贏": [[pair(a, b), 88.0]],
                     "位置Q": [[pair(a, b), 33.0], [pair(a, c), 44.0], [pair(b, c), 55.0]]}
    return {"date": race["date"], "venue": race["venue"], "race_no": race["race_no"], "class": race["class"],
            "distance": race["distance"], "going": race["going"], "race_name": race["race_name"],
            "surface": race["surface"], "course": race["course"], "rating_band": race["rating_band"],
            "has_sect": True, "runners": runners, "dividends": dividends or {}}


def odds_read(race, fav, read_at, by=None):
    """What odds.fetch_odds returns for one race: `fav` at 1.9, the rest 6 + number.
    read_at: UTC timestamp string of the read. by: {no: odds} to set prices outright."""
    nos = [x["no"] for x in race["runners"]]
    win = dict(by) if by else {no: (1.9 if no == fav else 6.0 + no) for no in nos}
    pairs = [(a, b) for a in nos for b in nos if a < b]
    return {"updated": read_at, "win": win, "place": {no: round(o / 3 + 1, 1) for no, o in win.items()},
            "scratched": [], "qin": {f"{a}-{b}": 30.0 + a + b for a, b in pairs},
            "qpl": {f"{a}-{b}": 12.0 + a + b for a, b in pairs}, "t": read_at}


def analysis_race(race, order, win_odds=None, longshots=(), post_hoc=False):
    """A saved analysis of a card race, as the paper account reads it. order: numbers by rank."""
    win_odds = win_odds or {}
    names = {x["no"]: x["horse"] for x in race["runners"]}
    out = {"race_no": race["race_no"], "race_name": race["race_name"], "post_time": race["post_time"],
           "runners": [{"no": no, "horse": names[no], "rank": i, "final_rank": i, "win_prob": round(0.5 / i, 4),
                        "win_odds": win_odds.get(no), "place_odds": None, "longshot": no in longshots}
                       for i, no in enumerate(order, 1)]}
    if post_hoc:
        out["post_hoc"] = True
    return out


def analysis_doc(day, venue, races):
    return {"meeting": {"date": day, "venue": venue, "venue_zh": "沙田" if venue == "ST" else "跑馬地"},
            "updated_at": f"{day}T00:00:00+00:00", "races": races}


# ------------------------------------------------------------------ tools some tests need

# set in .github/workflows/tests.yml: there a missing tool is a failure, not a skipped test
REQUIRE_ALL = os.environ.get("HKJC_TESTS_REQUIRE_ALL") == "1"


def unavailable(what):
    if REQUIRE_ALL:
        pytest.fail(f"{what} is needed for this test and is missing")
    pytest.skip(f"{what} not available here")


def need(tool):
    """Path of a command-line tool, or skip the test (fail it in the cloud)."""
    return shutil.which(tool) or unavailable(tool)
