"""Running comments, race incident reports and barrier trials from HKJC's public pages.

    python -m hkjc.reports races    # comments and incidents into data/results
    python -m hkjc.reports trials   # trials into data/trials/<date>.json

These pages show the latest meeting when they have nothing for the requested date,
so every page is checked against the horses we already know ran before it is stored.
"""
from __future__ import annotations

import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

from bs4 import BeautifulSoup

from . import scraper, store
from .backfill import _out_of_time, today_hkt

TRIALS = store.DATA / "trials"
EMPTY = TRIALS / "_empty.json"
CHECKED = TRIALS / "_checked.json"
INCIDENT_CHARS = 300   # long stewards' enquiries: the start says what happened
CODE = re.compile(r"\(([A-Z]\d{3})\)")


def _clean(el) -> str:
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True))


def _rows_under(soup: BeautifulSoup, header: str):
    """Rows (as td lists) of every table whose short header row mentions `header`."""
    for t in soup.find_all("table"):
        first = t.find("tr")
        if not first or header not in _clean(first) or len(_clean(first)) > 200:
            continue
        for tr in t.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if tds:
                yield tds


def _by_code(soup, header, text_col=-1) -> dict:
    out = {}
    for tds in _rows_under(soup, header):
        for td in tds:
            m = CODE.search(_clean(td))
            if m:
                out[m.group(1)] = _clean(tds[text_col])
                break
    return out


def _matches(found: dict, runners: list) -> bool:
    """Is this page really about these runners? (not HKJC's fallback to the latest meeting)"""
    codes = {x.get("horse_code") for x in runners if x.get("horse_code")}
    return bool(codes) and len(codes & set(found)) >= 0.8 * len(codes)


def attach_comments(race: dict) -> bool:
    html = scraper.get("/zh-hk/local/information/corunning",
                       Date=race["date"].replace("-", "/"), RaceNo=race["race_no"])
    found = _by_code(BeautifulSoup(html, "html.parser"), "走勢評述")
    ok = _matches(found, race["runners"])
    if ok:
        for x in race["runners"]:
            if x.get("horse_code") in found:
                x["comment"] = found[x["horse_code"]]
    race["has_comment"] = ok
    return ok


def attach_incidents(races: list[dict]) -> bool:
    """One page covers the whole meeting. Only available from about January 2024."""
    html = scraper.get("/zh-hk/local/information/racereportfull", Date=races[0]["date"].replace("-", "/"))
    found = _by_code(BeautifulSoup(html, "html.parser"), "競賽事件")
    runners = [x for r in races for x in r["runners"]]
    ok = _matches(found, runners)
    if ok:
        for x in runners:
            if x.get("horse_code") in found:
                x["incident"] = found[x["horse_code"]][:INCIDENT_CHARS]
    for r in races:
        r["has_incident"] = ok
    return ok


def enrich_meeting(path, pool=None) -> bool:
    """Fill in whatever a stored meeting is missing; True if anything was fetched."""
    races = store.read_json(path, []) or []
    todo = [r for r in races if "has_comment" not in r]
    need_incident = bool(races) and "has_incident" not in races[0]
    if not todo and not need_incident:
        return False
    if todo:
        if pool:
            list(pool.map(attach_comments, todo))
        else:
            for r in todo:
                attach_comments(r)
    if need_incident:
        attach_incidents(races)
    store.write_json(path, races)
    return True


def enrich_races(log=print) -> int:
    files = sorted((p for p in store.RESULTS.glob("20*_*.json") if "NONE" not in p.name), reverse=True)
    done = 0
    with ThreadPoolExecutor(max_workers=scraper.CONCURRENCY) as pool:
        for p in files:
            if _out_of_time():
                break
            try:
                if enrich_meeting(p, pool):
                    done += 1
                    races = store.read_json(p, [])
                    log(f"  {p.stem}: comments {sum(r.get('has_comment', False) for r in races)}/{len(races)}, "
                        f"incidents {'yes' if races[0].get('has_incident') else 'no'}")
            except RuntimeError as e:
                log(f"  ! {p.stem}: {e}")
    return done


# ------------------------------------------------------------------------- text tags

WIDE_WORDS = ("走外疊", "沒有遮擋", "走三疊", "走四疊", "第三疊", "第四疊")


def ran_wide(x: dict) -> bool:
    """Did the race caller or stewards say this run was spent wide of the rail?"""
    text = (x.get("comment") or "") + " " + (x.get("incident") or "")
    return any(w in text for w in WIDE_WORDS)


# ------------------------------------------------------------------------- trials

MARGINS = {"NOSE": 0.05, "SH": 0.1, "HD": 0.2, "HEAD": 0.2, "NECK": 0.3, "N": 0.3}


def margin_lengths(s: str) -> float:
    """'', 'Neck', '1/2L', '1-1/4L', '12L' -> lengths behind the winner."""
    t = (s or "").upper().replace("L", "").strip()
    if not t or t in ("---", "--"):
        return 0.0
    if t in MARGINS:
        return MARGINS[t]
    m = re.fullmatch(r"(?:(\d+)-)?(\d+)/(\d+)", t)
    if m:
        return int(m.group(1) or 0) + int(m.group(2)) / int(m.group(3))
    return float(t) if re.fullmatch(r"\d+(\.\d+)?", t) else 0.0


def parse_trials(html: str, day: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for t in soup.find_all("table"):
        first = t.find("tr")
        if not first or "完成" not in _clean(first) or len(_clean(first)) > 200:
            continue
        heads = [re.sub(r"\s+", "", c.get_text()) for c in first.find_all(["td", "th"])]
        col = {h: i for i, h in enumerate(heads)}
        rows = [tr.find_all("td") for tr in t.find_all("tr")[1:]]
        rows = [r for r in rows if len(r) >= len(heads) - 1]
        heading = t.find_previous(string=re.compile(r"第\s*\d+\s*組")) or ""
        m = re.search(r"(\d{3,4})米", heading)
        for pos, tds in enumerate(rows, 1):
            code = CODE.search(_clean(tds[0]))
            if not code:
                continue
            out.append({"date": day, "code": code.group(1), "pos": pos, "field": len(rows),
                        "lengths": margin_lengths(_clean(tds[col["頭馬距離"]])),
                        "jockey": _clean(tds[col["騎師"]]), "gear": _clean(tds[col["馬匹配備"]]),
                        "dist": int(m.group(1)) if m else None, "venue": "HV" if "跑馬地" in heading else "ST",
                        "comment": _clean(tds[col["沿途走勢評述"]])})
    return out


def fetch_trial_day(day: date) -> tuple[date, list]:
    html = scraper.get("/zh-hk/local/information/btresult", Date=day.strftime("%Y/%m/%d"))
    return day, parse_trials(html, day.isoformat())


def update_trials(start: date, end: date, log=print) -> int:
    """Scan days not seen yet. The last few days are always re-read (trials post with a delay)."""
    empty = set(store.read_json(EMPTY, []) or [])
    recent = today_hkt() - timedelta(days=3)
    days = []
    d = end
    while d >= start:
        iso = d.isoformat()
        if d >= recent or (iso not in empty and not (TRIALS / f"{iso}.json").exists()):
            days.append(d)
        d -= timedelta(days=1)
    log(f"  {len(days)} days to scan")
    found = 0
    for i in range(0, len(days), 40):
        if _out_of_time():
            break
        with ThreadPoolExecutor(max_workers=scraper.CONCURRENCY) as pool:
            for day, rows in pool.map(fetch_trial_day, days[i:i + 40]):
                if rows:
                    store.write_json(TRIALS / f"{day.isoformat()}.json", rows)
                    found += 1
                elif day < recent:
                    empty.add(day.isoformat())
        store.write_json(EMPTY, sorted(empty))
        log(f"  scanned {min(i + 40, len(days))}/{len(days)}, {found} days with trials")
    return found


REFRESH_HOURS = 3


def refresh_recent(days=7, log=print) -> None:
    """New trials, plus comments and incident reports of meetings already run.

    Reports come out some time after a meeting, so finished meetings are retried for
    a week. At most every few hours: live runs come every 5 minutes on a race day.
    """
    now = datetime.now(timezone.utc)
    stamp = store.read_json(CHECKED, {}) or {}
    last = stamp.get("at")
    if last and now - datetime.fromisoformat(last) < timedelta(hours=REFRESH_HOURS):
        return
    today = today_hkt()
    try:
        update_trials(today - timedelta(days=days), today, log=lambda m: None)
        for p in sorted(store.RESULTS.glob("20*_*.json")):
            if "NONE" in p.name:
                continue
            d = date.fromisoformat(p.stem[:10])
            if not (0 < (today - d).days <= days):
                continue
            races = store.read_json(p, []) or []
            changed = False
            for r in races:
                if not r.get("has_comment"):
                    changed = attach_comments(r) or changed
            if races and not races[0].get("has_incident"):
                changed = attach_incidents(races) or changed
            if changed:
                store.write_json(p, races)
                log(f"   reports added for {p.stem}")
    except RuntimeError as e:   # HKJC hiccup: today's analysis still runs on what we have
        log(f"   reports refresh skipped: {e}")
        return
    store.write_json(CHECKED, {"at": now.isoformat(timespec="seconds")})


def load_trials() -> dict:
    """{horse code: [trial, ...]} oldest first."""
    out: dict = {}
    for p in sorted(TRIALS.glob("20*.json")):
        for row in store.read_json(p, []) or []:
            out.setdefault(row["code"], []).append(row)
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "races"
    if what == "trials":
        print("days with trials:", update_trials(date(2022, 7, 1), today_hkt()))
    else:
        print("meetings enriched:", enrich_races())
