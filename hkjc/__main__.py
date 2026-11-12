"""Pipeline entry point.

    python -m hkjc run            # one full update (what the scheduled workflow runs)
    python -m hkjc results        # race day: new results onto the page first (fast step)
    python -m hkjc backfill 120   # download up to 120 past meetings
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone

import numpy as np

from . import agent, analyze, horses, model, paper, reports, scraper, store, writer
from .backfill import HKT, backfill, today_hkt


def _result_summary(res_race):
    top = sorted([x for x in res_race["runners"] if x.get("place")], key=lambda x: x["place"])[:3]
    return {"placings": [x["no"] for x in top],
            "win_odds": top[0].get("win_odds") if top else None,
            "times": res_race.get("sectionals")}


def has_running(r):
    return any(x.get("running") for x in r["runners"] if x.get("place"))


def complete_recent_results(days=4):
    """Re-fetch recent meetings whose stored results are missing races.

    Live runs save partial results during a meeting; if the last run of the
    day happened before the last race (GitHub cron is often late), the
    backfill would otherwise treat the date as done forever.
    """
    today = today_hkt()
    current = (store.read_json(store.DOCS / "data" / "schedule.json", {}) or {}).get("date")
    for p in sorted(store.CARDS.glob("20*_*.json")):
        card = store.read_json(p)
        d = date.fromisoformat(card["date"])
        if not (0 <= (today - d).days <= days):
            continue
        if d == today and card["date"] == current:   # run() reads this meeting's results itself
            continue
        have = store.read_json(store.result_path(card["date"], card["venue"]), [])
        if len(have) < len(card["races"]):
            res = scraper.fetch_meeting_results(d)
            if len(res) > len(have):
                horses.attach_card_fields(res, card)
                store.save_results(res)
                print(f"   completed {card['date']}: {len(have)} -> {len(res)} races")
        elif d < today and any(not r.get("has_sect") for r in have):
            # sectional times are sometimes published a while after the result
            for r in have:
                if not r.get("has_sect"):
                    scraper.attach_sectionals(r)
            store.save_results(have)
        elif d < today and any(not has_running(r) for r in have):
            # so are running positions (8 Nov race 5 was first posted without them)
            res = scraper.fetch_meeting_results(d, venue=card["venue"],
                                                race_nos=[r["race_no"] for r in have],
                                                have={r["race_no"]: r for r in have if has_running(r)})
            got = {r["race_no"]: r for r in res if has_running(r)}
            if got:
                fixed = [got.get(r["race_no"], r) for r in have]
                horses.attach_card_fields(fixed, card)
                store.save_results(fixed)
                print(f"   running positions added for {card['date']} races {sorted(got.keys() - {r['race_no'] for r in have if has_running(r)})}")
        elif any("rating" not in x for r in have for x in r["runners"]):
            # saved by the daily catch-up before the race card fields were merged in
            horses.attach_card_fields(have, card)
            store.save_results(have)


def validate_card(card):
    """Fail loudly (workflow goes red, GitHub emails you) if parsing looks broken."""
    problems = []
    for r in card["races"]:
        live = [x for x in r["runners"] if not x.get("scratched")]
        if len(live) < 4:
            problems.append(f"R{r['race_no']}: only {len(live)} runners parsed")
        if not r.get("distance"):
            problems.append(f"R{r['race_no']}: distance missing")
        if sum(1 for x in live if x.get("draw")) < len(live) * 0.8:
            problems.append(f"R{r['race_no']}: draws missing")
    if problems:
        raise RuntimeError("race card parse check failed: " + "; ".join(problems))


def write_if_changed(path, doc):
    """Skip rewriting when only the timestamp would change (avoids a commit every 20 min)."""
    old = store.read_json(path)
    if old and {k: v for k, v in old.items() if k != "updated_at"} == \
            json.loads(json.dumps({k: v for k, v in doc.items() if k != "updated_at"})):
        return False
    store.write_json(path, doc)
    return True


def race_schedule(card, resulted=()):
    """Post times and races already resulted, read by the cloud ticker: an update every
    5 minutes during a meeting, every 2 while a race's result is due."""
    posts = sorted(r["post_time"] for r in card["races"] if r.get("post_time"))
    return {"date": card["date"], "venue": card["venue"],
            "first_post": posts[0] if posts else None, "last_post": posts[-1] if posts else None,
            "posts": {str(r["race_no"]): r["post_time"] for r in card["races"] if r.get("post_time")},
            "resulted": sorted(resulted)}


def _minutes(hhmm):
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def now_hkt():
    return datetime.now(HKT)


def started_races(card, now=None):
    """Races already off: post time passed on race day, or every race of a past meeting."""
    now = now or now_hkt()
    day = date.fromisoformat(card["date"])
    if day > now.date():
        return set()
    if day < now.date():
        return {r["race_no"] for r in card["races"]}
    mins = now.hour * 60 + now.minute
    return {r["race_no"] for r in card["races"]
            if r.get("post_time") and _minutes(r["post_time"]) <= mins}


def live_results(card, race_nos=None):
    """Fetch today's results (all races on the card, or just race_nos) and store them.
    Returns the race numbers that now have a result."""
    path = store.result_path(card["date"], card["venue"])
    # Before a race has run, HKJC can answer with another meeting's race of the same number
    # (8 Nov: 4 Nov's race 1 was stored as today's 25 minutes before the off). A result
    # counts only if it is about the horses on today's card for that race.
    entered = {r["race_no"]: {x.get("horse_code") for x in r["runners"] if x.get("horse_code")}
               for r in card["races"]}

    def ours(r):
        want = entered.get(r["race_no"], set())
        return not want or len(want & {x.get("horse_code") for x in r["runners"]}) >= 0.6 * len(want)

    stored = store.read_json(path, []) or []
    wrong = [r["race_no"] for r in stored if not ours(r)]
    stored = [r for r in stored if ours(r)]
    # a race posted before its running positions or dividends is fetched again
    have = {r["race_no"]: r for r in stored
            if r.get("has_sect") and r.get("dividends") and has_running(r)}
    live = scraper.fetch_meeting_results(
        date.fromisoformat(card["date"]), venue=card["venue"],
        race_nos=race_nos or [r["race_no"] for r in card["races"]], have=have)
    wrong += [r["race_no"] for r in live if not ours(r)]
    live = [r for r in live if ours(r)]
    if wrong:
        print(f"   ignored results about other horses for races {sorted(set(wrong))}")
    # a race HKJC can't serve right now keeps its stored result
    got = {r["race_no"] for r in live}
    live = sorted(live + [r for r in stored if r["race_no"] not in got], key=lambda r: r["race_no"])
    horses.attach_card_fields(live, card)
    if live or wrong:
        store.write_json(path, live)
    return {r["race_no"] for r in live}


def quick_results(now=None):
    """Race day: put new results on the page as soon as HKJC shows them, before the full
    update reads odds and retrains. Asks only for races whose post time has passed and
    that have no result yet. Never fails: the full run follows in the same job."""
    sched = store.read_json(store.DOCS / "data" / "schedule.json", {}) or {}
    if sched.get("date") != today_hkt().isoformat():
        print("results: no meeting today")
        return False
    card = store.load_card(sched["date"], sched["venue"])
    if not card:
        print("results: no stored race card")
        return False
    now = now or now_hkt()
    mins = now.hour * 60 + now.minute
    path = store.result_path(card["date"], card["venue"])
    done = {r["race_no"] for r in store.read_json(path, []) or []}
    due = [r["race_no"] for r in card["races"] if r.get("post_time") and r["race_no"] not in done
           and _minutes(r["post_time"]) + 3 <= mins]
    if not due:
        print("results: none due")
        return False
    try:
        finished = live_results(card, race_nos=due)
    except Exception as e:  # the full run will try again
        print(f"results: {e!r}")
        return False
    new = finished - done
    if not new:
        print(f"results: not out yet for races {due}")
        return False
    doc_path = store.DOCS / "data" / f"{card['date']}_{card['venue']}.json"
    doc = store.read_json(doc_path)
    res = {r["race_no"]: r for r in store.read_json(path, []) or []}
    names = {r["race_no"]: r.get("race_name") for r in card["races"]}
    if doc:
        for a in doc.get("races", []):
            n = a["race_no"]
            # the pre-race analysis stays as it was; only the result is attached
            if n in new and n in res and "result" not in a and a.get("race_name") == names.get(n):
                a["result"] = _result_summary(res[n])
        store.write_json(doc_path, doc)
    write_if_changed(store.DOCS / "data" / "schedule.json", race_schedule(card, finished))
    print(f"results: races {sorted(new)} on the page")
    update_paper()
    return True


def with_open(saved, day):
    """Latest odds per race, plus the first snapshot taken on race day itself
    (the day-before market is too thin to compare with)."""
    out = {}
    for k, v in saved["latest"].items():
        first = next((h["win"] for h in saved["history"].get(k, [])
                      if datetime.fromisoformat(h["t"]).astimezone(HKT).date().isoformat() == day), None)
        out[int(k)] = {**v, "open": first}
    return out


def live_odds(card, skip=()):
    """Read live odds on race day and the day before; never fails the run."""
    days_ahead = (date.fromisoformat(card["date"]) - today_hkt()).days
    if days_ahead not in (0, 1):
        return {}
    path = store.DATA / "odds" / f"{card['date']}_{card['venue']}.json"
    saved = store.read_json(path, {"latest": {}, "history": {}})
    try:
        from . import odds as odds_mod
        nos = [r["race_no"] for r in card["races"] if r["race_no"] not in skip]
        snap = odds_mod.fetch_odds(card["date"], card["venue"], nos)
    except Exception as e:  # browser missing, page changed, blocked...
        print(f"   odds unavailable: {e!r}")
        return with_open(saved, card["date"])
    fetched = snap.pop("_fetched_at")
    at = datetime.fromisoformat(fetched).astimezone(HKT)
    posts = {str(r["race_no"]): r.get("post_time") for r in card["races"]}
    for n, o in snap.items():
        k = str(n)
        pools = {p: o.pop(p) for p in ("qin", "qpl") if p in o}
        saved["latest"][k] = o
        hist = saved["history"].setdefault(k, [])
        if not hist or hist[-1]["win"] != {str(x): v for x, v in o["win"].items()}:
            hist.append({"t": fetched, "win": o["win"], "place": o["place"]})
        if pools:
            # all four pools at once, for the pool test registered in pools.py: first seen,
            # last before the post time, and the latest (= final odds after the off)
            shot = {"t": fetched, "updated": o.get("updated"), "win": o["win"], "place": o["place"], **pools}
            book = saved.setdefault("pools", {}).setdefault(k, {})
            book.setdefault("first", shot)
            book["last"] = shot
            if posts.get(k) and at.date().isoformat() == card["date"] and at.strftime("%H:%M") < posts[k]:
                book["pre"] = shot
    store.write_json(path, saved)
    print(f"   odds for {len(snap)} races")
    return with_open(saved, card["date"])


def run(use_agent=True):
    print("1) catching up on results")
    try:
        backfill(max_meetings=8)
        complete_recent_results()
    except RuntimeError as e:   # HKJC hiccup: today's odds and analysis still matter
        print(f"   catch-up skipped: {e}")
    reports.refresh_recent()

    attach_results()

    print("2) race card")
    try:
        card = scraper.fetch_racecard()
    except RuntimeError as e:   # carry on with the stored card so odds keep updating
        print(f"   race card unavailable: {e}")
        known = store.read_json(store.DOCS / "data" / "schedule.json", {}) or {}
        card = store.load_card(known.get("date"), known.get("venue"))
    if not card or not card["races"]:
        print("   no race card published")
        return
    # a race HKJC stopped serving mid-meeting keeps its last good version
    known = store.load_card(card["date"], card["venue"]) or {}
    got = {r["race_no"] for r in card["races"]}
    kept = [r for r in known.get("races", []) if r["race_no"] not in got]
    if kept:
        print(f"   keeping stored card for races {[r['race_no'] for r in kept]}")
        card["races"] = sorted(card["races"] + kept, key=lambda r: r["race_no"])
    validate_card(card)
    store.save_card(card)
    horses.remember_card_horses(card)
    mkey = f"{card['date']}_{card['venue']}"
    meeting_day = date.fromisoformat(card["date"])
    print(f"   {mkey}: {len(card['races'])} races")
    stored = store.read_json(store.result_path(card["date"], card["venue"]), []) or []
    write_if_changed(store.DOCS / "data" / "schedule.json",
                     race_schedule(card, {r["race_no"] for r in stored}))

    finished = set()
    if meeting_day <= today_hkt():
        print("3) live results for today's meeting")
        finished = live_results(card)
        print(f"   {len(finished)} races finished")
        write_if_changed(store.DOCS / "data" / "schedule.json", race_schedule(card, finished))

    # A race's analysis is frozen at its post time: odds read after the off (betting closed,
    # ~13 minutes before the result appears) mustn't rewrite the pick it is scored on.
    # Taken before the odds are read, so a pick made from odds read before the off still counts.
    off = started_races(card)

    print("3b) live odds")
    odds = live_odds(card, skip=finished)

    print("4) training model on history")
    races = store.load_all_results()
    cfg = store.read_json(store.DATA / "model_config.json", {})  # set by the daily self-check
    state, params, report = model.train(races, cfg)
    report["config"] = cfg or {"l2": 0.05, "stages": 1}
    print(f"   {report}")

    print("5) analysing card")
    results_today = {r["race_no"]: r for r in races
                     if r["date"] == card["date"] and r["venue"] == card["venue"]}
    out_path = store.DOCS / "data" / f"{mkey}.json"
    previous = store.read_json(out_path, {}) or {}
    prev_races = {r["race_no"]: r for r in previous.get("races", [])}
    out_races = []
    for race in card["races"]:
        res = results_today.get(race["race_no"])
        prev = prev_races.get(race["race_no"])
        if (res or race["race_no"] in off) and prev and prev.get("race_name") == race.get("race_name"):
            # race off or already run: keep the pre-race prediction frozen, attach the result
            a = prev
        else:
            a = analyze.analyse_race(state, params, race, odds.get(race["race_no"]))
            if not a:
                continue
            if use_agent and not res:
                a["ai"] = agent.analyse(a, mkey)
            elif prev:
                a["ai"] = prev.get("ai")
            if res or race["race_no"] in off:
                # first seen after the off (final odds) or the race (features include the
                # result itself): shown, but not a pre-race prediction, so not scored
                a["post_hoc"] = True
            # free rule-based preview; shown whenever there is no AI text
            a["summary"] = writer.summarise(a)
        if res:
            a["result"] = _result_summary(res)
        out_races.append(a)

    doc = {
        "meeting": {"date": card["date"], "venue": card["venue"],
                    "venue_zh": "沙田" if card["venue"] == "ST" else "跑馬地"},
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": report,
        "history_races": len(races),
        "races": out_races,
    }
    if meeting_day <= today_hkt():  # how many updates the meeting really got (self-check reads this)
        doc["runs"] = previous.get("runs", []) + [doc["updated_at"]]
    if write_if_changed(out_path, doc):
        store.write_json(store.DOCS / "data" / "latest.json", doc)
        print(f"   wrote {out_path}")
    else:
        print("   analysis unchanged")
    update_track_record()
    update_paper()


def update_paper():
    """模擬倉 (paper.py). Never stops the update: a fault here only leaves the page's copy old."""
    try:
        paper.update()
    except Exception as e:
        print(f"   paper account skipped: {e!r}")


def attach_results():
    """Attach results to earlier meetings whose last races finished after our last run."""
    today = today_hkt().isoformat()
    for p in sorted((store.DOCS / "data").glob("20*_*.json")):
        d = store.read_json(p)
        if d["meeting"]["date"] > today or all("result" in r for r in d["races"]):
            continue
        res = store.read_json(store.result_path(d["meeting"]["date"], d["meeting"]["venue"]), [])
        by_no = {r["race_no"]: r for r in res}
        changed = False
        for r in d["races"]:
            if "result" not in r and r["race_no"] in by_no:
                r["result"] = _result_summary(by_no[r["race_no"]])
                changed = True
        if changed:
            store.write_json(p, d)


# First meeting whose picks were frozen at each race's post time. Earlier meetings were
# re-analysed afterwards and are left out of the live record: 4 Nov ran no updates during the
# meeting (odds read 20 minutes to 4 hours before the off), 8 Nov used odds read up to 12
# minutes after it (so every first choice was the final favourite).
LIVE_FROM = "2026-11-11"


def _favourites(res_race):
    """Final favourite(s) of a race from its result: lowest final win odds among the starters."""
    priced = {x["no"]: x["win_odds"] for x in res_race.get("runners", [])
              if x.get("win_odds") and model.started(x)}
    low = min(priced.values(), default=None)
    return [n for n, o in priced.items() if o == low]


def _paired_z(diff):
    """z of the mean of per-race differences (0 when they never differ)."""
    x = np.array(diff, dtype=float)
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return round(float(x.mean() / (sd / np.sqrt(len(x)))), 2) if sd > 0 else 0.0


def update_track_record():
    """Live record of frozen pre-race top picks vs actual results, beside the final favourite
    of the same races. The backtest is scored on final odds, which include money placed after
    our picks are frozen, so the favourite in the same races is the fair comparison."""
    rows = []
    for p in sorted((store.DOCS / "data").glob("20*_*.json")):
        d = store.read_json(p)
        if d["meeting"]["date"] < LIVE_FROM:
            continue
        res = {r["race_no"]: r for r in
               store.read_json(store.result_path(d["meeting"]["date"], d["meeting"]["venue"]), []) or []}
        for r in d.get("races", []):
            if "result" not in r or not r["result"]["placings"] or r.get("post_hoc"):
                continue
            top = r["runners"][0]   # the published first choice
            model_top = min(r["runners"], key=lambda x: x["rank"])
            placings = r["result"]["placings"]
            won = top["no"] == placings[0]
            favs = _favourites(res.get(r["race_no"]) or {})
            rows.append({"date": d["meeting"]["date"], "race_no": r["race_no"],
                         "pick": top["no"], "pick_prob": top["win_prob"],
                         "winner": placings[0], "won": won,
                         "placed": top["no"] in placings,
                         "model_pick": model_top["no"],
                         "model_won": model_top["no"] == placings[0],
                         "return": r["result"]["win_odds"] if won else 0,
                         # joint favourites share the race: each counts 1 / their number
                         "favourite": favs,
                         "fav_won": round(sum(n == placings[0] for n in favs) / len(favs), 3) if favs else 0.0,
                         "fav_placed": round(sum(n in placings for n in favs) / len(favs), 3) if favs else 0.0})
    n = len(rows)
    summary = {
        "from": LIVE_FROM,
        "races": n,
        "wins": sum(r["won"] for r in rows),
        "places": sum(r["placed"] for r in rows),
        "model_wins": sum(r["model_won"] for r in rows),
        "fav_wins": round(sum(r["fav_won"] for r in rows), 1),
        "fav_places": round(sum(r["fav_placed"] for r in rows), 1),
        # first choice minus final favourite, race by race (+ = first choice did better)
        "z_win_vs_fav": _paired_z([r["won"] - r["fav_won"] for r in rows]) if n else None,
        "z_place_vs_fav": _paired_z([r["placed"] - r["fav_placed"] for r in rows]) if n else None,
        "flat_roi": round(sum(r["return"] or 0 for r in rows) / n - 1, 4) if n else None,
        "rows": rows[-200:],
    }
    store.write_json(store.DOCS / "data" / "track_record.json", summary)


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "run"
    if cmd == "run":
        run(use_agent="--no-agent" not in argv)
    elif cmd == "results":
        quick_results()
    elif cmd == "backfill":
        backfill(int(argv[2]) if len(argv) > 2 else 120)
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv)
