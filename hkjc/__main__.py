"""Pipeline entry point.

    python -m hkjc run            # one full update (what the scheduled workflow runs)
    python -m hkjc backfill 120   # download up to 120 past meetings
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone

from . import agent, analyze, model, scraper, store, writer
from .backfill import HKT, backfill, today_hkt


def _result_summary(res_race):
    top = sorted([x for x in res_race["runners"] if x.get("place")], key=lambda x: x["place"])[:3]
    return {"placings": [x["no"] for x in top],
            "win_odds": top[0].get("win_odds") if top else None,
            "times": res_race.get("sectionals")}


def complete_recent_results(days=4):
    """Re-fetch recent meetings whose stored results are missing races.

    Live runs save partial results during a meeting; if the last run of the
    day happened before the last race (GitHub cron is often late), the
    backfill would otherwise treat the date as done forever.
    """
    today = today_hkt()
    for p in sorted(store.CARDS.glob("20*_*.json")):
        card = store.read_json(p)
        d = date.fromisoformat(card["date"])
        if not (0 <= (today - d).days <= days):
            continue
        have = store.read_json(store.result_path(card["date"], card["venue"]), [])
        if len(have) < len(card["races"]):
            res = scraper.fetch_meeting_results(d)
            if len(res) > len(have):
                store.save_results(res)
                print(f"   completed {card['date']}: {len(have)} -> {len(res)} races")
        elif d < today and any(not r.get("has_sect") for r in have):
            # sectional times are sometimes published a while after the result
            for r in have:
                if not r.get("has_sect"):
                    scraper.attach_sectionals(r)
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


def race_schedule(card):
    """Post times, read by the cloud ticker to speed up during a meeting."""
    posts = sorted(r["post_time"] for r in card["races"] if r.get("post_time"))
    return {"date": card["date"], "venue": card["venue"],
            "first_post": posts[0] if posts else None, "last_post": posts[-1] if posts else None}


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
    for n, o in snap.items():
        saved["latest"][str(n)] = o
        hist = saved["history"].setdefault(str(n), [])
        if not hist or hist[-1]["win"] != {str(k): v for k, v in o["win"].items()}:
            hist.append({"t": fetched, "win": o["win"]})
    store.write_json(path, saved)
    print(f"   odds for {len(snap)} races")
    return with_open(saved, card["date"])


def run(use_agent=True):
    print("1) catching up on results")
    backfill(max_meetings=8)
    complete_recent_results()

    attach_results()

    print("2) race card")
    card = scraper.fetch_racecard()
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
    mkey = f"{card['date']}_{card['venue']}"
    meeting_day = date.fromisoformat(card["date"])
    print(f"   {mkey}: {len(card['races'])} races")
    write_if_changed(store.DOCS / "data" / "schedule.json", race_schedule(card))

    finished = set()
    if meeting_day <= today_hkt():
        print("3) live results for today's meeting")
        path = store.result_path(card["date"], card["venue"])
        stored = store.read_json(path, []) or []
        have = {r["race_no"]: r for r in stored if r.get("has_sect")}
        live = scraper.fetch_meeting_results(
            meeting_day, venue=card["venue"], race_nos=[r["race_no"] for r in card["races"]],
            have=have)
        if live:
            store.save_results(live)
        finished = {r["race_no"] for r in live}
        print(f"   {len(live)} races finished")

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
        if res and prev and prev.get("race_name") == race.get("race_name"):
            # race already run: keep the pre-race prediction frozen, attach the result
            a = prev
        else:
            a = analyze.analyse_race(state, params, race, odds.get(race["race_no"]))
            if not a:
                continue
            if use_agent and not res:
                a["ai"] = agent.analyse(a, mkey)
            elif prev:
                a["ai"] = prev.get("ai")
            if res:  # first seen after the race: features include the result itself
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


def update_track_record():
    """Live record of frozen pre-race top picks vs actual results."""
    rows = []
    for p in sorted((store.DOCS / "data").glob("20*_*.json")):
        d = store.read_json(p)
        for r in d.get("races", []):
            if "result" not in r or not r["result"]["placings"] or r.get("post_hoc"):
                continue
            top = r["runners"][0]   # the published first choice
            model_top = min(r["runners"], key=lambda x: x["rank"])
            won = top["no"] == r["result"]["placings"][0]
            rows.append({"date": d["meeting"]["date"], "race_no": r["race_no"],
                         "pick": top["no"], "pick_prob": top["win_prob"],
                         "winner": r["result"]["placings"][0], "won": won,
                         "placed": top["no"] in r["result"]["placings"],
                         "model_pick": model_top["no"],
                         "model_won": model_top["no"] == r["result"]["placings"][0],
                         "return": r["result"]["win_odds"] if won else 0})
    n = len(rows)
    summary = {
        "races": n,
        "wins": sum(r["won"] for r in rows),
        "places": sum(r["placed"] for r in rows),
        "model_wins": sum(r["model_won"] for r in rows),
        "flat_roi": round(sum(r["return"] or 0 for r in rows) / n - 1, 4) if n else None,
        "rows": rows[-200:],
    }
    store.write_json(store.DOCS / "data" / "track_record.json", summary)


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "run"
    if cmd == "run":
        run(use_agent="--no-agent" not in argv)
    elif cmd == "backfill":
        backfill(int(argv[2]) if len(argv) > 2 else 120)
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv)
