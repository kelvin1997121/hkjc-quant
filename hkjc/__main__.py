"""Pipeline entry point.

    python -m hkjc run            # one full update (what the scheduled workflow runs)
    python -m hkjc backfill 120   # download up to 120 past meetings
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timezone

from . import agent, analyze, model, scraper, store
from .backfill import backfill, today_hkt


def _result_summary(res_race):
    top = sorted([x for x in res_race["runners"] if x.get("place")], key=lambda x: x["place"])[:3]
    return {"placings": [x["no"] for x in top],
            "win_odds": top[0].get("win_odds") if top else None,
            "times": res_race.get("sectionals")}


def run(use_agent=True):
    print("1) catching up on results")
    backfill(max_meetings=8)

    attach_results()

    print("2) race card")
    card = scraper.fetch_racecard()
    if not card or not card["races"]:
        print("   no race card published")
        return
    store.save_card(card)
    mkey = f"{card['date']}_{card['venue']}"
    meeting_day = date.fromisoformat(card["date"])
    print(f"   {mkey}: {len(card['races'])} races")

    if meeting_day <= today_hkt():
        print("3) live results for today's meeting")
        live = scraper.fetch_meeting_results(meeting_day)
        if live:
            store.save_results(live)
        print(f"   {len(live)} races finished")

    print("4) training model on history")
    races = store.load_all_results()
    state, params, report = model.train(races)
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
        if res and race["race_no"] in prev_races:
            # race already run: keep the pre-race prediction frozen, attach the result
            a = prev_races[race["race_no"]]
        else:
            a = analyze.analyse_race(state, params, race)
            if not a:
                continue
            if use_agent and not res:
                a["ai"] = agent.analyse(a, mkey)
            elif race["race_no"] in prev_races:
                a["ai"] = prev_races[race["race_no"]].get("ai")
            if res:  # first seen after the race: features include the result itself
                a["post_hoc"] = True
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
    store.write_json(out_path, doc)
    store.write_json(store.DOCS / "data" / "latest.json", doc)
    update_track_record()
    print(f"   wrote {out_path}")


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
            top = r["runners"][0]
            won = top["no"] == r["result"]["placings"][0]
            rows.append({"date": d["meeting"]["date"], "race_no": r["race_no"],
                         "pick": top["no"], "pick_prob": top["win_prob"],
                         "winner": r["result"]["placings"][0], "won": won,
                         "placed": top["no"] in r["result"]["placings"],
                         "return": r["result"]["win_odds"] if won else 0})
    n = len(rows)
    summary = {
        "races": n,
        "wins": sum(r["won"] for r in rows),
        "places": sum(r["placed"] for r in rows),
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
