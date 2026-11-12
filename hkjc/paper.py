"""模擬倉 (paper betting account), registered 2026-11-10 before any of its races were run.

Rules, fixed now and not to be changed after seeing results:
  * Starts with the 2026-11-11 meeting. Every book starts with HK$1,000 and bets $10 a time,
    flat. A book with less than $10 left places no more bets.
  * Bets come from each race's analysis as frozen at its post time (docs/data/<meeting>.json).
    Races analysed only after the off (post_hoc) get no bets. The value strategy also uses the
    last odds snapshot before the post time ("pre" in data/odds/<meeting>.json). Bets are paid
    at HKJC's real dividends (data/results, per $10). A bet on a withdrawn horse is refunded.
    Made exact on 2026-11-12, before any such case had come up in the account: a horse that
    is in the result with a withdrawn code (WV, WX...) counts as withdrawn even if the race
    card never marked it, and a bet in a pool HKJC did not pay out is refunded as well.
  * Four strategies, each with its own book:
      fav_place   首選位置       $10 place on the first choice
      top2_qin    首選+次選連贏   $10 quinella on the first two choices
      shot_place  冷門馬位置     $10 place on each 冷門馬 (win odds 10 or more, model top 5)
      value       彩池值博       chances from the win pool (Harville, as in pools.py); of the
                                place, quinella and quinella-place selections whose chance x
                                shown odds is at least 1.20, the two with the highest chance
                                (not the highest chance x odds: those are mostly 300-1 pairs
                                where Harville is weakest, and too rare to judge in 300 races);
                                $10 each
  * The main account (模擬倉) has its own HK$1,000 and places the bets of the strategy it
    follows. It starts on top2_qin, the best of the three backtested strategies at real
    dividends on unseen races (about -6%; value has no backtest, as pool odds were not kept).
  * Learning: after every 300 settled races (300, 600, ...), each other strategy is compared
    with the one followed over those last 300 races, race by race (difference in flat-stake
    profit, $10 a bet, whatever the books' balances). The main account switches to the best
    of them only if that difference is above zero with z >= 2.5. Nothing else changes it.

Every strategy here lost money at real dividends in the backtests, so the expected result is a
slow loss of about the betting take; the account is a live test, not a way to make money.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np

from . import pools, store
from .backfill import HKT
from .model import started

START = "2026-11-11"
BANKROLL = 1000.0
STAKE = 10.0
LOOK_EVERY = 300
Z_PASS = 2.5
VALUE = 1.20
VALUE_MAX = 2
FIRST = "top2_qin"

STRATEGIES = {
    "fav_place": {"name": "首選位置", "rule": "每場 $10 位置買排名第一嘅馬", "backtest": "首選 位置"},
    "top2_qin": {"name": "首選+次選連贏", "rule": "每場 $10 連贏買排名頭兩匹", "backtest": "首選+次選 連贏"},
    "shot_place": {"name": "冷門馬位置", "rule": "每匹冷門馬(獨贏 10 倍以上而模型排頭五)$10 位置", "backtest": "冷門馬 位置"},
    "value": {"name": "彩池值博", "rule": "用獨贏賠率估位置、連贏、位置Q 機會;機會 × 賠率 ≥ 1.2 嘅注項入面,揀最大機會嘅兩注,每注 $10", "backtest": None},
}
POOL_ZH = {"pla": "位置", "qin": "連贏", "qpl": "位置Q"}


def _key(pool, sel):
    return sel if pool == "pla" else tuple(sorted(sel))


def _sel_text(pool, sel):
    return str(sel) if pool == "pla" else f"{sel[0]}-{sel[1]}"


def _shown(shot):
    """Odds shown in a snapshot, keyed like pools.chances: place by horse, pairs as (a, b)."""
    pair = lambda d: {tuple(sorted(map(int, k.split("-")))): o for k, o in (d or {}).items()}
    return {"pla": {int(n): o for n, o in (shot.get("place") or {}).items()},
            "qin": pair(shot.get("qin")), "qpl": pair(shot.get("qpl"))}


def pick_bets(race, shot):
    """Each strategy's selections for one analysed race: {strategy: [(pool, sel, odds, chance)]}.

    `race` is the analysis (runners with final_rank, longshot, win/place odds); `shot` the
    odds snapshot for the value strategy (or None)."""
    runners = sorted(race.get("runners") or [], key=lambda x: x.get("final_rank") or x.get("rank") or 99)
    out = {k: [] for k in STRATEGIES}
    if not runners:
        return out
    ch = pools.chances(shot["win"]) if shot and len(shot.get("win") or {}) >= 2 else None
    shown = _shown(shot) if shot else {"pla": {}, "qin": {}, "qpl": {}}
    p = lambda pool, sel: ch[pool].get(_key(pool, sel)) if ch else None
    o = lambda pool, sel: shown[pool].get(_key(pool, sel))

    a = runners[0]
    out["fav_place"].append(("pla", a["no"], o("pla", a["no"]) or a.get("place_odds"), p("pla", a["no"])))
    if len(runners) > 1:
        b = runners[1]
        sel = (a["no"], b["no"])
        out["top2_qin"].append(("qin", sel, o("qin", sel), p("qin", sel)))
    for x in runners:
        if x.get("longshot"):
            out["shot_place"].append(("pla", x["no"], o("pla", x["no"]) or x.get("place_odds"), p("pla", x["no"])))
    if ch and len(shot["win"]) >= 7:
        cands = []
        for pool in ("pla", "qin", "qpl"):
            for key, odds in shown[pool].items():
                c = ch[pool].get(key)
                if c is not None and odds and c * odds >= VALUE:
                    cands.append((c, pool, key, odds, c))
        cands.sort(key=lambda t: -t[0])   # most likely first
        out["value"] = [(pool, key, odds, c) for _, pool, key, odds, c in cands[:VALUE_MAX]]
    return out


def settle(pool, sel, result, withdrawn):
    """Money back for a $10 bet: the dividend; the stake if a horse was withdrawn or the pool
    was not paid out; else 0."""
    horses = {sel} if pool == "pla" else set(sel)
    if horses & withdrawn or not pools.paid_out(result.get("dividends") or {}, pool):
        return STAKE, True
    return float(pools._paid(result.get("dividends") or {}, pool, _key(pool, sel))), False


def _withdrawn(result, card_race):
    ran = {x["no"] for x in result.get("runners", []) if x.get("no") is not None and x.get("place")}
    listed = {x["no"] for x in result.get("runners", []) if x.get("no") is not None}
    scratched = {x["no"] for x in (card_race or {}).get("runners", []) if x.get("scratched")}
    entered = {x["no"] for x in (card_race or {}).get("runners", [])} or listed
    # in the result but never started: withdrawn at the start, after the last race card we saw
    out = {x["no"] for x in result.get("runners", []) if x.get("no") is not None and not started(x)}
    # on the card but not in the result at all, or marked scratched and never placed
    return {n for n in entered if n not in listed} | (scratched - ran) | out


def _started(day, post, now):
    if day < now.date().isoformat():
        return True
    if day > now.date().isoformat() or not post:
        return False
    return post <= now.strftime("%H:%M")


class Book:
    def __init__(self):
        self.balance, self.staked, self.returned = BANKROLL, 0.0, 0.0
        self.settled_staked = 0.0   # stakes of settled bets only: profit and return use these
        self.bets = self.hits = self.pending = 0

    def place(self):
        if self.balance < STAKE:
            return False
        self.balance -= STAKE
        self.staked += STAKE
        self.bets += 1
        return True

    def pay(self, back, refund):
        self.balance += back
        self.returned += back
        self.settled_staked += STAKE
        if back > 0 and not refund:
            self.hits += 1

    def row(self):
        net = self.returned - self.settled_staked
        return {"balance": round(self.balance, 1), "staked": round(self.staked, 1),
                "settled_staked": round(self.settled_staked, 1),
                "returned": round(self.returned, 1), "net": round(net, 1), "bets": self.bets,
                "hits": self.hits, "pending": self.pending,
                "roi": round(net / self.settled_staked, 4) if self.settled_staked else None}


def _bet_row(pool, sel, odds, chance, names):
    nm = names.get(sel) if pool == "pla" else " / ".join(names.get(n, "") for n in sel)
    return {"pool": POOL_ZH[pool], "sel": _sel_text(pool, sel), "names": nm,
            "odds": odds, "chance": round(chance, 4) if chance is not None else None,
            "value": round(chance * odds, 2) if chance is not None and odds else None,
            "stake": STAKE, "back": None, "refund": False, "placed": True}


def _z(diff):
    x = np.array(diff, dtype=float)
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return float(x.mean() / (sd / math.sqrt(len(x)))) if sd > 0 else 0.0


def build(now=None):
    """Replay every race since START in order and return the whole account (paper.json)."""
    now = now or datetime.now(HKT)
    books = {k: Book() for k in STRATEGIES}
    main, following = Book(), FIRST
    flat = {k: [] for k in STRATEGIES}   # flat-stake profit per settled race, for the looks
    switches, looks, meetings = [], [], []
    settled = 0

    for path in sorted((store.DOCS / "data").glob("20*_*.json")):
        d = store.read_json(path, {}) or {}
        m = d.get("meeting") or {}
        if not m or m.get("date", "") < START:
            continue
        day, venue = m["date"], m["venue"]
        results = {r["race_no"]: r for r in store.read_json(store.result_path(day, venue), []) or []}
        odds = store.read_json(store.DATA / "odds" / f"{day}_{venue}.json", {}) or {}
        card = {r["race_no"]: r for r in (store.load_card(day, venue) or {}).get("races", [])}
        rows = []
        for race in sorted(d.get("races", []), key=lambda r: r["race_no"]):
            n = race["race_no"]
            res = results.get(n)
            off = res is not None or _started(day, race.get("post_time"), now)
            row = {"race_no": n, "post_time": race.get("post_time"), "race_name": race.get("race_name"),
                   "following": following, "main": [], "books": {}}
            if race.get("post_hoc"):
                row["state"] = "post_hoc"
                rows.append(row)
                continue
            book = (odds.get("pools") or {}).get(str(n)) or {}
            # after the off only the last pre-race snapshot counts; before it, the latest odds
            shot = book.get("pre") if off else (book.get("pre") or book.get("last"))
            picks = pick_bets(race, shot)
            names = {x["no"]: x.get("horse", "") for x in race.get("runners", [])}
            row["state"] = "settled" if res and res.get("dividends") else ("placed" if off else "planned")
            if res and not res.get("dividends"):
                row["state"] = "void"   # no dividends (abandoned or not parsed): nothing is bet
            withdrawn = _withdrawn(res, card.get(n)) if row["state"] == "settled" else set()

            def run(bk, sels):
                rows_out = []
                for pool, sel, o, c in sels:
                    b = _bet_row(pool, sel, o, c, names)
                    if row["state"] == "void" or not bk.place():
                        b["placed"] = False
                        rows_out.append(b)
                        continue
                    if row["state"] == "settled":
                        back, refund = settle(pool, sel, res, withdrawn)
                        bk.pay(back, refund)
                        b["back"], b["refund"] = back, refund
                    else:
                        bk.pending += 1
                    rows_out.append(b)
                return rows_out

            for k in STRATEGIES:
                row["books"][k] = run(books[k], picks[k])
            row["main"] = run(main, picks[following])

            if row["state"] == "settled":
                settled += 1
                for k in STRATEGIES:
                    profit = 0.0
                    for pool, sel, o, c in picks[k]:
                        back, _ = settle(pool, sel, res, withdrawn)
                        profit += back - STAKE
                    flat[k].append(profit)
                top = sorted((x for x in res["runners"] if x.get("place")), key=lambda x: x["place"])[:3]
                row["result"] = [x["no"] for x in top]
                if settled % LOOK_EVERY == 0:
                    look = {"at": settled, "following": following, "compare": []}
                    base = flat[following][-LOOK_EVERY:]
                    best = None
                    for k in STRATEGIES:
                        if k == following:
                            continue
                        diff = [a - b for a, b in zip(flat[k][-LOOK_EVERY:], base)]
                        z, mean = _z(diff), float(np.mean(diff))
                        look["compare"].append({"strategy": k, "diff_per_race": round(mean, 2), "z": round(z, 2)})
                        if mean > 0 and z >= Z_PASS and (best is None or z > best[1]):
                            best = (k, z)
                    look["switched_to"] = best[0] if best else None
                    looks.append(look)
                    if best:
                        switches.append({"at": settled, "from": following, "to": best[0], "z": round(best[1], 2),
                                         "date": day, "race_no": n})
                        following = best[0]
            rows.append(row)
        meetings.append({"date": day, "venue": venue, "venue_zh": m.get("venue_zh"), "races": rows})

    for mt in meetings:
        mt["review"] = review(mt)
    out = {
        "start": START, "bankroll": BANKROLL, "stake": STAKE, "look_every": LOOK_EVERY, "z_pass": Z_PASS,
        "races": settled, "next_look": (settled // LOOK_EVERY + 1) * LOOK_EVERY,
        "main": {"following": following, "following_name": STRATEGIES[following]["name"],
                 **main.row(), "switches": switches, "looks": looks},
        "books": [{"key": k, **{f: v for f, v in s.items() if f != "backtest"}, **books[k].row(),
                   "backtest_roi": _backtest(s["backtest"])} for k, s in STRATEGIES.items()],
        "meetings": meetings[::-1][:8],
    }
    return out


def _backtest(name):
    if not name:
        return None
    sc = store.read_json(store.DOCS / "data" / "selfcheck.json", {}) or {}
    for x in (sc.get("bet_returns") or {}).get("unseen", []):
        if x.get("name") == name:
            return x.get("roi")
    return None


def _money(v):
    return f"{'+' if v > 0 else '-' if v < 0 else ''}${abs(v):,.0f}"


def review(mt):
    """Plain-language notes for one meeting: what the account did and how each strategy fared."""
    races = mt["races"]
    lines = []
    main_bets = [b for r in races for b in r["main"] if b["placed"]]
    done = [b for b in main_bets if b["back"] is not None]
    if not main_bets:
        return ["今日冇落注。"]
    fol = {r["following"] for r in races if r["main"]}
    names = "、".join(STRATEGIES[k]["name"] for k in fol)
    planned = sum(1 for r in races if r["state"] == "planned")
    lines.append(f"模擬倉今日跟「{names}」,共 {len(main_bets)} 注,每注 ${STAKE:.0f}" +
                 (f";當中 {planned} 場未開跑,注項會跟賠率變,開跑一刻固定。" if planned else "。"))
    if done:
        back = sum(b["back"] for b in done)
        hits = sum(1 for b in done if b["back"] > 0 and not b["refund"])
        lines.append(f"已結算 {len(done)} 注:中 {hits} 注,收回 ${back:,.0f},淨 {_money(back - STAKE * len(done))}。")
        per = []
        for k, s in STRATEGIES.items():
            bs = [b for r in races for b in r["books"].get(k, []) if b["placed"] and b["back"] is not None]
            if bs:
                net = sum(b["back"] for b in bs) - STAKE * len(bs)
                per.append(f"{s['name']} {len(bs)} 注 {_money(net)}")
        if per:
            lines.append("各策略今日(已結算):" + ";".join(per) + "。")
        best = max(done, key=lambda b: b["back"] - STAKE)
        if best["back"] > STAKE:
            lines.append(f"最好一注:{best['pool']} {best['sel']}({best['names']}),派 ${best['back']:,.1f}。")
    lines.append("一日嘅賺蝕主要係運氣:一個賽馬日 8 至 11 場,中唔中一注連贏已經可以左右結果。"
                 f"策略要累積夠 {LOOK_EVERY} 場先比較,唔會因為一兩日輸贏就改。")
    return lines


def update(now=None):
    """Rebuild docs/data/paper.json; rewritten only when something besides the time changed
    (otherwise every update would make a commit)."""
    doc = build(now)
    path = store.DOCS / "data" / "paper.json"
    old = store.read_json(path) or {}
    old.pop("updated_at", None)
    if old == json.loads(json.dumps(doc)):
        return False
    doc["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    store.write_json(path, doc)
    print(f"   paper account: {doc['races']} races settled, main balance ${doc['main']['balance']}")
    return True
