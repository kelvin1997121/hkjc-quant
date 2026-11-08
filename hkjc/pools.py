"""Pool test, registered 2026-11-08 before any of its data existed.

Question: do the 位置 (place), 連贏 (quinella) and 位置Q (quinella place) pools pay more
than the win pool says they should, often enough to make money at real dividends?

Rules, fixed now and not to be changed after seeing results:
  * Odds: the last snapshot taken before each race's post time ("pre" in
    data/odds/<meeting>.json, saved by __main__.live_odds). Races without one are skipped,
    as are races with fewer than 7 priced runners (place and 位置Q pay 3 places only then).
  * Chances from the win pool: p_i proportional to 1 / win odds; finishing orders by the
    Harville formula, P(a, b, c) = p_a * p_b / (1 - p_a) * p_c / (1 - p_a - p_b).
  * A bet is every place, quinella or quinella-place selection whose chance x shown odds
    is at least VALUE (1.20). $10 each, paid at the real dividends in data/results.
  * Each pool is judged on its own at LOOKS races: it passes only if the mean profit per
    race is above zero with z >= Z_PASS (races as the unit). Below that it has not shown
    anything; at the last look it is closed either way.
  * Shown alongside, not used to decide: the same bets below VALUE, which should do worse.

Harville is known to overrate favourites for 2nd and 3rd, so "value" it finds may be its
own error; that is why the verdict is the return at real dividends, not the model's opinion.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from . import store

REGISTERED = "2026-11-08"
VALUE = 1.20
LOOKS = (300, 600)
Z_PASS = 2.5
POOLS = {"pla": "位置", "qin": "連贏", "qpl": "位置Q"}


def chances(win: dict) -> dict:
    """win odds {no: odds} -> {"pla": {no: p}, "qin": {(a, b): p}, "qpl": {(a, b): p}}"""
    q = {int(n): 1 / o for n, o in win.items() if o}
    t = sum(q.values())
    p = {n: v / t for n, v in q.items()}
    pla, qin, qpl = {n: 0.0 for n in p}, {}, {}
    for a, b in itertools.permutations(p, 2):
        pab = p[a] * p[b] / (1 - p[a])
        key = tuple(sorted((a, b)))
        qin[key] = qin.get(key, 0.0) + pab
        for c in p:
            if c in (a, b):
                continue
            pabc = pab * p[c] / max(1 - p[a] - p[b], 1e-9)
            for n in (a, b, c):
                pla[n] += pabc
            for x, y in ((a, b), (a, c), (b, c)):
                k = tuple(sorted((x, y)))
                qpl[k] = qpl.get(k, 0.0) + pabc
    return {"pla": pla, "qin": qin, "qpl": qpl}


def _paid(dividends, pool, key):
    want = {str(key)} if isinstance(key, int) else {f"{key[0]},{key[1]}", f"{key[1]},{key[0]}"}
    return sum(amount for combo, amount in dividends.get(POOLS[pool], [])
               if str(combo).replace(" ", "") in want)


def race_bets(shot: dict, result: dict) -> dict:
    """Per pool: (staked, returned) for bets at or above VALUE, and for those below."""
    win = shot.get("win") or {}
    if len(win) < 7 or not result.get("dividends"):
        return {}
    ch = chances(win)
    shown = {"pla": {int(n): o for n, o in (shot.get("place") or {}).items()},
             "qin": {tuple(sorted(map(int, k.split("-")))): o for k, o in (shot.get("qin") or {}).items()},
             "qpl": {tuple(sorted(map(int, k.split("-")))): o for k, o in (shot.get("qpl") or {}).items()}}
    out = {}
    for pool, odds in shown.items():
        if not odds:
            continue
        above = below = (0, 0.0)
        for key, o in odds.items():
            p = ch[pool].get(key)
            if p is None:
                continue
            back = _paid(result["dividends"], pool, key)
            if p * o >= VALUE:
                above = (above[0] + 10, above[1] + back)
            else:
                below = (below[0] + 10, below[1] + back)
        out[pool] = {"above": above, "below": below}
    return out


def status() -> dict:
    """Races collected so far and, once a look is reached, each pool's verdict."""
    per = {pool: [] for pool in POOLS}
    below = {pool: [0, 0.0] for pool in POOLS}
    for path in sorted((store.DATA / "odds").glob("20*_*.json")):
        if path.stem[:10] < REGISTERED:
            continue
        saved = store.read_json(path, {}) or {}
        results = {r["race_no"]: r for r in store.read_json(store.RESULTS / f"{path.stem}.json", []) or []}
        for k, book in (saved.get("pools") or {}).items():
            shot, res = book.get("pre"), results.get(int(k))
            if not shot or not res:
                continue
            for pool, v in race_bets(shot, res).items():
                staked, back = v["above"]
                per[pool].append(back - staked)
                below[pool][0] += v["below"][0]
                below[pool][1] += v["below"][1]
    out = {"registered": REGISTERED, "races": max(len(v) for v in per.values()), "pools": {}}
    for pool, profits in per.items():
        n = len(profits)
        look = max([x for x in LOOKS if n >= x], default=None)
        row = {"name": POOLS[pool], "races": n, "next": next((x for x in LOOKS if x > n), None)}
        if look:
            x = np.array(profits[:look])
            z = float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if x.std() > 0 else 0.0
            row.update(look=look, profit_per_race=round(float(x.mean()), 2), z=round(z, 2),
                       verdict="過關" if z >= Z_PASS else ("未過關,完結" if look == LOOKS[-1] else "未過關"),
                       below_value_return=round(below[pool][1] / below[pool][0] - 1, 3) if below[pool][0] else None)
        out["pools"][pool] = row
    return out
