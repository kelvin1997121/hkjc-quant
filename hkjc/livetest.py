"""New-information test, registered 2026-11-12 before any of its races were run.

Question: three things the system now saves before every race have never been tested. Does
any of them say more about the winner than the win odds at our freeze?

Rules, fixed now and not to be changed after seeing results:
  * Races: every race from FROM (2026-11-14) with a result, exactly one winner, a "pre" odds
    snapshot (the last one read before the post time, data/odds) with at least 7 priced
    runners. Runners: those priced in "pre". Races of 4, 8 and 11 Nov are not used: the page's
    self-check has already shown late-money counts for them.
  * Base: P(i) proportional to exp(a * log q_i), q = 1 / "pre" win odds, normalised.
    Each candidate adds b * f_i, alone.
  * Candidates:
      steam       log(q_pre / q_open): the change in market chance from the first snapshot read
                  on race day (Hong Kong time, about 08:00) to "pre", both normalised over the
                  runners. 0 for a horse not priced in both; races without a race-day first
                  snapshot are left out of this candidate.
      fast_work   number of 快操 entries dated in the 14 days before race day, from
                  data/trackwork (saved before the race).
      vet_recent  1 if the horse has a vet record dated in the 90 days before race day, else 0.
    The two trackwork candidates use only races whose trackwork file has the race, with the vet
    page read correctly (vet_ok) and every runner in it.
  * Judged per candidate at LOOKS (300, 600) races, cross-validated in 5 date-ordered blocks:
    each block's races are predicted by fits on the other four. Gain per race = base log loss
    minus candidate log loss. Pass: mean gain > 0 with z >= Z_PASS, and the same run with the
    candidate shuffled among the runners of each race gives z < PLACEBO_MAX. Below that it has
    shown nothing; at the last look it is closed either way.
  * A pass is reported for the user to decide on (a model factor or a page mark). It never
    changes the ranking by itself.

Why these three: the review of 12 Nov found that the system's ranking is the market at the
freeze, so only information the market may not fully price can help. Late money is the one
the backtest cannot see (past results keep final odds only); trackwork and vet records are
kept by HKJC for active horses only, so they can only be tested fairly on races saved from now.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import numpy as np

from . import store
from .backfill import HKT
from .model import started
from .oldbank import fit, logloss

REGISTERED = "2026-11-12"
FROM = "2026-11-14"
LOOKS = (300, 600)
BLOCKS, Z_PASS, PLACEBO_MAX = 5, 2.5, 1.0
MIN_PRICED = 7
WORK_DAYS, VET_DAYS = 14, 90
CANDIDATES = ["steam", "fast_work", "vet_recent"]
NAMES_ZH = {"steam": "落飛(朝早至開跑前賠率變化)", "fast_work": "兩星期內快操次數",
            "vet_recent": "三個月內有獸醫紀錄"}
SEED = 20261112


def _probs(win: dict, nos) -> np.ndarray | None:
    inv = np.array([1 / win[n] if win.get(n) else 0.0 for n in nos])
    return inv / inv.sum() if inv.sum() > 0 else None


def _open_win(history: list, day: str) -> dict:
    """Win odds of the first snapshot taken on race day itself (Hong Kong time)."""
    for h in history:
        if datetime.fromisoformat(h["t"]).astimezone(HKT).date().isoformat() == day:
            return {int(k): v for k, v in h["win"].items()}
    return {}


def _work(horse: dict, day: str) -> tuple[float, float]:
    on = date.fromisoformat(day)
    since_w = (on - timedelta(days=WORK_DAYS)).isoformat()
    since_v = (on - timedelta(days=VET_DAYS)).isoformat()
    fast = sum(1 for d, _ in (horse.get("work") or {}).get("快操", []) if since_w <= d < day)
    vet = any(since_v <= row[0] < day for row in horse.get("vet") or [] if row)
    return float(fast), float(vet)


def build() -> list[dict]:
    """One sample per usable race: {"key", "y", "logq", "f": {candidate: array or None}}."""
    out = []
    for path in sorted((store.DATA / "odds").glob("20*_*.json")):
        day, venue = path.stem.split("_")
        if day < FROM:
            continue
        saved = store.read_json(path, {}) or {}
        results = {r["race_no"]: r for r in store.read_json(store.result_path(day, venue), []) or []}
        work = (store.read_json(store.DATA / "trackwork" / path.name, {}) or {}).get("races", {})
        for k, book in sorted((saved.get("pools") or {}).items(), key=lambda kv: int(kv[0])):
            res, pre = results.get(int(k)), (book.get("pre") or {}).get("win") or {}
            pre = {int(n): o for n, o in pre.items() if o}
            if not res or len(pre) < MIN_PRICED:
                continue
            runners = [x for x in res["runners"] if started(x) and x.get("no") in pre]
            winners = [i for i, x in enumerate(runners) if x.get("place") == 1]
            if len(winners) != 1:
                continue
            nos = [x["no"] for x in runners]
            q = _probs(pre, nos)
            q_open = _probs(_open_win(saved.get("history", {}).get(k, []), day), nos)
            steam = np.array([math.log(a / b) if b > 0 and a > 0 else 0.0 for a, b in zip(q, q_open)]) \
                if q_open is not None else None
            f = {"steam": steam, "fast_work": None, "vet_recent": None}
            race_work = work.get(k) or {}
            horses = race_work.get("horses") or {}
            if race_work.get("vet_ok") and all(x.get("horse_code") in horses for x in runners):
                fw = np.array([_work(horses[x["horse_code"]], day) for x in runners])
                f["fast_work"], f["vet_recent"] = fw[:, 0], fw[:, 1]
            out.append({"key": (day, venue, int(k)), "y": winners[0], "logq": np.log(q), "f": f})
    return out


def judge(samples: list[dict], cand: str, placebo=False) -> dict:
    """Date-ordered block cross-validation of base vs base + candidate."""
    rng = np.random.default_rng(SEED)
    use = [s for s in samples if s["f"][cand] is not None]
    y = [s["y"] for s in use]
    f = [rng.permutation(s["f"][cand]) if placebo else s["f"][cand] for s in use]
    base = [s["logq"][:, None] for s in use]
    full = [np.column_stack([s["logq"], v]) for s, v in zip(use, f)]
    edges = np.linspace(0, len(use), BLOCKS + 1).astype(int)
    gains = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        train = [i for i in range(len(use)) if not lo <= i < hi]
        wb = fit([base[i] for i in train], [y[i] for i in train])
        wc = fit([full[i] for i in train], [y[i] for i in train])
        gains += [logloss(base[i], y[i], wb) - logloss(full[i], y[i], wc) for i in range(lo, hi)]
    g = np.array(gains)
    z = float(g.mean() / (g.std(ddof=1) / math.sqrt(len(g)))) if len(g) > 1 and g.std() > 0 else 0.0
    return {"gain": round(float(g.mean()), 5), "z": round(z, 2)}


def status() -> dict:
    """Races collected per candidate and, once a look is reached, its verdict."""
    samples = build()
    out = {"registered": REGISTERED, "from": FROM, "candidates": {}}
    for c in CANDIDATES:
        n = sum(1 for s in samples if s["f"][c] is not None)
        look = max([x for x in LOOKS if n >= x], default=None)
        row = {"name": NAMES_ZH[c], "races": n, "next": next((x for x in LOOKS if x > n), None)}
        if look:
            first = [s for s in samples if s["f"][c] is not None][:look]
            res = judge(first, c)
            res["placebo_z"] = judge(first, c, placebo=True)["z"]
            passed = res["gain"] > 0 and res["z"] >= Z_PASS and res["placebo_z"] < PLACEBO_MAX
            row.update(look=look, **res, passed=passed,
                       verdict="過關" if passed else ("未過關,完結" if look == LOOKS[-1] else "未過關"))
        out["candidates"][c] = row
    return out
