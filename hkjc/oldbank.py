"""Old-season test, registered 2026-11-11, before any of its data was downloaded.

    HKJC_DATA_DIR=<bank> python -m hkjc.oldbank download 2019-09-01 2020-08-31
    HKJC_DATA_DIR=<bank> python -m hkjc.oldbank test

Why: beside the market odds the model adds nothing (blend weight 0), and the 323 locked races
are used up. After the 11 Nov 2026 meeting the user asked whether the system misses horses whose
recent runs hid their ability: a strong race (its other runners placed again later, or it was run
fast), or trouble in running (three wide, a bad draw, blocked in the straight). These candidates,
plus a retest of jockey_up, are tested ONCE on three seasons the system has never seen. Nothing
below may change after the data has been looked at, and every result is reported, pass or fail.

Data
  Meetings 2019-09-01 .. 2022-08-31 from racing.hkjc.com: results (places, draws, final win
  odds, running positions, sectionals) and running comments (走勢評述). Stewards' incident
  reports are not served for these dates (only from about 2024), so the text factors read the
  running comments only. The bank lives on the `oldbank` branch and is read through
  HKJC_DATA_DIR; it is never put in data/ on main, so live training never sees it.
  The test runs only when all three seasons are downloaded and at least 90% of test races have
  comments; otherwise it stops before looking at any result.

Timeline
  Horse histories, standard times, draw statistics and jockey rates are built from 2019-09-01.
  Races before 2020-03-01 only build that state. Test races: 2020-03-01 .. 2022-08-31, every
  race whose starters all have final win odds and that has exactly one winner.

Market
  Final win odds on the results page; chances q = (1/odds) normalised (model.market_probs).

Test, each candidate f on its own
  base: P(i) ∝ exp(a·log q_i)              cand: P(i) ∝ exp(a·log q_i + b·f_i)
  Test races in date order. The first 300 only train. The rest is cut into 4 date-ordered folds,
  each predicted by base and cand fitted (maximum likelihood) on all test races before it.
  Gain per race = log loss(base) - log loss(cand).
  Pass: mean gain > 0 with z = mean / (sd / sqrt(n)) >= 2.5, and a placebo run (f shuffled among
  the runners of each race, in fitting and in predicting) gives z < 1.0.
  Seven candidates at z >= 2.5: about a 4% chance that any of them passes by luck alone.

  A candidate that passes joins the model's factors and the model's blend with the market is
  refitted; the published ranking changes only if that blend then gives the model weight above 0.
  A candidate that fails is not used.

Amended 2026-11-11, still before any bank data was read: at the user's request 強組 follows
their definition. A strong race need not be a class drop: its field ran a fast time, and its
beaten horses went on to place at later meetings. Looked at over the horse's last 3 runs, any
of which may count. (First version: all other runners' later starts, mean of the last 2 runs.)

Candidates (only information from before the race)
  field_after   強組, the beaten horses: for each of the horse's last 3 runs, that race's beaten
                horses (not its winner, not this horse) whose next start came before today; the
                share of them that finished in the first 3 in that next start, shrunk towards
                0.25 with the weight of 5 horses, minus 0.25. Best of those runs; 0 if none.
  race_time     強組, the time: for each of the horse's last 3 runs, the winner's speed figure
                (model.State.speed_fig, as known before that race) minus the mean winner figure
                of up to the previous 200 races of the same class (needs 20). Best; 0 if none.
  strong_race   強組, both together: 1 if in any of the last 3 runs the horse finished outside
                the first 3 in a race whose race_time figure is >= +3 (0.3 s per km faster than
                its class) and in which at least 40% of the beaten horses (at least 3 of them
                with a next start before today) placed in their next start; else 0.
  forgiven_form 原諒有藉口嘅輸: the model's form (finish score over the last 5 runs, weights
                0.6 per run back) recomputed without the runs that finished outside the first 3
                and whose comment shows trouble (TROUBLE), minus the normal form. 0 if no run or
                every run is left out.
  last_draw_bad 上仗檔位不利: minus the model's draw statistic for the last run (log of that
                draw's top-3 rate against all draws on that course and distance, as known
                before that race). Positive = the last draw was a poor one. 0 if no last run.
  blocked_late  直路受阻: 1 if one sentence of the last run's comment has a BLOCKED word and a
                LATE word, else 0.
  jockey_up     轉騎師 (retest): log win rate of today's jockey minus that of the last run's
                jockey, when they differ (model.RateBook as in hkjc/model.py); else 0.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np

from . import backfill, model, reports, scraper, store

REGISTERED = "2026-11-11"
BANK_FROM, TEST_FROM, BANK_TO = "2019-09-01", "2020-03-01", "2022-08-31"
SEASONS = (("2019-09-01", "2020-08-31"), ("2020-09-01", "2021-08-31"), ("2021-09-01", "2022-08-31"))
FIRST_TRAIN, FOLDS, Z_PASS, PLACEBO_MAX = 300, 4, 2.5, 1.0
MIN_COMMENTS = 0.90
LAST_N, SHRINK_N, TOP3_BASE = 3, 5, 0.25
FAST, PLACED_SHARE, MIN_BEATEN = 3.0, 0.40, 3
CLASS_RACES, CLASS_MIN = 200, 20
FORM_RUNS, FORM_DECAY = 5, 0.6

BLOCKED = ("未能望空", "難以望空", "無法望空", "不能望空", "沒有空位", "無位可走", "沒有走位", "被困", "受阻")
CHECKED = ("收慢", "勒避", "受擠迫", "被夾")
SLOW_START = ("出閘緩慢", "起步緩慢", "出閘笨拙")
HEALTH = ("流鼻血", "心律不正常", "跛行", "不良於行")
TROUBLE = tuple(reports.WIDE_WORDS) + BLOCKED + CHECKED + SLOW_START + HEALTH
LATE = ("直路", "最後", "末段", "一百", "二百", "三百", "四百", "終點")
CANDIDATES = ["field_after", "race_time", "strong_race", "forgiven_form", "last_draw_bad",
              "blocked_late", "jockey_up"]
NAMES_ZH = {"field_after": "強組(輸嘅馬再出入頭三)", "race_time": "強組(嗰場跑得快)",
            "strong_race": "強組(跑得快兼輸嘅馬再出入頭三)",
            "forgiven_form": "原諒有藉口嘅輸", "last_draw_bad": "上仗檔位不利",
            "blocked_late": "上仗直路受阻", "jockey_up": "轉騎師(重測)"}


def _guard():
    if store.DATA.resolve() == (store.ROOT / "data").resolve():
        raise SystemExit("set HKJC_DATA_DIR to the old-season bank; it must never be the live data/")


# ------------------------------------------------------------------------- download

def download(start: date, end: date, log=print) -> None:
    """Results (with sectionals) for the range, then running comments for those meetings."""
    _guard()
    backfill.backfill_range(start, end, log)
    files = sorted(p for p in store.RESULTS.glob("20*_*.json")
                   if "NONE" not in p.name and start.isoformat() <= p.stem[:10] <= end.isoformat())
    with ThreadPoolExecutor(max_workers=scraper.CONCURRENCY) as pool:
        for p in files:
            if backfill._out_of_time():
                break
            races = store.read_json(p, []) or []
            todo = [r for r in races if "has_comment" not in r]
            if not todo:
                continue
            try:
                list(pool.map(reports.attach_comments, todo))
            except RuntimeError as e:
                log(f"  ! {p.stem}: {e}")
                continue
            store.write_json(p, races)
            log(f"  comments {p.stem}: {sum(r.get('has_comment', False) for r in races)}/{len(races)}")


# ------------------------------------------------------------------------- factors

def trouble(text: str) -> bool:
    return any(w in text for w in TROUBLE)


def blocked_late(text: str) -> bool:
    return any(any(b in s for b in BLOCKED) and any(w in s for w in LATE) for s in text.split("。"))


def _form(runs):
    w = np.array([FORM_DECAY ** i for i in range(len(runs))][::-1])
    return float(np.dot(w, [x["fscore"] for x in runs]) / w.sum())


def beaten_next(run, code, today, hist, race_runs):
    """That race's beaten horses (not its winner, not `code`) whose next start came before
    today: how many, and how many of them finished in the first 3 in that next start."""
    n = top3 = 0
    for c, place in race_runs[run["key"]]:
        if c == code or place == 1:
            continue
        nxt = next((h for h in hist.get(c, []) if h["date"] > run["date"]), None)
        if nxt and nxt["date"] < today:
            n += 1
            top3 += bool(nxt["place"] and nxt["place"] <= 3)
    return n, top3


def features(x, today, hist, race_runs, race_fig, st) -> list[float]:
    runs = hist.get(x["horse_code"], [])
    out = {}
    after, times, strong = [], [], 0.0
    for run in runs[-LAST_N:]:
        n, top3 = beaten_next(run, x["horse_code"], today, hist, race_runs)
        if n:
            after.append((top3 + TOP3_BASE * SHRINK_N) / (n + SHRINK_N) - TOP3_BASE)
        fig = race_fig.get(run["key"])
        if fig is not None:
            times.append(fig)
        lost = not run["place"] or run["place"] > 3
        if lost and fig is not None and fig >= FAST and n >= MIN_BEATEN and top3 / n >= PLACED_SHARE:
            strong = 1.0
    out["field_after"] = max(after) if after else 0.0
    out["race_time"] = max(times) if times else 0.0
    out["strong_race"] = strong
    last = runs[-FORM_RUNS:]
    keep = [run for run in last if not ((not run["place"] or run["place"] > 3) and trouble(run["comment"]))]
    out["forgiven_form"] = _form(keep) - _form(last) if last and keep and len(keep) < len(last) else 0.0
    out["last_draw_bad"] = -runs[-1]["draw_edge"] if runs else 0.0
    out["blocked_late"] = 1.0 if runs and blocked_late(runs[-1]["comment"]) else 0.0
    j, lj = x.get("jockey_id"), (runs[-1]["jockey"] if runs else None)
    out["jockey_up"] = math.log(st.jockey.rate(j)) - math.log(st.jockey.rate(lj)) if j and lj and j != lj else 0.0
    return [out[c] for c in CANDIDATES]


def build(races):
    """Replay the bank in date order. Returns one sample per test race."""
    st = model.State()
    hist = defaultdict(list)      # horse code -> earlier runs
    race_runs = {}                # race key -> starters' (code, place)
    race_fig = {}                 # race key -> winner figure minus class par
    class_figs = defaultdict(list)
    samples = []
    for r in races:
        key = (r["date"], r["venue"], r["race_no"])
        starters = [x for x in r["runners"] if x.get("horse_code") and model.started(x)]
        if r["date"] >= TEST_FROM and starters:
            odds = [x.get("win_odds") for x in starters]
            won = [i for i, x in enumerate(starters) if x.get("place") == 1]
            if all(o and o > 0 for o in odds) and len(won) == 1:
                samples.append({"date": r["date"], "y": won[0],
                                "logq": np.log(model.market_probs(odds)),
                                "F": np.array([features(x, r["date"], hist, race_runs, race_fig, st)
                                               for x in starters])})
        # what later races may know about this one, from the state before it
        win = next((x for x in starters if x.get("place") == 1), None)
        fig = st.speed_fig(r, win.get("time"), win.get("weight")) if win else None
        cls = model.class_num(r.get("class"))
        par = class_figs[cls][-CLASS_RACES:] if cls is not None else []
        race_fig[key] = fig - float(np.mean(par)) if fig is not None and len(par) >= CLASS_MIN else None
        if fig is not None and cls is not None:
            class_figs[cls].append(fig)
        field = len(starters)
        for x in starters:
            pos = x.get("place") if isinstance(x.get("place"), int) else None
            hist[x["horse_code"]].append({
                "date": r["date"], "key": key, "place": pos,
                "fscore": 1 - (pos - 1) / max(field - 1, 1) if pos else 0.0,
                "comment": x.get("comment") or "", "draw_edge": st.draw_bias(r, x.get("draw")),
                "jockey": x.get("jockey_id")})
        race_runs[key] = [(x["horse_code"], x.get("place") if isinstance(x.get("place"), int) else None)
                          for x in starters]
        st.update(r)
    return samples


# ------------------------------------------------------------------------- test

def fit(X, y, iters=60):
    """Maximum-likelihood conditional logit, P(i) ∝ exp(X_i · w), by Newton's method."""
    k = X[0].shape[1]
    w = np.zeros(k)
    w[0] = 1.0
    for _ in range(iters):
        g, H = np.zeros(k), np.zeros((k, k))
        for Xi, yi in zip(X, y):
            z = Xi @ w
            p = np.exp(z - z.max())
            p /= p.sum()
            mu = p @ Xi
            g += Xi[yi] - mu
            H -= (Xi * p[:, None]).T @ Xi - np.outer(mu, mu)
        step = np.linalg.solve(H - 1e-6 * np.eye(k), g)
        w = w - step
        if np.abs(step).max() < 1e-9:
            break
    return w


def logloss(Xi, yi, w):
    z = Xi @ w
    return float(np.log(np.exp(z - z.max()).sum()) + z.max() - z[yi])


def forward(samples, col, placebo=False):
    rng = np.random.default_rng(20261111)
    y = [s["y"] for s in samples]
    f = [s["F"][:, CANDIDATES.index(col)] for s in samples]
    if placebo:
        f = [rng.permutation(v) for v in f]
    base = [s["logq"][:, None] for s in samples]
    cand = [np.column_stack([s["logq"], v]) for s, v in zip(samples, f)]
    edges = np.linspace(FIRST_TRAIN, len(samples), FOLDS + 1).astype(int)
    gains, w = [], None
    for lo, hi in zip(edges[:-1], edges[1:]):
        wb, w = fit(base[:lo], y[:lo]), fit(cand[:lo], y[:lo])
        gains += [logloss(base[i], y[i], wb) - logloss(cand[i], y[i], w) for i in range(lo, hi)]
    g = np.array(gains)
    z = float(g.mean() / (g.std(ddof=1) / math.sqrt(len(g))))
    return {"gain": round(float(g.mean()), 5), "z": round(z, 2), "races": len(g),
            "a": round(float(w[0]), 3), "b": round(float(w[1]), 3)}


def coverage(races):
    days = {r["date"] for r in races}
    seasons = [sum(1 for d in days if s <= d <= e) for s, e in SEASONS]
    test = [r for r in races if r["date"] >= TEST_FROM]
    share = sum(1 for r in test if r.get("has_comment")) / max(len(test), 1)
    return {"meetings_per_season": seasons, "races": len(races), "test_races": len(test),
            "comment_share": round(share, 3)}


def run_test():
    _guard()
    races = [r for r in store.load_all_results() if BANK_FROM <= r["date"] <= BANK_TO]
    cov = coverage(races)
    print("coverage:", cov)
    if min(cov["meetings_per_season"]) < 60 or cov["comment_share"] < MIN_COMMENTS:
        raise SystemExit("bank incomplete: download the rest before testing (no result looked at)")
    samples = build(races)
    out = {"registered": REGISTERED, "run": date.today().isoformat(), "coverage": cov,
           "samples": len(samples), "results": {}}
    for c in CANDIDATES:
        col = np.concatenate([s["F"][:, CANDIDATES.index(c)] for s in samples])
        res = forward(samples, c)
        res["placebo_z"] = forward(samples, c, placebo=True)["z"]
        res["nonzero"] = round(float(np.mean(col != 0)), 3)
        res["passed"] = res["gain"] > 0 and res["z"] >= Z_PASS and res["placebo_z"] < PLACEBO_MAX
        out["results"][c] = res
        print(f"  {c:14s} {NAMES_ZH[c]}: gain {res['gain']:+.5f}  z {res['z']:+.2f}  "
              f"placebo z {res['placebo_z']:+.2f}  b {res['b']:+.3f}  "
              f"non-zero {res['nonzero']:.0%}  -> {'PASS' if res['passed'] else 'fail'}")
    store.write_json(store.DATA / "oldbank_result.json", out)
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "download":
        download(date.fromisoformat(sys.argv[2]), date.fromisoformat(sys.argv[3]))
    elif what == "test":
        print(json.dumps(run_test()["results"], ensure_ascii=False, indent=1))
    else:
        print(__doc__)
