"""Daily self-check: data health, walk-forward accuracy vs. the market,
calibration, factor contributions and guarded auto-tuning.

    python -m hkjc.selfcheck

Writes docs/data/selfcheck.json (shown on the dashboard) and, when a
setting clearly beats the current one, data/model_config.json (read by
model.train).
"""
from __future__ import annotations

import math
import os
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import numpy as np
import requests

from . import livetest, model, pools, scraper, store
from .backfill import HKT, today_hkt

CLOCK_LIVE = "2026-11-05"   # meetings before this ran on GitHub cron, not the 5-minute clock
MAX_UPDATE_GAP = 20         # minutes; one hung update (15-minute limit) plus the next tick
REPO = os.environ.get("GITHUB_REPOSITORY", "kelvin1997121/hkjc-quant")
TRACKWORK_FROM = "2026-11-08"   # first meeting whose trackwork and vet records were saved

GROUPS = {
    "速度/表現評分": ["speed_best", "speed_last", "speed_missing", "perf_best", "perf_last",
                 "handicap_edge"],
    "近況": ["form", "dist_form", "class_drop", "layoff", "debut", "bw_change", "form_trend"],
    "跑法/步速": ["early", "early_x_pace", "early_x_today", "early_x_course", "closing"],
    "檔位": ["draw_bias", "draw_x_today"],
    "騎練": ["jockey", "trainer", "jockey_form", "trainer_form", "combo", "trainer_venue"],
    "市場往績": ["mkt_rating", "vs_mkt"],
    "分段時間": ["late_speed", "early_speed"],
    "負磅/評分": ["weight_rel", "rating_move"],
    "配備": ["gear_first", "gear_off", "gear_back"],
    "上仗走外疊": ["run_wide"],
    "試閘": ["trial_score", "trial_count"],
}
CANDIDATES = [{"l2": l2, "stages": st} for l2 in (0.02, 0.05, 0.2) for st in (1, 3)]
MIN_GAIN = 0.002  # log-loss improvement needed before a setting is switched

# Self-improvement. A factor on trial (model.TRIAL_FEATURES), or the removal of a factor group,
# is judged only on races run after it was registered, so no race helps both choose a change
# and confirm it. Each test is judged when it reaches 300, 500 and 800 new races, and changes
# the model only at z >= 2.5 (with about 15 tests running, roughly one false change in 400).
LEARN_PATH = store.DATA / "learning.json"
LEARN_FROM = "2026-11-08"
LOOKS = (300, 500, 800)
Z_DECIDE = 2.5
STATUS_ZH = {"testing": "測試中", "adopted": "已採用", "rejected": "已放棄", "retired": "已刪走",
             "kept": "保留"}


def _samples():
    races = store.load_all_results()
    _, samples = model.build_dataset(races)
    start = np.datetime64(samples[0]["date"]) + np.timedelta64(model.WARMUP_DAYS, "D")
    return races, [s for s in samples if np.datetime64(s["date"]) >= start]


def _fit(train, cols, cfg):
    saved = model.FEATURES
    model.FEATURES = cols
    try:
        return model.fit(train, l2=cfg["l2"], stages=cfg["stages"])
    finally:
        model.FEATURES = saved


def walk_forward(samples, cols, cfg, folds=4, test_frac=0.4, with_blend=False):
    """Train on everything before each fold, score the fold. Returns per-race rows."""
    F = model.FEATURES
    idx = [F.index(c) for c in cols]
    S = [dict(s, X=s["X"][:, idx]) for s in samples]
    n = len(S)
    edges = np.linspace(int(n * (1 - test_frac)), n, folds + 1).astype(int)
    rows, weights = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        p = _fit(S[:lo], cols, cfg)
        bw = None
        if with_blend:
            cut = int(lo * 0.75)
            bw = model.fit_blend(_fit(S[:cut], cols, cfg), S[cut:lo])
            weights.append(bw)
        for s in S[lo:hi]:
            pm = model.predict(p, s["X"])
            q = model.market_probs(s["odds"])
            r = {"ll": -math.log(max(pm[s["y"]], 1e-9)), "hit": int(np.argmax(pm)) == s["y"],
                 "p_win": float(pm[s["y"]]), "probs": pm, "y": s["y"],
                 "odds": s["odds"], "order": s.get("order") or [s["y"]],
                 "date": s["date"], "nos": s["nos"], "dividends": s["dividends"]}
            if q is not None:
                r["ll_mkt"] = -math.log(q[s["y"]])
                r["fav_hit"] = int(np.argmax(q)) == s["y"]
                if bw:
                    pb = model.blend(pm, q, bw["a"], bw["b"])
                    r["blend_probs"] = pb
                    r["ll_blend"] = -math.log(pb[s["y"]])
                    r["blend_hit"] = int(np.argmax(pb)) == s["y"]
                    order = list(np.argsort(-pb))
                    r["winner_rank"] = order.index(s["y"]) + 1
                    r["first_placed"] = order[0] in (s.get("order") or [s["y"]])
                    r["first_odds"] = float(s["odds"][order[0]])
                    r["first_no"] = s["nos"][order[0]]
                    r["second_no"] = s["nos"][order[1]]
            rows.append(r)
    return rows, weights


def _summary(rows):
    ll = np.array([r["ll"] for r in rows])
    out = {"races": len(rows), "logloss": round(float(ll.mean()), 4),
           "top_hit": round(float(np.mean([r["hit"] for r in rows])), 4)}
    mk = [r for r in rows if "ll_mkt" in r]
    if mk:
        out["market_logloss"] = round(float(np.mean([r["ll_mkt"] for r in mk])), 4)
        out["favourite_hit"] = round(float(np.mean([r["fav_hit"] for r in mk])), 4)
    bl = [r for r in mk if "ll_blend" in r]
    if bl:
        gain = np.array([r["ll_mkt"] - r["ll_blend"] for r in bl])
        out["blend_hit"] = round(float(np.mean([r["blend_hit"] for r in bl])), 4)
        out["winner_in_top2"] = round(float(np.mean([r["winner_rank"] <= 2 for r in bl])), 4)
        out["winner_in_top3"] = round(float(np.mean([r["winner_rank"] <= 3 for r in bl])), 4)
        out["first_placed"] = round(float(np.mean([r["first_placed"] for r in bl])), 4)
        out["blend_gain"] = round(float(gain.mean()), 4)
        out["blend_gain_ci"] = round(float(1.96 * gain.std() / math.sqrt(len(gain))), 4)
    return out


def calibration(rows, bins=(0, .05, .1, .15, .2, .3, .45, 1.01)):
    """Predicted vs. actual win rate per probability band (all runners)."""
    pred, won = [], []
    for r in rows:
        for i, p in enumerate(r["probs"]):
            pred.append(p)
            won.append(1 if i == r["y"] else 0)
    pred, won = np.array(pred), np.array(won)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (pred >= lo) & (pred < hi)
        if m.sum() >= 30:
            out.append({"band": f"{lo:.0%}-{min(hi, 1):.0%}", "n": int(m.sum()),
                        "predicted": round(float(pred[m].mean()), 4),
                        "actual": round(float(won[m].mean()), 4)})
    return out


def payout(div, pool, *nos):
    """Return per $1 from real dividends: 0 if the bet lost, None if that pool wasn't paid."""
    rows = (div or {}).get(pool)
    if not rows or any(a is None for _, a in rows):
        return None
    want = set(map(str, nos))
    return sum(a for c, a in rows if set(c.split(",")) == want) / 10


def _returns(name, bets):
    """Flat-stake record of [(date, return per $1), ...]. The range resamples whole meetings."""
    days = defaultdict(lambda: [0, 0.0])
    for day, back in bets:
        days[day][0] += 1
        days[day][1] += back
    stake, back = np.array(list(days.values())).T
    pick = np.random.default_rng(1108).integers(0, len(stake), (2000, len(stake)))
    boot = back[pick].sum(1) / stake[pick].sum(1) - 1
    return {"name": name, "bets": len(bets), "hit": round(sum(b > 0 for _, b in bets) / len(bets), 4),
            "roi": round(float(back.sum() / stake.sum() - 1), 4),
            "low": round(float(np.percentile(boot, 2.5)), 4), "high": round(float(np.percentile(boot, 97.5)), 4)}


def bet_returns(rows, races):
    """Each way of backing the published picks, at real dividends on races the model hadn't
    seen, then the market's top two on all history. Final odds choose the horses, so someone
    betting before the off couldn't quite do this."""
    from .analyze import LONGSHOT_ODDS, LONGSHOT_RANK
    bets = defaultdict(list)

    def add(name, day, back):
        if back is not None:
            bets[name].append((day, back))

    for r in rows:
        if "first_no" not in r or not r["dividends"]:
            continue
        d, day, a, b = r["dividends"], r["date"], r["first_no"], r["second_no"]
        add("首選 獨贏", day, payout(d, "獨贏", a))
        add("首選 位置", day, payout(d, "位置", a))
        add("首選+次選 連贏", day, payout(d, "連贏", a, b))
        add("首選+次選 位置Q", day, payout(d, "位置Q", a, b))
        ranks = (-r["probs"]).argsort().argsort() + 1
        for i, no in enumerate(r["nos"]):
            if r["odds"][i] >= LONGSHOT_ODDS and ranks[i] <= LONGSHOT_RANK:
                add("冷門馬 獨贏", day, payout(d, "獨贏", no))
                add("冷門馬 位置", day, payout(d, "位置", no))
                if no != a:
                    add("首選+冷門馬 連贏", day, payout(d, "連贏", a, no))
                    add("首選+冷門馬 位置Q", day, payout(d, "位置Q", a, no))
    unseen = [_returns(k, v) for k, v in bets.items()]

    bets = defaultdict(list)
    for race in races:
        run = sorted((x for x in race["runners"] if x.get("win_odds") and model.started(x)),
                     key=lambda x: (x["win_odds"], x["no"]))
        if race.get("dividends") and len(run) >= 5:
            add("頭兩熱門 連贏", race["date"], payout(race["dividends"], "連贏", run[0]["no"], run[1]["no"]))
            add("頭兩熱門 位置Q", race["date"], payout(race["dividends"], "位置Q", run[0]["no"], run[1]["no"]))
    return {"unseen": unseen, "history": [_returns(k, v) for k, v in bets.items()]}


def pick_bands(rows):
    """First choice's record by the 把握 band on the race (walk-forward, final odds)."""
    from .analyze import confidence_band
    out = {}
    for r in rows:
        if "first_odds" not in r:
            continue
        b = out.setdefault(confidence_band(r["first_odds"]),
                           {"n": 0, "wins": 0, "placed": 0, "win_back": 0.0, "place_back": 0.0})
        b["n"] += 1
        b["wins"] += r["blend_hit"]
        b["placed"] += r["first_placed"]
        b["win_back"] += payout(r["dividends"], "獨贏", r["first_no"]) or 0.0
        b["place_back"] += payout(r["dividends"], "位置", r["first_no"]) or 0.0
    return {k: {"n": v["n"], "win": round(v["wins"] / v["n"], 4), "place": round(v["placed"] / v["n"], 4),
                "roi": round(v["win_back"] / v["n"] - 1, 4),
                "place_roi": round(v["place_back"] / v["n"] - 1, 4)} for k, v in out.items()}


def longshot_stats(rows):
    """Record of the 冷門馬 rule (odds >= 10, model top 5) vs. all odds >= 10 runners,
    at final odds on walk-forward test races."""
    from .analyze import LONGSHOT_ODDS, LONGSHOT_RANK
    picked, base, others = [], [], []
    for r in rows:
        odds = r["odds"]
        if not (odds > 0).all():
            continue
        ranks = (-r["probs"]).argsort().argsort() + 1
        for i, o in enumerate(odds):
            won = payout(r["dividends"], "獨贏", r["nos"][i])
            placed = payout(r["dividends"], "位置", r["nos"][i])
            if o < LONGSHOT_ODDS or won is None or placed is None:
                continue
            row = (o, won, placed)
            base.append(row)
            (picked if ranks[i] <= LONGSHOT_RANK else others).append(row)

    def summary(sel):
        if not sel:
            return None
        k = len(sel)
        return {"n": k, "win": round(sum(w > 0 for _, w, _ in sel) / k, 4),
                "top3": round(sum(p > 0 for _, _, p in sel) / k, 4),
                "avg_odds": round(float(np.mean([o for o, _, _ in sel])), 1),
                "win_roi": round(sum(w for _, w, _ in sel) / k - 1, 4),
                "place_roi": round(sum(p for _, _, p in sel) / k - 1, 4)}

    # the rule picks shorter prices than other longshots, so only a like-for-like comparison is fair
    same_price = []
    for lo, hi in ((10, 20), (20, 50)):
        a = [p for o, _, p in picked if lo <= o < hi]
        b = [p for o, _, p in others if lo <= o < hi]
        if a and b:
            same_price.append({"band": f"{lo}-{hi}", "n": len(a),
                               "picked": round(sum(p > 0 for p in a) / len(a), 4),
                               "others": round(sum(p > 0 for p in b) / len(b), 4)})
    return {"rule": f"獨贏 {LONGSHOT_ODDS} 倍以上而模型排頭 {LONGSHOT_RANK}",
            "picked": summary(picked), "all_longshots": summary(base), "same_price": same_price}


def steam_stats():
    """Win rate of horses whose odds shortened from race day's first snapshot to the off.

    Past results only have final odds, so this builds up meeting by meeting until
    there is enough to test as a model factor.
    """
    from .analyze import STEAM_RATIO
    groups = {"steam": [0, 0], "steady": [0, 0], "drift": [0, 0]}
    meetings = 0
    for p in sorted((store.DATA / "odds").glob("20*_*.json")):
        day, venue = p.stem.split("_")
        hist = store.read_json(p, {}).get("history", {})
        for race in store.read_json(store.result_path(day, venue), []) or []:
            snaps = [h["win"] for h in hist.get(str(race["race_no"]), [])
                     if datetime.fromisoformat(h["t"]).astimezone(HKT).date().isoformat() == day]
            if not snaps:
                continue
            for x in race["runners"]:
                was, final = snaps[0].get(str(x["no"])), x.get("win_odds")
                if not was or not final:
                    continue
                g = "steam" if was / final >= STEAM_RATIO else "drift" if final / was >= STEAM_RATIO else "steady"
                groups[g][0] += 1
                groups[g][1] += x.get("place") == 1
        meetings += 1
    return {"meetings": meetings,
            **{g: {"n": n, "wins": w} for g, (n, w) in groups.items()}}


def meeting_updates(doc):
    """Did a finished meeting get its live updates? The longest gap catches what a
    count hides, such as plenty of updates early and none after race 1."""
    day = doc["meeting"]["date"]
    posts = sorted(r["post_time"] for r in doc["races"] if r.get("post_time"))
    if not posts:
        return {"name": "賽馬日更新", "ok": False, "detail": f"{day} 冇開跑時間,計唔到更新次數"}
    mins = lambda hm: int(hm[:2]) * 60 + int(hm[3:5])
    start, end = mins(posts[0]) - 60, mins(posts[-1]) + 45   # the window the ticker runs every 5 min
    times = [datetime.fromisoformat(t).astimezone(HKT) for t in doc.get("runs", [])]
    inside = sorted(m for m in (t.hour * 60 + t.minute for t in times if t.date().isoformat() == day)
                    if start <= m <= end)
    edges = [start, *inside, end]
    gap = max(b - a for a, b in zip(edges, edges[1:]))
    n = len(inside)
    if day < CLOCK_LIVE:  # this meeting's fault is known; only a meeting since then can prove the fix
        upcoming = (store.read_json(store.DOCS / "data" / "schedule.json", {}) or {}).get("date", "下一個賽馬日")
        return {"name": "賽馬日更新", "ok": None,
                "detail": f"{day} 只更新 {n} 次,係修正前嘅舊故障;新時鐘要到 {upcoming} 賽馬日先驗證到"}
    if n >= (end - start) / 5 / 2 and gap <= MAX_UPDATE_GAP:
        return {"name": "賽馬日更新", "ok": True, "detail": f"{day} 更新 {n} 次,最長相隔 {gap} 分鐘"}
    return {"name": "賽馬日更新", "ok": False,
            "detail": f"{day} 只更新 {n} 次,最長相隔 {gap} 分鐘(應該約 5 分鐘一次)"}


def ticker_check():
    """Is the cloud clock running now, and has it been handing over to itself?"""
    url = f"https://api.github.com/repos/{REPO}/actions/workflows/ticker.yml/runs?per_page=20"
    token = os.environ.get("GITHUB_TOKEN")
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {token}"} if token else {}, timeout=20)
        r.raise_for_status()
        runs = r.json()["workflow_runs"]
    except (requests.RequestException, KeyError, ValueError) as e:
        return {"name": "雲端時鐘", "ok": False, "detail": f"查唔到狀態({type(e).__name__})"}
    if not any(x["status"] != "completed" for x in runs):
        return {"name": "雲端時鐘", "ok": False, "detail": "停咗,賽馬日唔會自動更新"}
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    relays = sum(1 for x in runs if (x.get("triggering_actor") or {}).get("login") == "github-actions[bot]"
                 and datetime.fromisoformat(x["created_at"]) > since)
    return {"name": "雲端時鐘", "ok": True,
            "detail": f"行緊,最近 24 小時接力 {relays} 次" if relays else "行緊,最近 24 小時未接力過"}


def odds_check(meeting_file):
    """Did the last meeting get live odds? Any old file would otherwise pass for ever."""
    stem = meeting_file.stem
    history = (store.read_json(store.DATA / "odds" / f"{stem}.json") or {}).get("history", {})
    total = len(store.read_json(meeting_file)["races"])
    return {"name": "即時賠率", "ok": bool(total) and len(history) >= 0.8 * total,
            "detail": f"{stem[:10]} 有 {len(history)}/{total} 場賠率紀錄"}


def trackwork_check(past):
    """A log that quietly stops would only be noticed months later, when it's needed."""
    folder = store.DATA / "trackwork"
    done = [p for p in past if p.stem[:10] >= TRACKWORK_FROM]
    if not done:
        saved = sum(1 for _ in folder.glob("20*_*.json"))
        return {"name": "晨操紀錄", "ok": None,
                "detail": f"由 {TRACKWORK_FROM} 開始,跑完先驗證到(已儲存 {saved} 個賽馬日)"}
    stem = done[-1].stem
    races = (store.read_json(folder / f"{stem}.json") or {}).get("races", {})
    total = len(store.read_json(done[-1])["races"])
    horses = sum(len(r["horses"]) for r in races.values())
    return {"name": "晨操紀錄", "ok": bool(total) and len(races) >= total,
            "detail": f"{stem[:10]} 儲咗 {len(races)}/{total} 場、{horses} 匹馬"}


def data_health(races):
    today = today_hkt()
    last = max(r["date"] for r in races)
    meetings = sorted({(r["date"], r["venue"]) for r in races})
    recent = [r for r in races if r["date"] >= (np.datetime64(last) - np.timedelta64(60, "D")).astype(str)]
    checks = []
    try:
        # today's date may be listed for an overseas simulcast or an unfinished meeting
        listed = [d for d in scraper.list_result_dates() if d < today]
        missing = [d.isoformat() for d in listed[:10]
                   if not store.have_results(d.isoformat())]
    except RuntimeError:
        missing = None
    checks.append({"name": "賽果更新", "ok": missing == [] if missing is not None else False,
                   "detail": "最近 10 個賽馬日齊全" if missing == [] else
                   (f"欠 {', '.join(missing)}" if missing else "未能連接馬會網站")})
    sect = np.mean([bool(r.get("has_sect")) for r in recent]) if recent else 0
    checks.append({"name": "分段時間", "ok": bool(sect >= 0.9),  # a numpy bool isn't valid JSON
                   "detail": f"近 60 日 {sect:.0%} 賽事有分段時間"})
    trial_files = sorted((store.DATA / "trials").glob("20*.json"))
    last_trial = trial_files[-1].stem if trial_files else None
    checks.append({"name": "試閘數據", "ok": bool(last_trial) and (today - date.fromisoformat(last_trial)).days <= 21,
                   "detail": f"最近一次:{last_trial}" if last_trial else "未有試閘紀錄"})
    past = [p for p in sorted((store.DOCS / "data").glob("20*_*.json"))
            if p.stem[:10] < today.isoformat()]
    if past:
        checks.append(odds_check(past[-1]))
        checks.append(meeting_updates(store.read_json(past[-1])))
    checks.append(ticker_check())
    checks.append(trackwork_check(past))
    void = sum(1 for r in recent for x in r["runners"]
               if x.get("place") and not x.get("time"))
    checks.append({"name": "賽果解析", "ok": void == 0,
                   "detail": "完成時間齊全" if void == 0 else f"{void} 匹完成馬冇時間"})
    return {"checks": checks, "last_result": last, "meetings": len(meetings),
            "races": len(races)}


def forward_gain(samples, start, base_cols, cand_cols, cfg, shuffle=None):
    """Per-race log-loss gain of cand over base on samples[start:], each third predicted by
    models trained only on the races before it. shuffle: columns to permute within each race."""
    rng = np.random.default_rng(1108)
    edges = np.linspace(start, len(samples), 4).astype(int)
    F = model.FEATURES
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        fits = {}
        for name, cols in (("base", base_cols), ("cand", cand_cols)):
            idx = [F.index(c) for c in cols]
            train = [dict(s, X=s["X"][:, idx]) for s in samples[:lo]]
            fits[name] = (idx, _fit(train, cols, cfg))
        for s in samples[lo:hi]:
            X = s["X"]
            if shuffle:
                X = X.copy()
                for c in shuffle:
                    X[:, F.index(c)] = rng.permutation(X[:, F.index(c)])
            ll = {k: -math.log(max(model.predict(p, X[:, idx])[s["y"]], 1e-9)) for k, (idx, p) in fits.items()}
            out.append(ll["base"] - ll["cand"])
    g = np.array(out)
    z = float(g.mean() / (g.std() / math.sqrt(len(g)))) if len(g) > 1 and g.std() > 0 else 0.0
    return round(float(g.mean()), 4), round(z, 2)


def learning(samples, cfg):
    """Run any forward-only test that has reached its next look; adopt or retire on a pass."""
    log = store.read_json(LEARN_PATH, {}) or {}
    tests = log.setdefault("tests", {})
    for f, since in model.TRIAL_FEATURES.items():
        tests.setdefault(f"add:{f}", {"kind": "add", "cols": [f], "since": since, "status": "testing",
                                      "looks": []})
    for g, cols in GROUPS.items():
        tests.setdefault(f"drop:{g}", {"kind": "drop", "cols": cols, "since": LEARN_FROM, "status": "testing",
                                       "looks": []})
    today = today_hkt().isoformat()
    added, removed = list(cfg.get("added", [])), list(cfg.get("removed", []))
    for name, t in tests.items():
        start = next((i for i, s in enumerate(samples) if s["date"] >= t["since"]), len(samples))
        t["races"] = len(samples) - start
        if t["status"] != "testing":
            continue
        done = [lk["at"] for lk in t["looks"]]
        due = [n for n in LOOKS if t["races"] >= n and n not in done]
        if not due:
            continue
        base = model.active_features({**cfg, "added": added, "removed": removed})
        if t["kind"] == "add":
            cand = base + [c for c in t["cols"] if c not in base]
            gain, z = forward_gain(samples, start, base, cand, cfg)
            _, placebo = forward_gain(samples, start, base, cand, cfg, shuffle=t["cols"])
        else:
            cand = [c for c in base if c not in t["cols"]]
            gain, z = forward_gain(samples, start, base, cand, cfg)
            placebo = None
        t["looks"].append({"at": max(due), "date": today, "gain": gain, "z": z, "placebo_z": placebo})
        last = max(due) == LOOKS[-1]
        if t["kind"] == "add":
            if z >= Z_DECIDE and placebo < 1.0:
                t["status"] = "adopted"
                added += [c for c in t["cols"] if c not in added]
            elif z <= -Z_DECIDE or last:
                t["status"] = "rejected"
        else:
            if z >= Z_DECIDE:
                t["status"] = "retired"
                removed += [c for c in t["cols"] if c not in removed]
            elif z <= -Z_DECIDE or last:
                t["status"] = "kept"
        if t["status"] != "testing":
            t["decided"] = today
        print(f"  learning {name}: {t['races']} new races, gain {gain:+.4f}, z {z:+.2f} -> {t['status']}")
    log["checked"] = today
    store.write_json(LEARN_PATH, log)
    if (added, removed) != (cfg.get("added", []), cfg.get("removed", [])):
        store.write_json(store.DATA / "model_config.json", {**cfg, "added": added, "removed": removed})
    zh = dict(model_zh(), **{f"drop:{g}": f"刪走:{g}" for g in GROUPS})
    return [{"test": zh.get(name, name), "status": STATUS_ZH[t["status"]], "races": t["races"],
             "next": next((n for n in LOOKS if n > t["races"]), None) if t["status"] == "testing" else None,
             "last": t["looks"][-1] if t["looks"] else None}
            for name, t in tests.items()]


def model_zh():
    from .analyze import FEATURE_ZH
    return {f"add:{f}": f"加入:{FEATURE_ZH.get(f, f)}" for f in model.TRIAL_FEATURES}


def live_check(tr):
    """Are the frozen pre-race first choices doing as well as the final favourite of the same races?

    Changed 2026-11-12 from a comparison with the backtest: that is scored on final odds, which
    include money placed after our picks are frozen (11 Nov: in 3 of 9 races the final favourite
    was another horse than at the freeze), so live picks are expected to sit below it. Over the
    first 29 races the final favourite itself won 4, which the old check would have flagged too.
    Race by race against the favourite, a bad run for favourites is not mistaken for a fault."""
    n = tr.get("races", 0)
    since = "/".join(str(int(v)) for v in tr["from"].split("-")[1:]) if tr.get("from") else ""
    if n < 30:
        return {"name": "實戰表現", "ok": None,
                "detail": f"實戰 {n} 場" + (f"(由 {since} 起)" if since else "") + ",夠 30 場先比較"}
    zs = [tr.get("z_win_vs_fav") or 0.0, tr.get("z_place_vs_fav") or 0.0]
    ok = min(zs) > -2
    return {"name": "實戰表現", "ok": ok,
            "detail": f"實戰 {n} 場:首選勝 {tr['wins'] / n:.0%}、入位 {tr['places'] / n:.0%};"
                      f"同場最後熱門勝 {tr['fav_wins'] / n:.0%}、入位 {tr['fav_places'] / n:.0%}"
                      + ("" if ok else ",首選明顯差過最後熱門,要檢查")}


def load_config():
    return store.read_json(store.DATA / "model_config.json", {"l2": 0.05, "stages": 1})


def run():
    model.FEATURES = list(model.ALL_FEATURES)   # build every factor; each test picks its columns
    races, samples = _samples()
    current = load_config()
    F = model.active_features(current)
    print(f"self-check on {len(samples)} races, config {current}")

    base_rows, weights = walk_forward(samples, F, current, with_blend=True)
    base = _summary(base_rows)
    print("  base", base)

    # factor groups: how much worse does log loss get without each group?
    contrib = []
    for name, cols in GROUPS.items():
        keep = [c for c in F if c not in cols]
        rows, _ = walk_forward(samples, keep, current)
        delta = float(np.mean([r["ll"] for r in rows])) - base["logloss"]
        contrib.append({"group": name, "delta_logloss": round(delta, 4)})
        print(f"  without {name}: {delta:+.4f}")
    contrib.sort(key=lambda c: -c["delta_logloss"])

    # guarded auto-tuning
    tried = []
    for cfg in CANDIDATES:
        if cfg == {k: current.get(k) for k in cfg}:
            tried.append({**cfg, **{k: base[k] for k in ("logloss", "top_hit")}})
            continue
        s = _summary(walk_forward(samples, F, cfg)[0])
        tried.append({**cfg, "logloss": s["logloss"], "top_hit": s["top_hit"]})
    best = min(tried, key=lambda t: t["logloss"])
    switched = False
    if best["logloss"] < base["logloss"] - MIN_GAIN and best["top_hit"] >= base["top_hit"] - 0.01:
        store.write_json(store.DATA / "model_config.json", {**current, "l2": best["l2"], "stages": best["stages"]})
        switched = True
    print("  auto-tune best", best, "switched" if switched else "kept")

    learned = learning(samples, load_config())

    # live record of frozen pre-race picks
    tr = store.read_json(store.DOCS / "data" / "track_record.json", {}) or {}

    notes = []
    if base.get("blend_gain") is not None:
        if base["blend_gain"] - base["blend_gain_ci"] > 0:
            notes.append("模型可以改善市場賠率(綜合後 log loss 較低,超出誤差範圍),值博標記有參考價值。")
        else:
            notes.append("模型暫時未能提供市場賠率以外嘅資訊(綜合增益未超出誤差範圍),"
                         "所以唔建議單靠模型落注。")
    weak = [c["group"] for c in contrib if c["delta_logloss"] < -0.001]
    if weak:
        notes.append(f"以下因素組拎走之後反而更準,可以考慮刪減:{'、'.join(weak)}。")
    strong = [c["group"] for c in contrib[:3] if c["delta_logloss"] > 0.001]
    if strong:
        notes.append(f"最有用嘅因素組:{'、'.join(strong)}。")
    cal = calibration(base_rows)
    off = [c for c in cal if abs(c["actual"] - c["predicted"]) > 0.05 and c["n"] >= 100]
    if off:
        notes.append("部分機率區間校準有偏差:" + ";".join(
            f"{c['band']} 預測 {c['predicted']:.0%} 實際 {c['actual']:.0%}" for c in off) + "。")
    ls = longshot_stats(base_rows)
    if ls["picked"] and ls["all_longshots"]:
        p = ls["picked"]
        same = "、".join(f"{b['band']} 倍 {b['picked']:.0%}(其他馬 {b['others']:.0%})" for b in ls["same_price"])
        notes.append(f"冷門馬規則({ls['rule']}):同賠率比較入位率,{same};"
                     f"平注回報獨贏 {p['win_roi']:+.0%}、位置 {p['place_roi']:+.0%}。")
    if tr.get("races", 0) < 50:
        notes.append(f"實戰往績只有 {tr.get('races', 0)} 場,樣本太少,未能下結論。")
    try:  # registered 2026-11-08; judged only at its looks (see pools.py)
        pool_test = pools.status()
        done = [p for p in pool_test["pools"].values() if p.get("verdict")]
        if done:
            notes.append("彩池測試(位置/連贏/位置Q 對比獨贏推算):" + ";".join(
                f"{p['name']} {p['look']} 場每場平均 {p['profit_per_race']:+.1f} 元,z {p['z']:+.1f},{p['verdict']}"
                for p in done) + "。")
        else:
            notes.append(f"彩池測試:已收集 {pool_test['races']} 場開跑前賠率,"
                         f"{pools.LOOKS[0]} 場先判斷位置、連贏、位置Q 有冇比獨贏彩池更值博。")
    except Exception as e:  # a new check must not cost the rest of the report
        pool_test = {"error": repr(e)}

    try:   # new-information test registered 2026-11-12 (livetest.py)
        live_test = livetest.status()
        rows = live_test["candidates"].values()
        done = [c for c in rows if c.get("verdict")]
        if done:
            notes.append("新資料測試(對比開跑前賠率):" + ";".join(
                f"{c['name']} {c['look']} 場 z {c['z']:+.1f}(打亂後 {c['placebo_z']:+.1f}),{c['verdict']}"
                for c in done) + "。")
        else:
            notes.append(f"新資料測試(由 {live_test['from']} 起):" + "、".join(
                f"{c['name']} {c['races']} 場" for c in rows)
                + f",{livetest.LOOKS[0]} 場先判斷有冇開跑前賠率以外嘅資訊。")
    except Exception as e:
        live_test = {"error": repr(e)}

    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "health": data_health(races),
        "learning": learned,
        "walk_forward": base,
        "blend_weights": weights,
        "calibration": cal,
        "factor_groups": contrib,
        "tuning": {"current": current, "tried": tried, "switched": switched,
                   "chosen": {"l2": best["l2"], "stages": best["stages"]} if switched else current},
        "live": {k: tr.get(k) for k in ("from", "races", "wins", "places", "fav_wins", "fav_places",
                                        "z_win_vs_fav", "z_place_vs_fav", "flat_roi")},
        "longshots": ls,
        "pick_bands": pick_bands(base_rows),
        "bet_returns": bet_returns(base_rows, races),
        "steam": steam_stats(),
        "pool_test": pool_test,
        "live_test": live_test,
        "notes": notes,
    }
    report["health"]["checks"].append(live_check(tr))
    store.write_json(store.DOCS / "data" / "selfcheck.json", report)
    print("  wrote selfcheck.json")
    return report


if __name__ == "__main__":
    run()
