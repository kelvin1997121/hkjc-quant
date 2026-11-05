"""Daily self-check: data health, walk-forward accuracy vs. the market,
calibration, factor contributions and guarded auto-tuning.

    python -m hkjc.selfcheck

Writes docs/data/selfcheck.json (shown on the dashboard) and, when a
setting clearly beats the current one, data/model_config.json (read by
model.train).
"""
from __future__ import annotations

import math
from datetime import date, datetime, timezone

import numpy as np

from . import model, scraper, store
from .backfill import HKT, today_hkt

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
}
CANDIDATES = [{"l2": l2, "stages": st} for l2 in (0.02, 0.05, 0.2) for st in (1, 3)]
MIN_GAIN = 0.002  # log-loss improvement needed before a setting is switched


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
                 "odds": s["odds"], "order": s.get("order") or [s["y"]]}
            if q is not None:
                r["ll_mkt"] = -math.log(q[s["y"]])
                r["fav_hit"] = int(np.argmax(q)) == s["y"]
                if bw:
                    pb = model.blend(pm, q, bw["a"], bw["b"])
                    r["ll_blend"] = -math.log(pb[s["y"]])
                    r["blend_hit"] = int(np.argmax(pb)) == s["y"]
                    order = list(np.argsort(-pb))
                    r["winner_rank"] = order.index(s["y"]) + 1
                    r["first_placed"] = order[0] in (s.get("order") or [s["y"]])
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


def longshot_stats(rows):
    """Record of the 冷門馬 rule (odds >= 10, model top 5) vs. all odds >= 10 runners,
    at final odds on walk-forward test races."""
    from .analyze import LONGSHOT_ODDS, LONGSHOT_RANK
    picked, base = [], []
    for r in rows:
        odds = r["odds"]
        if not (odds > 0).all():
            continue
        ranks = (-r["probs"]).argsort().argsort() + 1
        top3 = set(r["order"])
        for i, o in enumerate(odds):
            if o < LONGSHOT_ODDS:
                continue
            row = (o, i == r["y"], i in top3)
            base.append(row)
            if ranks[i] <= LONGSHOT_RANK:
                picked.append(row)

    def summary(sel):
        if not sel:
            return None
        k = len(sel)
        return {"n": k, "win": round(sum(w for _, w, _ in sel) / k, 4),
                "top3": round(sum(t for _, _, t in sel) / k, 4),
                "avg_odds": round(float(np.mean([o for o, _, _ in sel])), 1),
                "win_roi": round(sum(o for o, w, _ in sel if w) / k - 1, 4)}

    return {"rule": f"獨贏 {LONGSHOT_ODDS} 倍以上而模型排頭 {LONGSHOT_RANK}",
            "picked": summary(picked), "all_longshots": summary(base)}


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
    checks.append({"name": "分段時間", "ok": sect >= 0.9,
                   "detail": f"近 60 日 {sect:.0%} 賽事有分段時間"})
    odds_files = sorted((store.DATA / "odds").glob("*.json"))
    last_odds = odds_files[-1].stem if odds_files else None
    checks.append({"name": "即時賠率", "ok": bool(last_odds),
                   "detail": f"最近一次:{last_odds}" if last_odds else "未有賠率紀錄"})
    # did the last meeting really get its live updates? (4 Nov got one all evening)
    past = [p for p in sorted((store.DOCS / "data").glob("20*_*.json"))
            if p.stem[:10] < today.isoformat()]
    if past:
        last_doc = store.read_json(past[-1])
        runs = len(last_doc.get("runs", []))
        checks.append({"name": "賽馬日更新", "ok": runs >= 20,
                       "detail": f"{last_doc['meeting']['date']} 更新咗 {runs} 次(正常應該 50 次以上)"})
    void = sum(1 for r in recent for x in r["runners"]
               if x.get("place") and not x.get("time"))
    checks.append({"name": "賽果解析", "ok": void == 0,
                   "detail": "完成時間齊全" if void == 0 else f"{void} 匹完成馬冇時間"})
    return {"checks": checks, "last_result": last, "meetings": len(meetings),
            "races": len(races)}


def load_config():
    return store.read_json(store.DATA / "model_config.json", {"l2": 0.05, "stages": 1})


def run():
    races, samples = _samples()
    F = list(model.FEATURES)
    current = load_config()
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
        store.write_json(store.DATA / "model_config.json", {"l2": best["l2"], "stages": best["stages"]})
        switched = True
    print("  auto-tune best", best, "switched" if switched else "kept")

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
        p, b = ls["picked"], ls["all_longshots"]
        better = p["top3"] > b["top3"] * 1.2
        notes.append(f"冷門馬規則({ls['rule']}):入位率 {p['top3']:.0%},一般冷門 {b['top3']:.0%}"
                     + (",仍然有效" if better else ",已經冇明顯優勢,要重新檢討")
                     + f";獨贏平注回報 {p['win_roi']:+.0%}。")
    if tr.get("races", 0) < 50:
        notes.append(f"實戰往績只有 {tr.get('races', 0)} 場,樣本太少,未能下結論。")

    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "health": data_health(races),
        "walk_forward": base,
        "blend_weights": weights,
        "calibration": cal,
        "factor_groups": contrib,
        "tuning": {"current": current, "tried": tried, "switched": switched,
                   "chosen": {"l2": best["l2"], "stages": best["stages"]} if switched else current},
        "live": {k: tr.get(k) for k in ("races", "wins", "places", "flat_roi")},
        "longshots": ls,
        "steam": steam_stats(),
        "notes": notes,
    }
    for c in report["health"]["checks"]:
        c["ok"] = bool(c["ok"])  # numpy bools are not JSON serialisable
    store.write_json(store.DOCS / "data" / "selfcheck.json", report)
    print("  wrote selfcheck.json")
    return report


if __name__ == "__main__":
    run()
