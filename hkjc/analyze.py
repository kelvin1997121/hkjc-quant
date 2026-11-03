"""Turn model output for a race card into readable, explainable analysis."""
from __future__ import annotations

import numpy as np

from .model import (FEATURES, blend, contributions, market_probs, pace_bucket, predict,
                    prior_vector, style_label)

FEATURE_ZH = {
    "speed_best": "速度指數(近績最佳)", "speed_last": "上仗速度指數", "speed_missing": "缺速度數據",
    "form": "近況", "dist_form": "路程適應", "early": "前置跑法",
    "early_x_pace": "步速形勢", "early_x_today": "今日場地走勢", "early_x_course": "跑道跑法偏差",
    "draw_bias": "檔位統計", "draw_x_today": "今日檔位走勢", "jockey": "騎師", "trainer": "練馬師",
    "weight_rel": "負磅", "layoff": "休賽", "debut": "初出",
    "mkt_rating": "過往市場評價", "vs_mkt": "跑贏市場預期", "closing": "後段追勁",
    "class_drop": "降班", "bw_change": "體重變化", "perf_best": "表現評分(近績最佳)",
    "perf_last": "上仗表現評分", "handicap_edge": "讓磅著數",
    "late_speed": "末段速度(分段時間)", "early_speed": "前段速度(分段時間)",
    "jockey_form": "騎師近況", "trainer_form": "練馬師近況", "combo": "騎練組合",
    "venue_form": "本場地往績",
}


def pace_text(state, ctx, venue):
    """Pace read backed by history, not a rule of thumb.

    In HK, leaders win most often at every projected pace; a fast pace only
    narrows their edge. So the verdict comes from past win rates per running
    style for this venue and projected pace. Returns (bucket, note, favours).
    """
    bucket = pace_bucket(ctx["pace"])
    head = {"快": "預計步速偏快", "慢": "預計步速偏慢", "中": "預計步速中等"}[bucket] \
        + f"({ctx['n_lead']} 匹慣常跑喺最前嘅位置)"
    rates, v = state.pace_history(venue, bucket)
    if len(rates) < 3:
        return bucket, head + "。", "中性"
    where = {"HV": "跑馬地", "ST": "沙田"}.get(v, "過往")
    stats = "、".join(f"{s} {r:.1%}" for s, (r, _) in rates.items())
    front = max(rates.get("領放", (0, 0))[0], rates.get("前列", (0, 0))[0])
    back = rates.get("後上", (0, 0))[0]
    if front > back * 1.2:
        favours, verdict = "前置", "前置馬仍然較有利"
    elif back > front * 1.2:
        favours, verdict = "後上", "後上馬較有利"
    else:
        favours, verdict = "中性", "跑法影響唔大"
    if bucket != "中":
        normal, _ = state.pace_history(venue, "中")
        if "領放" in rates and "領放" in normal:
            diff = rates["領放"][0] - normal["領放"][0]
            if diff < -0.005:
                verdict += ",不過領放馬優勢比平時細"
            elif diff > 0.005:
                verdict += ",而且領放馬優勢比平時大"
    return bucket, f"{head}。{where}同類步速歷來每匹馬勝出率:{stats};{verdict}", favours


def today_label(ctx):
    if ctx["today_n"] == 0:
        return "今日未有已完成賽事,未能計算即日場地走勢"
    parts = []
    s, d = ctx["today_style"], ctx["today_draw"]
    # relative to a normal day (leaders normally have an edge in HK)
    parts.append("前置馬比平時更著數" if s > 0.04 else "後上馬比平時更著數" if s < -0.04
                 else "跑法同平時差唔多")
    parts.append("內檔有利" if d > 0.04 else "外檔有利" if d < -0.04 else "檔位無明顯偏差")
    return f"根據今日已跑 {ctx['today_n']} 場:" + ",".join(parts)


VALUE_EV = 1.15  # flag a bet only when blended prob x odds clears takeout comfortably


def today_sect_label(ctx):
    """Plain-language read of today's sectional times (empty before any data)."""
    if not ctx.get("today_sect_n"):
        return ""
    v, lt = ctx["today_variant"], ctx["today_late"]
    if v < -0.2:
        speed = f"跑道偏快(比標準快約 {abs(v):.1f} 秒/公里)"
    elif v > 0.2:
        speed = f"跑道偏慢(比標準慢約 {v:.1f} 秒/公里)"
    else:
        speed = "跑道速度正常"
    if lt > 0.07:
        late = "末段追勁作用大,末段快嘅馬有著數"
    elif lt < -0.07:
        late = "末段追勁作用細,前置馬較易守得住"
    else:
        late = "末段追勁影響正常"
    return f"今日分段時間({ctx['today_sect_n']} 場):{speed};{late}"


def analyse_race(state, params, race, odds=None):
    """odds: {"win": {no: odds}, "place": {...}, "updated": str, "scratched": [no]} or None."""
    win = {int(k): v for k, v in (odds or {}).get("win", {}).items()}
    place = {int(k): v for k, v in (odds or {}).get("place", {}).items()}
    scratched = {int(n) for n in (odds or {}).get("scratched", [])}
    runners = [x for x in race["runners"]
               if x.get("draw") and not x.get("scratched") and x["no"] not in scratched]
    if not runners:
        return None
    X, extra, ctx = state.race_features(race, runners)
    if params:
        p = predict(params, X)
        contrib = contributions(params, X)
    else:  # no trained weights yet -> domain-knowledge prior weights
        z = (X - X.mean(0)) / (X.std(0) + 1e-6)
        contrib = z * prior_vector()
        s = contrib.sum(1)
        p = np.exp(s - s.max()) / np.exp(s - s.max()).sum()
    # live market view
    q = market_probs([win.get(x["no"], 0) for x in runners]) if win else None
    bw = (params or {}).get("blend")
    pb = blend(p, q, bw["a"], bw["b"]) if q is not None and bw else None
    order = np.argsort(-p)
    out = []
    for rank, i in enumerate(order, 1):
        x, e = runners[i], extra[i]
        prof = e["profile"]
        c = contrib[i]
        # indicator features only make sense as a label when the flag is set
        flags = {FEATURES.index("speed_missing"), FEATURES.index("debut")}
        def shown(j):
            return j not in flags or X[i][j] == 1
        pos = [FEATURE_ZH[FEATURES[j]] for j in np.argsort(-c) if c[j] > 0.05 and shown(j)][:3]
        neg = [FEATURE_ZH[FEATURES[j]] for j in np.argsort(c) if c[j] < -0.05 and shown(j)][:2]
        out.append({
            "rank": rank, "no": x["no"], "horse": x["horse"], "horse_code": x.get("horse_code"),
            "jockey": x.get("jockey"), "trainer": x.get("trainer"), "draw": x.get("draw"),
            "weight": x.get("weight"), "rating": x.get("rating"), "last6": x.get("last6"),
            "gear": x.get("gear"),
            "win_prob": round(float(p[i]), 4),
            "fair_odds": round(1 / max(float(p[i]), 1e-4), 1),
            "style": style_label(prof["early"]),
            "speed_best": None if prof["speed_best"] is None else round(prof["speed_best"], 1),
            "perf_best": None if prof["perf_best"] is None else round(prof["perf_best"], 1),
            "handicap_edge": round(float(X[i][FEATURES.index("handicap_edge")]), 1),
            "runs_in_db": prof["n_runs"],
            "draw_edge": round(float(X[i][FEATURES.index("draw_bias")]), 3),
            "pros": pos, "cons": neg,
            "win_odds": win.get(x["no"]), "place_odds": place.get(x["no"]),
            "market_prob": None if q is None else round(float(q[i]), 4),
            "blend_prob": None if pb is None else round(float(pb[i]), 4),
            "ev": None if pb is None or not win.get(x["no"])
                  else round(float(pb[i]) * win[x["no"]], 3),
        })
    for r in out:
        r["value"] = bool(r["ev"] and r["ev"] >= VALUE_EV and bw and bw["a"] > 0)
    pace_lv, pace_txt, pace_favours = pace_text(state, ctx, race["venue"])
    return {
        "race_no": race["race_no"], "race_name": race.get("race_name"),
        "post_time": race.get("post_time"), "class": race.get("class"),
        "distance": race.get("distance"), "surface": race.get("surface"),
        "course": race.get("course"), "going": race.get("going"),
        "pace": pace_lv, "pace_note": pace_txt, "pace_favours": pace_favours,
        "n_leaders": ctx["n_lead"],
        "today_bias": today_label(ctx),
        "today_sect": today_sect_label(ctx),
        "today_sect_data": {"variant": round(ctx["today_variant"], 2),
                            "late": round(ctx["today_late"], 3), "races": ctx["today_sect_n"]},
        "course_bias": ("此跑道路程歷來前置馬有利" if ctx["course_edge"] > 0.05 else
                        "此跑道路程歷來後上馬有利" if ctx["course_edge"] < -0.05 else
                        "此跑道路程跑法偏差不明顯"),
        "odds_updated": (odds or {}).get("updated"),
        "runners": out,
    }
