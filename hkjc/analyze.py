"""Turn model output for a race card into readable, explainable analysis."""
from __future__ import annotations

import numpy as np

from .model import FEATURES, contributions, predict, prior_vector

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
}


def style_label(early):
    if early is None:
        return "未知"
    if early <= 0.2:
        return "領放"
    if early <= 0.4:
        return "前列"
    if early <= 0.65:
        return "中游"
    return "後上"


def pace_label(pace):
    if pace > 0.5:
        return "快", "多匹前領馬爭位,預計步速偏快,後上馬有利"
    if pace < -0.4:
        return "慢", "前領馬少,預計步速偏慢,前置馬有利"
    return "中", "預計步速中等"


def today_label(ctx):
    if ctx["today_n"] == 0:
        return "今日未有已完成賽事,未能計算即日場地走勢"
    parts = []
    s, d = ctx["today_style"], ctx["today_draw"]
    parts.append("前置馬有利" if s > 0.04 else "後上馬有利" if s < -0.04 else "跑法無明顯偏差")
    parts.append("內檔有利" if d > 0.04 else "外檔有利" if d < -0.04 else "檔位無明顯偏差")
    return f"根據今日已跑 {ctx['today_n']} 場:" + ",".join(parts)


def analyse_race(state, params, race):
    runners = [x for x in race["runners"] if x.get("draw") and not x.get("scratched")]
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
        })
    pace_lv, pace_txt = pace_label(ctx["pace"])
    return {
        "race_no": race["race_no"], "race_name": race.get("race_name"),
        "post_time": race.get("post_time"), "class": race.get("class"),
        "distance": race.get("distance"), "surface": race.get("surface"),
        "course": race.get("course"), "going": race.get("going"),
        "pace": pace_lv, "pace_note": pace_txt, "n_leaders": ctx["n_lead"],
        "today_bias": today_label(ctx),
        "course_bias": ("此跑道路程歷來前置馬有利" if ctx["course_edge"] > 0.05 else
                        "此跑道路程歷來後上馬有利" if ctx["course_edge"] < -0.05 else
                        "此跑道路程跑法偏差不明顯"),
        "runners": out,
    }
