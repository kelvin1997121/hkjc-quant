"""Free, rule-based Cantonese race preview built from the quant analysis.

Used whenever the Claude analyst is unavailable (no key, no credit, API
error), so every race always has a written summary.
"""
from __future__ import annotations


def _name(x):
    return f"{x['no']}號{x['horse']}"


def _names(xs, n=3):
    return "、".join(_name(x) for x in xs[:n])


def _pct(p):
    return f"{p * 100:.0f}%"


def summarise(a: dict) -> str:
    runners = a["runners"]
    if not runners:
        return ""
    by_model = sorted(runners, key=lambda x: x["rank"])
    top5 = by_model[:5]
    lines = []

    # 1. pace and shape
    leaders = [x for x in runners if x["style"] == "領放"]
    pace = a.get("pace")
    s = f"【步速】預計步速{pace}"
    if leaders:
        s += f",{_names(leaders)}有機會爭放"
    s += "。"
    if pace == "快":
        fit = [x for x in top5 if x["style"] in ("後上", "中游")]
        s += f"前面有得拉,後上馬著數{(',留意' + _names(fit, 2)) if fit else ''}。"
    elif pace == "慢":
        fit = [x for x in top5 if x["style"] in ("領放", "前列")]
        s += f"前領馬少,前置馬著數{(',留意' + _names(fit, 2)) if fit else ''}。"
    else:
        s += "步速中等,跑法影響不大。"
    lines.append(s)

    # 2. track and draw
    s = f"【場地】{a.get('today_bias', '')};{a.get('course_bias', '')}。"
    good_draw = sorted([x for x in runners if (x.get("draw_edge") or 0) > 0.08],
                       key=lambda x: -x["draw_edge"])
    bad_draw = [x for x in top5 if (x.get("draw_edge") or 0) < -0.08]
    if good_draw:
        s += f"檔位統計著數:{_names(good_draw, 2)}。"
    if bad_draw:
        s += f"{_names(bad_draw, 2)}檔位統計較差。"
    lines.append(s)

    # 3. picks
    # "best of recent runs" is biased upward, so compare each horse's handicap
    # edge with the field's typical edge rather than quoting the raw number
    edges = sorted(x["handicap_edge"] for x in runners if x.get("perf_best") is not None)
    mid = edges[len(edges) // 2] if edges else 0
    picks = []
    for label, x in zip(("首選", "次選", "三選"), by_model[:3]):
        why = "、".join(x["pros"][:3]) or "綜合數據"
        t = f"{label}{_name(x)}(模型 {_pct(x['win_prob'])},公平賠率 {x['fair_odds']}):{why}佔優"
        rel = (x.get("handicap_edge") or 0) - mid
        if x.get("perf_best") is not None and rel >= 3:
            t += f",讓磅著數比同場中位數多約 {rel:.0f} 分"
        picks.append(t)
    lines.append("【選擇】" + ";".join(picks) + "。")

    # 4. market view and risks
    risks = []
    with_odds = [x for x in runners if x.get("market_prob")]
    if with_odds:
        fav = max(with_odds, key=lambda x: x["market_prob"])
        if fav["no"] != by_model[0]["no"]:
            risks.append(f"市場大熱門係{_name(fav)},同模型首選唔同")
        value = [x for x in runners if x.get("value")]
        if value:
            risks.append(f"按綜合機率計,{_names(value, 2)}有值博")
        else:
            risks.append("按綜合機率計暫時冇明顯值博")
    thin = [x for x in by_model[:3] if x.get("runs_in_db", 0) == 0]
    if thin:
        risks.append(f"{_names(thin)}數據不足(初出或冇往績),評估把握較低")
    if by_model[0]["win_prob"] < 0.2:
        risks.append(f"首選機率只有 {_pct(by_model[0]['win_prob'])},形勢開放")
    cons = [x for x in by_model[:2] if x.get("cons")]
    for x in cons:
        risks.append(f"{_name(x)}要留意{'、'.join(x['cons'])}")
    risks.append("賠率會變,以開跑前為準")
    lines.append("【風險】" + ";".join(risks) + "。")
    return "\n".join(lines)
