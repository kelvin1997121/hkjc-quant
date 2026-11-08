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
    # final_rank = model + market when odds are in; older saved races only have the model's rank
    by_final = sorted(runners, key=lambda x: x.get("final_rank", x["rank"]))
    by_model = sorted(runners, key=lambda x: x["rank"])
    top5 = by_final[:5]
    lines = []

    # 1. pace and shape: verdict comes from history (analyze.pace_text), not a rule of thumb
    leaders = [x for x in runners if x["style"] == "領放"]
    s = "【步速】"
    if leaders:
        s += f"{_names(leaders)}有機會爭放。"
    s += a.get("pace_note", "") + "。"
    favours = a.get("pace_favours")
    if favours == "前置":
        fit = [x for x in top5 if x["style"] in ("領放", "前列")]
    elif favours == "後上":
        fit = [x for x in top5 if x["style"] in ("後上", "中游")]
    else:
        fit = []
    if fit:
        s += f"跑法配合嘅熱門選擇:{_names(fit, 2)}。"
    lines.append(s)

    # 2. track and draw
    s = f"【場地】{a.get('today_bias', '')};{a.get('course_bias', '')}。"
    if a.get("today_sect"):
        s += a["today_sect"] + "。"
        late = (a.get("today_sect_data") or {}).get("late", 0)
        if late > 0.07:
            fast_finish = [x for x in top5 if "末段速度(分段時間)" in x["pros"]]
            if fast_finish:
                s += f"末段快嘅{_names(fast_finish, 2)}受惠。"
    good_draw = sorted([x for x in runners if (x.get("draw_edge") or 0) > 0.08],
                       key=lambda x: -x["draw_edge"])
    bad_draw = [x for x in top5 if (x.get("draw_edge") or 0) < -0.08]
    if good_draw:
        s += f"檔位統計著數:{_names(good_draw, 2)}。"
    if bad_draw:
        s += f"{_names(bad_draw, 2)}檔位統計較差。"
    lines.append(s)

    # 3. picks
    picks = []
    for label, x in zip(("首選", "次選", "三選"), by_final[:3]):
        why = "、".join(x["pros"][:3]) or "綜合數據"
        # the ranking is the blend; the model's own view is left to 【風險】 and the page's note
        p = (f"綜合 {_pct(x['final_prob'])}" + (f",獨贏 {x['win_odds']:g} 倍" if x.get("win_odds") else "")) \
            if x.get("final_prob") else f"模型 {_pct(x['win_prob'])},公平賠率 {x['fair_odds']}"
        t = f"{label}{_name(x)}({p}):{why}佔優"
        rel = x.get("handicap_edge") or 0  # already relative to the field median
        if rel >= 3:
            t += f",讓磅著數比同場中位數多約 {rel:.0f} 分"
        picks.append(t)
    lines.append("【選擇】" + ";".join(picks) + "。")
    if a.get("confidence"):
        lines.append(f"【把握】{a['confidence']}:首選獨贏 {by_final[0]['win_odds']:g} 倍。"
                     "把握係睇賠率高低,唔代表值博。")

    # 3b. longshots: odds >= 10 but ranked top 5 by the model (place more often than
    # typical longshots, but win bets on longshots have lost money historically)
    shots = a.get("longshots") or []
    if shots:
        parts = []
        for x in shots:
            why = "、".join(x["pros"][:2]) or "綜合數據"
            parts.append(f"{_name(x)}(獨贏 {x['win_odds']:g},模型排第 {x['rank']}):{why}")
        lines.append("【冷門】" + ";".join(parts) +
                     "。呢類馬入位率略高過同賠率嘅其他馬,但過往買獨贏、位置都係蝕錢,只供參考,唔係值博訊號。")
    elif any(x.get("win_odds") for x in runners):
        lines.append("【冷門】今場冇賠率 10 倍以上而模型排頭五嘅馬。")

    # 3c. late money (information only: not yet backtested, so not in the probabilities)
    steam = [x for x in runners if x.get("steam")]
    if steam:
        lines.append("【落飛】" + ";".join(f"{_name(x)}(早段 {x['open_odds']:g} → 而家 {x['win_odds']:g})"
                                         for x in steam[:3]) + "。")

    # 4. market view and risks
    risks = []
    with_odds = [x for x in runners if x.get("market_prob")]
    if with_odds:
        if by_final[0]["no"] != by_model[0]["no"]:
            risks.append(f"模型自己最睇好{_name(by_model[0])},同市場睇法唔同")
        value = [x for x in runners if x.get("value")]
        if value:
            risks.append(f"按綜合機率計,{_names(value, 2)}有值博")
        else:
            risks.append("按綜合機率計暫時冇明顯值博")
    thin = [x for x in by_final[:3] if x.get("runs_in_db", 0) == 0]
    if thin:
        risks.append(f"{_names(thin)}數據不足(初出或冇往績),評估把握較低")
    lead_p = by_final[0].get("final_prob") or by_final[0]["win_prob"]
    if lead_p < 0.2:
        risks.append(f"首選機率只有 {_pct(lead_p)},形勢開放")
    cons = [x for x in by_final[:2] if x.get("cons")]
    for x in cons:
        risks.append(f"{_name(x)}要留意{'、'.join(x['cons'])}")
    risks.append("賠率會變,以開跑前為準")
    lines.append("【風險】" + ";".join(risks) + "。")
    return "\n".join(lines)
