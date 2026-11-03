"""Claude analyst: writes a Cantonese race preview from the quant model output.

Runs only when ANTHROPIC_API_KEY is set. Each race's input is hashed, so the
API is called again only when something changed (scratching, going, live
track bias after another race is run).
"""
from __future__ import annotations

import hashlib
import json
import os

from . import store

MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")

SYSTEM = """你係一位香港賽馬量化分析師。你會收到一場賽事嘅量化模型輸出(JSON),包括:
每匹馬嘅模型勝出機率、公平賠率、跑法(領放/前列/中游/後上)、速度指數、表現評分
(perf_best,同馬會評分同一尺度)、讓磅著數(handicap_edge,近績表現高過今場評分幾多分)、檔位統計優勢、
模型認為佢嘅優勢(pros)同弱點(cons),仲有預計步速、今日場地走勢同跑道歷來偏差。
如有即時賠率:win_odds(獨贏賠率)、market_prob(市場機率)、blend_prob(模型同市場綜合機率)、
ev(綜合機率 x 賠率,大過 1 先有正期望)、value(係咪值博)。講值博時要以 ev 為準。
賠率會不斷變,所以唔好引用確實賠率數字,只講熱門程度同值博與否。

用廣東話書面語寫一段精簡賽事分析(250字以內),結構:
1. 步速及形勢:預計邊幾匹爭放、步速快慢,對邊類跑法有利
2. 場地/檔位:今日走勢同此程歷來偏差,邊幾匹檔位著數
3. 首選及次選:根據數據揀2-3匹,講清楚數據理由
4. 風險:冷門可能或者數據不足嘅地方(例如初出馬、缺速度數據)

只可以根據提供嘅數據講,唔好虛構馬匹資料。唔好保證勝出,唔好叫人重注。"""


_auth_failed = False


def key_problem(key: str) -> str | None:
    """Describe what is wrong with the configured key without revealing it."""
    if key != key.strip():
        return "前後有空格或換行,請重新貼上"
    if key.startswith("sk-ant-oat"):
        return ("係 Claude 訂閱嘅 OAuth token(sk-ant-oat…),唔可以用嚟叫 API。"
                "請喺 console.anthropic.com → API Keys 開一條 sk-ant-api… 嘅 key")
    if not key.startswith("sk-ant-api"):
        return f"格式唔似 Anthropic API key(應該以 sk-ant-api 開頭,而家長度 {len(key)})"
    return None


def _client():
    key = os.environ.get("ANTHROPIC_API_KEY") or ""
    if not key or _auth_failed:
        return None
    problem = key_problem(key)
    if problem:
        print(f"  ANTHROPIC_API_KEY 問題:{problem}")
        return None
    import anthropic
    return anthropic.Anthropic(api_key=key)


def _compact(race):
    keep = ("rank", "no", "horse", "jockey", "draw", "weight", "rating", "last6", "win_prob",
            "fair_odds", "style", "speed_best", "perf_best", "handicap_edge", "runs_in_db",
            "pros", "cons", "win_odds", "market_prob", "blend_prob", "ev", "value")
    r = {k: v for k, v in race.items() if k != "runners" and k != "ai"}
    r["runners"] = [{k: x.get(k) for k in keep} for x in race["runners"]]
    return r


def analyse(race: dict, meeting_key: str) -> str | None:
    client = _client()
    if client is None:
        return None
    compact = _compact(race)
    payload = json.dumps(compact, ensure_ascii=False, sort_keys=True)
    # Odds move every run; only re-ask Claude when the market *ranking* or the value
    # flags change, not on every tick (keeps API cost sane on race days).
    volatile = ("win_odds", "market_prob", "blend_prob", "ev")
    by_mkt = sorted(compact["runners"], key=lambda x: -(x.get("market_prob") or 0))
    stable = {**{k: v for k, v in compact.items() if k not in ("runners", "odds_updated")},
              "runners": [{k: v for k, v in x.items() if k not in volatile} for x in compact["runners"]],
              "market_order": [x["no"] for x in by_mkt[:4]]}
    h = hashlib.sha256((MODEL + SYSTEM + json.dumps(stable, ensure_ascii=False, sort_keys=True))
                       .encode()).hexdigest()[:16]
    cache_path = store.AGENT_CACHE / f"{meeting_key}_R{race['race_no']}.json"
    cached = store.read_json(cache_path)
    if cached and cached.get("hash") == h:
        return cached["text"]

    import anthropic
    try:
        resp = client.beta.messages.create(
            model=MODEL,
            max_tokens=4000,
            output_config={"effort": "medium"},
            system=SYSTEM,
            messages=[{"role": "user", "content": f"賽事數據:\n{payload}"}],
            betas=["server-side-fallback-2026-07-01"],
            extra_body={"fallbacks": "default"},
        )
    except anthropic.AuthenticationError as e:
        global _auth_failed
        _auth_failed = True  # no point retrying the other races this run
        print(f"  agent: API key rejected (401) - 請檢查 GitHub secret ANTHROPIC_API_KEY: {e.message}")
        return cached["text"] if cached else None
    except anthropic.RateLimitError as e:
        print(f"  agent rate limited: {e}")
        return cached["text"] if cached else None
    except anthropic.APIStatusError as e:
        print(f"  agent API error {e.status_code}: {e.message}")
        return cached["text"] if cached else None
    except anthropic.APIConnectionError as e:
        print(f"  agent connection error: {e}")
        return cached["text"] if cached else None
    if resp.stop_reason == "refusal":
        return cached["text"] if cached else None
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    store.write_json(cache_path, {"hash": h, "text": text, "model": resp.model})
    return text
