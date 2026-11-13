"""Race-day watch: is the live system doing right now what it should be doing?

    python -m hkjc.watch            # read the latest data on GitHub and report
    python -m hkjc.watch --act      # also start the update or the ticker again when it has stopped

It reads the files the cloud commits (through git, without touching the working copy) and, when
the gh CLI or GitHub's public API answers, the recent workflow runs. It changes no code and no
data. With --act the only things it may do are `gh workflow run update.yml` and
`gh workflow run ticker.yml`, the same two buttons a person would press.

Lines starting with NOTIFY: are what is worth interrupting someone for: a new problem, a problem
still there after half an hour, a recovery, one "all ready" before the first race and one summary
after the last. Everything else is the routine report. What has already been said is kept in
tools/watch-state.json (not committed).

Nothing here needs more than the standard library, so a fresh clone can run it as it is.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import store

HKT = timezone(timedelta(hours=8))
REPO = "kelvin1997121/hkjc-quant"
STATE = store.ROOT / "tools" / "watch-state.json"
LOG = store.ROOT / "tools" / "watch.log"

# minutes; a healthy meeting (11 Nov 2026: 92 updates, longest gap 6) stays well inside these
WINDOW_BEFORE, WINDOW_AFTER = 60, 45     # the ticker's 5-minute clock: 60 before the first post to 45 after the last
MAX_UPDATE_GAP = 12                      # between full updates during the meeting
MAX_ODDS_AGE = 12                        # last read of a race not yet off, during the meeting
MAX_ODDS_AGE_MORNING = 45                # the same before the window (updates come every 20 to 30 minutes)
MAX_PRE_AGE = 8                          # the "pre" snapshot must be read this close to the post time
RESULT_DUE = 30                          # a result should be on the page this long after the post time
PAPER_DUE = 40                           # and the paper account settled by then
REMIND_AFTER = 30                        # say an unsolved problem again after this long
RETRIGGER_AFTER = 6                      # don't start another update within this of the last one


def _t(text):
    return datetime.fromisoformat(text).astimezone(HKT)


def _post(day, hhmm):
    return datetime.fromisoformat(f"{day}T{hhmm}:00").replace(tzinfo=HKT)


def _mins(a, b):
    """Whole minutes from a to b."""
    return int((b - a).total_seconds() // 60)


# ------------------------------------------------------------------ the checks

def assess(data: dict, now: datetime, runs: dict | None = None) -> dict:
    """data: schedule, doc, latest, odds, paper, selfcheck, track, trackwork (as on GitHub; any may
    be None). runs: {"update": [...], "ticker": [...]} newest first, each {"status", "conclusion",
    "created" (datetime), "url"}; None when GitHub could not be asked.

    Returns {"phase", "headline", "problems", "actions", "ready", "summary"}. A problem is
    {"key", "text", "lasting"}: lasting = a fact about a race already run, which nothing can
    change now (said once); the others are faults that can still be put right."""
    sched = data.get("schedule") or {}
    day, venue, posts = sched.get("date"), sched.get("venue"), sched.get("posts") or {}
    today = now.date().isoformat()
    out = {"phase": "idle", "headline": "", "problems": [], "actions": [], "ready": None, "summary": None}
    problems = out["problems"]

    def add(key, text, lasting=False):
        problems.append({"key": key, "text": text, "lasting": lasting})

    if not day or not posts or day < today:
        out["headline"] = "頁面上冇未跑嘅賽馬日"
        _workflows(runs, add, out["actions"])
        return out
    label = f"{int(day[5:7])}/{int(day[8:10])} {'沙田' if venue == 'ST' else '跑馬地'}"
    order = sorted(posts, key=int)
    if day > today:
        out["headline"] = f"下一個賽馬日係 {label},今日唔係賽馬日"
        _odds_ready(data, sched, now, add, opened_only=True)
        _workflows(runs, add, out["actions"])
        return out

    first, last = _post(day, posts[order[0]]), _post(day, posts[order[-1]])
    start, end = first - timedelta(minutes=WINDOW_BEFORE), last + timedelta(minutes=WINDOW_AFTER)
    doc, odds = data.get("doc") or {}, data.get("odds") or {}
    races = {str(r["race_no"]): r for r in doc.get("races", [])}
    if (data.get("latest") or {}).get("meeting", {}).get("date") != day:
        add("latest", "網頁顯示緊嘅唔係今日嘅賽事")
    if len(races) < len(posts):
        add("card", f"分析檔得 {len(races)}/{len(posts)} 場")

    if now < start:
        out["phase"] = "before"
        out["headline"] = f"{label} 今日賽事,{posts[order[0]]} 開第 1 場,未入 5 分鐘更新時段"
        _odds_ready(data, sched, now, add)
        _morning(data, sched, add)
        _workflows(runs, add, out["actions"])
        return out

    off = [k for k in order if _post(day, posts[k]) <= now - timedelta(minutes=1)]
    resulted = [k for k in order if races.get(k, {}).get("result")]

    # every race already off: frozen before the off, from the last odds read before it
    pools = odds.get("pools") or {}
    for k in off:
        race, post = races.get(k), _post(day, posts[k])
        if not race:
            continue
        if race.get("post_hoc"):
            add(f"post_hoc:{k}", f"第{k}場開跑後先有分析,唔計入往績同模擬倉", lasting=True)
            continue
        pre = (pools.get(k) or {}).get("pre")
        if not pre:
            add(f"no_pre:{k}", f"第{k}場冇開跑前賠率(模擬倉值博同測試用唔到呢場)", lasting=True)
            continue
        age = _mins(_t(pre["t"]), post)
        if age > MAX_PRE_AGE:
            add(f"pre_old:{k}", f"第{k}場開跑前賠率係開跑前 {age} 分鐘讀", lasting=True)
        if any(pre["win"].get(str(x["no"])) != x.get("win_odds") for x in race.get("runners", [])):
            add(f"pre_differs:{k}", f"第{k}場首選用嘅賠率同開跑前賠率唔係同一份", lasting=True)
    for k in order:
        waited = _mins(_post(day, posts[k]), now)
        if k not in resulted and waited >= RESULT_DUE:
            add(f"result_late:{k}", f"第{k}場開跑 {waited} 分鐘都未有賽果")
    paper_rows = {str(r["race_no"]): r for m in (data.get("paper") or {}).get("meetings", [])
                  if m.get("date") == day for r in m.get("races", [])}
    for k in resulted:
        row = paper_rows.get(k)
        if _mins(_post(day, posts[k]), now) >= PAPER_DUE and \
                (not row or row.get("state") not in ("settled", "void", "post_hoc")):
            add(f"paper:{k}", f"第{k}場有賽果但模擬倉未結算")

    if now <= end:
        out["phase"] = "live"
        nxt = next((k for k in order if _post(day, posts[k]) > now), None)
        out["headline"] = f"{label} 進行中:已開跑 {len(off)} 場,有賽果 {len(resulted)} 場" + \
            (f",下一場第{nxt}場 {posts[nxt]}" if nxt else ",全部開跑咗")
        out["ready"] = f"{label} 賽前檢查正常,{posts[order[0]]} 開第 1 場。之後有問題先會通知"
        # the window has only just opened: the last update may still be from the slower morning clock
        settled = now >= start + timedelta(minutes=MAX_UPDATE_GAP)
        max_gap, max_age = (MAX_UPDATE_GAP, MAX_ODDS_AGE) if settled else (MAX_ODDS_AGE_MORNING, MAX_ODDS_AGE_MORNING)
        updates = [_t(t) for t in doc.get("runs", [])]
        gap = _mins(updates[-1], now) if updates else None
        if gap is None or gap > max_gap:
            add("stalled", "今日未有完整更新" if gap is None else f"已經 {gap} 分鐘冇完整更新(正常約 5 分鐘一次)")
            out["actions"].append("update")
        else:
            coming = [k for k in order if _post(day, posts[k]) > now]
            stale = [k for k in coming if _odds_age(odds, k, now) > max_age]
            four = [k for k in coming if k not in stale and _odds_age(odds, k, now, four_pools=True) > max_age]
            if stale:
                add("odds_stale", f"未開跑嘅 {len(stale)} 場賠率超過 {max_age} 分鐘冇更新(賠率頁可能讀唔到)")
            if four:
                add("quinella_stale", f"未開跑嘅 {len(four)} 場讀唔到連贏同位置Q 賠率,只得獨贏位置")
        _workflows(runs, add, out["actions"])
        return out

    out["phase"] = "after"
    out["headline"] = f"{label} 已完:{len(resulted)}/{len(order)} 場有賽果"
    _workflows(runs, add, out["actions"])
    out["summary"] = _summary(data, label, day, order, resulted, races, problems)
    return out


def _odds_age(odds, k, now, four_pools=False):
    """Minutes since race k's odds were last read (a large number when never)."""
    last = (((odds.get("pools") or {}).get(k) or {}).get("last") or {}).get("t")
    read = last if four_pools else ((odds.get("latest") or {}).get(k) or {}).get("t") or last
    return _mins(_t(read), now) if read else 10 ** 6


def _odds_ready(data, sched, now, add, opened_only=False):
    """Before the meeting: every race read, with quinella odds, and read recently."""
    odds, total = data.get("odds") or {}, len(sched.get("posts") or {})
    read = len(odds.get("history") or {})
    if opened_only and not read:
        return                                   # the day before, betting not open yet
    four = len(odds.get("pools") or {})
    if min(read, four) < 0.8 * total:
        add("odds_missing", f"賠率只讀到 {read}/{total} 場,{four} 場有連贏同位置Q")
        return
    if now.hour * 60 + now.minute >= 8 * 60 + 45:        # updates start at 08:00
        newest = min(_odds_age(odds, k, now) for k in sched["posts"])
        if newest > MAX_ODDS_AGE_MORNING:
            add("odds_stale", f"賠率已經 {newest} 分鐘冇更新")


def _morning(data, sched, add):
    bad = [c["name"] for c in ((data.get("selfcheck") or {}).get("health") or {}).get("checks", [])
           if c.get("ok") is False]
    if bad:
        add("health", "自我檢查有紅燈:" + "、".join(bad))
    work = data.get("trackwork")
    if work is not None and len(work.get("races") or {}) < len(sched.get("posts") or {}):
        add("trackwork", f"晨操紀錄只儲咗 {len(work.get('races') or {})}/{len(sched['posts'])} 場")


def _workflows(runs, add, actions):
    if not runs:
        return
    tick = runs.get("ticker") or []
    if tick and not any(r["status"] != "completed" for r in tick):
        add("ticker", "雲端時鐘停咗")
        actions.append("ticker")
    done = [r for r in runs.get("update") or [] if r["status"] == "completed"]
    if len(done) >= 2 and all(r["conclusion"] == "failure" for r in done[:2]):
        add("update_failed", f"最近兩次更新都失敗 {done[0].get('url', '')}".strip())


def _summary(data, label, day, order, resulted, races, problems):
    doc = data.get("doc") or {}
    text = f"{label}完成:{len(resulted)}/{len(order)} 場有賽果"
    posts = [_post(day, r["post_time"]) for r in doc.get("races", []) if r.get("post_time")]
    if doc.get("runs") and posts:
        start, end = min(posts) - timedelta(minutes=WINDOW_BEFORE), max(posts) + timedelta(minutes=WINDOW_AFTER)
        inside = sorted(t for t in map(_t, doc["runs"]) if start <= t <= end)
        edges = [start, *inside, end]
        text += f";更新 {len(inside)} 次,最長相隔 {max(_mins(a, b) for a, b in zip(edges, edges[1:]))} 分鐘"
    rows = [r for r in (data.get("track") or {}).get("rows", []) if r.get("date") == day]
    if rows:
        text += (f";首選 {sum(r['won'] for r in rows)} 勝 {sum(r['placed'] for r in rows)} 位"
                 f"(最後熱門 {sum(r.get('fav_won', 0) for r in rows):g} 勝)")
    today = [m for m in (data.get("paper") or {}).get("meetings", []) if m.get("date") == day]
    if today:
        net = sum(b["back"] - b["stake"] for m in today for r in m.get("races", []) for b in r.get("main", [])
                  if b.get("placed") and b.get("back") is not None)
        text += f";模擬倉今日 {'+' if net > 0 else '-' if net < 0 else ''}${abs(net):,.0f}"
    faults = sum(1 for p in problems if not p["lasting"])
    notes = sum(1 for p in problems if p["lasting"])
    text += ";運作正常" if not faults else f";仲有 {faults} 個問題未解決"
    return text + (f";{notes} 場有備註" if notes else "")


# ------------------------------------------------------------------ what to say, and when

def decide(report: dict, state: dict, now: datetime, day: str | None) -> tuple[list[str], dict]:
    """Which NOTIFY lines this run gives, and the state to keep for the next run."""
    if state.get("day") != day:
        state = {"day": day, "open": {}, "ready": False, "done": False}
    said, open_ = [], dict(state.get("open") or {})
    stamp = now.isoformat(timespec="seconds")
    current = {p["key"]: p for p in report["problems"]}
    faults = [k for k, p in current.items() if not p["lasting"]]
    fresh = [k for k in current if k not in open_]
    again = [k for k in faults if k in open_ and _mins(_t(open_[k]["said"]), now) >= REMIND_AFTER]
    if fresh or again:
        acted = {"update": "已重新觸發更新", "ticker": "已重新啟動時鐘"}
        did = [acted[a] for a in report.get("done_actions", [])]
        head = "賽馬系統" + ("有問題" if any(k in faults for k in fresh) else "問題未解決" if again else "備註")
        said.append(f"{head}:" + ";".join(current[k]["text"] for k in fresh + again) + ("。" + "、".join(did) if did else ""))
        for k in fresh + again:
            open_[k] = {"since": open_.get(k, {}).get("since", stamp), "said": stamp, "lasting": current[k]["lasting"]}
    solved = [k for k, v in open_.items() if k not in current and not v.get("lasting")]
    for k in [k for k in open_ if k not in current]:
        del open_[k]
    if solved and not faults:
        said.append("賽馬系統回復正常")
    state["open"] = open_
    if report["phase"] == "live" and not state.get("ready") and not faults and report.get("ready"):
        said.append(report["ready"])
        state["ready"] = True
    if report["phase"] == "after" and not state.get("done") and report.get("summary"):
        said.append(report["summary"])
        state["done"] = True
    return [s if len(s) <= 195 else s[:192] + "…" for s in said], state


# ------------------------------------------------------------------ reading GitHub

def _git(*args):
    return subprocess.run(["git", "-C", str(store.ROOT), *args], capture_output=True)


def _show(ref, path):
    done = _git("show", f"{ref}:{path}")
    return json.loads(done.stdout.decode("utf-8")) if done.returncode == 0 and done.stdout else None


def load(ref="origin/main") -> dict:
    sched = _show(ref, "docs/data/schedule.json") or {}
    stem = f"{sched.get('date')}_{sched.get('venue')}"
    return {"schedule": sched, "doc": _show(ref, f"docs/data/{stem}.json"),
            "latest": _show(ref, "docs/data/latest.json"), "odds": _show(ref, f"data/odds/{stem}.json"),
            "paper": _show(ref, "docs/data/paper.json"), "selfcheck": _show(ref, "docs/data/selfcheck.json"),
            "track": _show(ref, "docs/data/track_record.json"), "trackwork": _show(ref, f"data/trackwork/{stem}.json")}


def workflow_runs() -> dict | None:
    """Recent update and ticker runs, from the gh CLI or GitHub's public API; None if neither answers."""
    out = {}
    for name in ("update", "ticker"):
        rows = None
        if shutil.which("gh"):
            done = subprocess.run(["gh", "run", "list", "--repo", REPO, "--workflow", f"{name}.yml", "--limit", "8",
                                   "--json", "status,conclusion,createdAt,url"], capture_output=True, timeout=60)
            if done.returncode == 0:
                rows = [{"status": r["status"], "conclusion": r["conclusion"], "created": _t(r["createdAt"]),
                         "url": r["url"]} for r in json.loads(done.stdout.decode("utf-8"))]
        if rows is None:
            try:
                url = f"https://api.github.com/repos/{REPO}/actions/workflows/{name}.yml/runs?per_page=8"
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "hkjc-watch"}),
                                            timeout=30) as r:
                    rows = [{"status": x["status"], "conclusion": x["conclusion"], "created": _t(x["created_at"]),
                             "url": x["html_url"]} for x in json.load(r)["workflow_runs"]]
            except Exception:
                return None
        out[name] = rows
    return out


def act(report: dict, runs: dict | None, now: datetime) -> list[str]:
    """Start the update or the ticker again. Returns what was started."""
    did = []
    if not shutil.which("gh") or not runs:
        return did
    updates = runs.get("update") or []
    busy = any(r["status"] != "completed" for r in updates)
    recent = bool(updates) and _mins(updates[0]["created"], now) < RETRIGGER_AFTER
    for what in dict.fromkeys(report["actions"]):
        if what not in ("update", "ticker") or (what == "update" and (busy or recent)):
            continue
        done = subprocess.run(["gh", "workflow", "run", f"{what}.yml", "--repo", REPO], capture_output=True, timeout=60)
        if done.returncode == 0:
            did.append(what)
    return did


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--act", action="store_true", help="start the update or ticker workflow again when it has stopped")
    ap.add_argument("--now", help="pretend it is this time (ISO, Hong Kong time if no offset): replays history, never acts")
    ap.add_argument("--ref", default="origin/main")
    ap.add_argument("--state", default=str(STATE))
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    replay = bool(args.now)
    if replay:
        now = datetime.fromisoformat(args.now)
        now = now.replace(tzinfo=HKT) if now.tzinfo is None else now.astimezone(HKT)
        ref = _git("rev-list", "-1", f"--before={now.isoformat()}", args.ref).stdout.decode().strip() or args.ref
        runs = None
    else:
        now, ref = datetime.now(HKT), args.ref
        _git("fetch", "-q", "origin", "main")
        runs = workflow_runs()
    data = load(ref)
    report = assess(data, now, runs)
    report["done_actions"] = act(report, runs, now) if args.act and not replay else []

    state_path = Path(args.state)
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    say, state = decide(report, state, now, (data.get("schedule") or {}).get("date"))
    if not replay or args.state != str(STATE):
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")

    faults = [p for p in report["problems"] if not p["lasting"]]
    lines = [f"{now:%Y-%m-%d %H:%M} {'PROBLEM' if faults else 'OK'} [{report['phase']}] {report['headline']}"]
    if runs is None and not replay:
        lines.append("  (GitHub 冇回應,今次睇唔到 workflow 狀態)")
    lines += [f"  - {'(備註) ' if p['lasting'] else ''}{p['text']}" for p in report["problems"]]
    lines += [f"  已做:重新觸發 {a}" for a in report["done_actions"]]
    lines += [f"NOTIFY: {s}" for s in say]
    text = "\n".join(lines)
    print(text)
    if not replay:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
