"""Race-day watch (hkjc/watch.py): what counts as a problem, what is worth a notification,
and what it may restart. A made-up Sha Tin meeting on 14 Nov 2026 with three races."""
import subprocess
import sys
from datetime import timedelta

import pytest

from helpers import analysis_doc, analysis_race, hkt, make_card, odds_read, utc_iso
from hkjc import watch

DAY, VENUE = "2026-11-14", "ST"
POSTS = {1: "12:30", 2: "13:00", 3: "13:55"}
CARD = make_card(DAY, VENUE, POSTS)
RACES = {r["race_no"]: r for r in CARD["races"]}


def at(clock, day=DAY):
    return hkt(day, clock)


def meeting(now, updated=None, read=None, results=(), pre=None, post_hoc=(), settled=None, runs_every=5):
    """The files as a healthy cloud would have them at `now` (a clock time on race day).

    updated / read: minutes ago of the last full update and of the last odds read (default 3).
    pre: {race: clock time its pre snapshot was read} for races already off (default 2 minutes
    before each post). results: races with a result. settled: races settled in the paper
    account (default: the same as results)."""
    t = at(now)
    updated = 3 if updated is None else updated
    read = 3 if read is None else read
    last_update = t - timedelta(minutes=updated)
    runs, x = [], last_update
    while x >= at("08:00"):
        runs.append(x.astimezone(watch.timezone.utc).isoformat(timespec="seconds"))
        x -= timedelta(minutes=runs_every)
    read_at = (t - timedelta(minutes=read)).astimezone(watch.timezone.utc).isoformat(timespec="seconds")
    odds = {"latest": {}, "history": {}, "pools": {}}
    analyses = []
    for n, race in RACES.items():
        post = at(POSTS[n] + ":00")
        snap = odds_read(race, 1, read_at)
        odds["latest"][str(n)] = {"win": {str(k): v for k, v in snap["win"].items()}, "t": read_at}
        odds["history"][str(n)] = [{"t": read_at, "win": snap["win"]}]
        book = {"first": dict(snap), "last": dict(snap)}
        if post <= t:
            clock = (pre or {}).get(n, (post - timedelta(minutes=2)).strftime("%H:%M:%S"))
            if clock:
                p = odds_read(race, 1, utc_iso(DAY, clock))
                p["win"] = {str(k): v for k, v in p["win"].items()}
                book["pre"] = p
        odds["pools"][str(n)] = book
        a = analysis_race(race, list(range(1, 9)), win_odds=snap["win"], post_hoc=n in post_hoc)
        if n in results:
            a["result"] = {"placings": [1, 2, 3], "win_odds": 1.9}
        analyses.append(a)
    doc = analysis_doc(DAY, VENUE, analyses)
    doc["runs"] = sorted(runs)
    settled = results if settled is None else settled
    paper = {"meetings": [{"date": DAY, "races": [
        {"race_no": n, "state": "settled" if n in settled else "placed",
         "main": [{"placed": True, "stake": 10.0, "back": 88.0 if n in settled else None}]} for n in RACES]}]}
    return {"schedule": {"date": DAY, "venue": VENUE, "posts": {str(n): p for n, p in POSTS.items()}},
            "doc": doc, "latest": {"meeting": {"date": DAY}}, "odds": odds, "paper": paper,
            "selfcheck": {"health": {"checks": [{"name": "賽果更新", "ok": True}]}},
            "track": {"rows": [{"date": DAY, "won": n == 1, "placed": True, "fav_won": 1.0} for n in results]},
            "trackwork": {"races": {str(n): {} for n in RACES}}}


def check(now, runs=None, **kw):
    return watch.assess(meeting(now, **kw), at(now), runs)


def keys(report):
    return [p["key"] for p in report["problems"]]


OK_RUNS = {"ticker": [{"status": "in_progress", "conclusion": None, "created": at("08:00:00"), "url": "t"}],
           "update": [{"status": "completed", "conclusion": "success", "created": at("12:00:00"), "url": "u"}]}


# ------------------------------------------------------------------ a healthy day

@pytest.mark.parametrize("now, phase", [("09:00:00", "before"), ("11:29:00", "before"), ("11:30:00", "live"),
                                        ("12:45:00", "live"), ("14:40:00", "live"), ("14:41:00", "after")])
def test_healthy_meeting_has_no_problems(now, phase):
    done = [n for n in RACES if at(POSTS[n] + ":00") + timedelta(minutes=15) <= at(now)]
    report = check(now, runs=OK_RUNS, results=done)
    assert report["phase"] == phase and report["problems"] == [] and report["actions"] == []


def test_headline_says_where_the_meeting_is():
    report = check("13:10:00", results=[1])
    assert "已開跑 2 場" in report["headline"] and "有賽果 1 場" in report["headline"]
    assert "下一場第3場 13:55" in report["headline"]


def test_not_a_race_day():
    data = meeting("12:00:00")
    assert watch.assess(data, at("12:00:00", "2026-11-13"))["phase"] == "idle"       # the day before
    assert "今日唔係賽馬日" in watch.assess(data, at("12:00:00", "2026-11-13"))["headline"]
    assert watch.assess(data, at("12:00:00", "2026-11-15"))["headline"] == "頁面上冇未跑嘅賽馬日"
    assert watch.assess({}, at("12:00:00"))["problems"] == []


# ------------------------------------------------------------------ faults that can still be put right

def test_updates_stopped():
    report = check("12:45:00", updated=13)
    assert keys(report) == ["stalled"] and report["actions"] == ["update"]
    assert check("12:45:00", updated=12)["problems"] == []


def test_slow_morning_clock_is_allowed_as_the_window_opens():
    assert check("11:33:00", updated=28, read=28)["problems"] == []          # 3 minutes into the window
    assert keys(check("11:43:00", updated=28, read=28)) == ["stalled"]       # 13 minutes in: should have sped up


def test_odds_not_read():
    report = check("12:45:00", read=20)
    assert keys(report) == ["odds_stale"] and "2 場" in report["problems"][0]["text"]     # races 2 and 3


def test_quinella_page_not_read():
    data = meeting("12:45:00")
    old = utc_iso(DAY, "12:20:00")
    for k in ("2", "3"):
        data["odds"]["pools"][k]["last"]["t"] = old         # win and place still fresh in "latest"
    assert keys(watch.assess(data, at("12:45:00"))) == ["quinella_stale"]


def test_result_late_and_paper_not_settled():
    assert keys(check("12:59:00")) == []                                     # 29 minutes: not late yet
    assert keys(check("13:00:00")) == ["result_late:1"]
    assert keys(check("13:09:00", results=[1], settled=[])) == []            # result in, 39 minutes
    assert keys(check("13:10:00", results=[1], settled=[])) == ["paper:1"]


def test_page_shows_another_meeting():
    data = meeting("12:45:00")
    data["latest"] = {"meeting": {"date": "2026-11-11"}}
    assert keys(watch.assess(data, at("12:45:00"))) == ["latest"]


def test_ticker_stopped_and_updates_failing():
    runs = {"ticker": [{"status": "completed", "conclusion": "success", "created": at("06:00:00"), "url": "t"}],
            "update": [{"status": "completed", "conclusion": "failure", "created": at("12:40:00"), "url": "u2"},
                       {"status": "completed", "conclusion": "failure", "created": at("12:35:00"), "url": "u1"}]}
    report = check("12:45:00", runs=runs)
    assert keys(report) == ["ticker", "update_failed"] and report["actions"] == ["ticker"]
    one_bad = dict(runs, update=[runs["update"][0], {**runs["update"][1], "conclusion": "success"}])
    assert keys(check("12:45:00", runs=dict(one_bad, ticker=OK_RUNS["ticker"]))) == []    # one failure is not a pattern


def test_without_github_the_workflow_checks_are_skipped():
    assert check("12:45:00", runs=None)["problems"] == []


def test_morning_checks():
    data = meeting("09:00:00")
    data["selfcheck"]["health"]["checks"].append({"name": "馬會頁面", "ok": False})
    data["trackwork"] = {"races": {"1": {}}}
    del data["odds"]["pools"]["2"], data["odds"]["pools"]["3"]
    report = watch.assess(data, at("09:00:00"))
    assert keys(report) == ["odds_missing", "health", "trackwork"]
    assert "馬會頁面" in report["problems"][1]["text"]
    assert keys(check("10:30:00", read=50)) == ["odds_stale"]
    assert keys(check("08:20:00", read=300)) == []                           # before the day's updates start


def test_the_day_before_only_checks_odds_once_betting_is_open():
    data = meeting("12:00:00")
    data["odds"] = {"latest": {}, "history": {}, "pools": {}}
    assert watch.assess(data, at("15:00:00", "2026-11-13"))["problems"] == []
    data["odds"]["history"] = {"1": [{}]}                                    # open, but only one race read
    assert keys(watch.assess(data, at("15:00:00", "2026-11-13"))) == ["odds_missing"]


# ------------------------------------------------------------------ facts about races already run

def test_race_without_a_pre_snapshot():
    report = check("12:45:00", pre={1: None})
    assert keys(report) == ["no_pre:1"] and report["problems"][0]["lasting"] is True


def test_pre_snapshot_read_too_long_before_the_post():
    assert keys(check("12:45:00", pre={1: "12:22:00"})) == []                # 8 minutes
    assert keys(check("12:45:00", pre={1: "12:21:00"})) == ["pre_old:1"]


def test_pick_made_from_other_odds_than_the_pre_snapshot():
    data = meeting("12:45:00")
    data["odds"]["pools"]["1"]["pre"]["win"]["3"] = 2.2                      # 11 Nov race 2 looked like this
    report = watch.assess(data, at("12:45:00"))
    assert keys(report) == ["pre_differs:1"] and report["problems"][0]["lasting"] is True


def test_race_analysed_after_the_off():
    assert keys(check("12:45:00", post_hoc=[1])) == ["post_hoc:1"]


def test_a_race_is_not_judged_until_a_minute_after_its_post():
    assert keys(check("12:30:30", pre={1: None})) == []
    assert keys(check("12:31:00", pre={1: None})) == ["no_pre:1"]


# ------------------------------------------------------------------ after the meeting

def test_summary_after_the_last_race():
    report = check("14:45:00", results=[1, 2, 3])
    assert report["phase"] == "after" and report["problems"] == []
    s = report["summary"]
    assert "11/14 沙田完成:3/3 場有賽果" in s and "最長相隔 5 分鐘" in s
    assert "首選 1 勝 3 位(最後熱門 3 勝)" in s and "模擬倉今日 +$234" in s and "運作正常" in s


def test_summary_counts_faults_and_notes_apart():
    report = check("14:45:00", results=[1, 2], pre={1: None})
    assert keys(report) == ["no_pre:1", "result_late:3"]
    assert "仲有 1 個問題未解決" in report["summary"] and "1 場有備註" in report["summary"]


# ------------------------------------------------------------------ a race HKJC cancels
# 14 Nov 2026, race 6: the watch reported "no result" every 30 minutes for the rest of the day.

def cancel(data, n):
    """The files once the system knows race n was cancelled."""
    next(r for r in data["doc"]["races"] if r["race_no"] == n)["cancelled"] = True
    row = next(r for r in data["paper"]["meetings"][0]["races"] if r["race_no"] == n)
    row.update(state="void", cancelled=True, main=[{"placed": False, "void": True, "stake": 10.0, "back": None}])
    return data


def test_a_cancelled_race_is_a_note_not_a_missing_result():
    now = "13:40:00"                                                           # 40 minutes after race 2's post
    assert "result_late:2" in keys(check(now, results=[1]))
    report = watch.assess(cancel(meeting(now, results=[1]), 2), at(now))
    assert keys(report) == ["cancelled:2"] and report["problems"][0]["lasting"]
    assert report["problems"][0]["text"] == "第2場馬會宣佈取消,所有注項退款"
    assert "有賽果 1 場,取消 1 場" in report["headline"] and report["actions"] == []


def test_a_cancelled_race_is_not_judged_on_how_old_its_last_odds_were():
    now = "13:40:00"                                # its pools closed 13 minutes before the post
    assert "pre_old:2" in keys(check(now, results=[1], pre={2: "12:47:00"}))
    report = watch.assess(cancel(meeting(now, results=[1], pre={2: "12:47:00"}), 2), at(now))
    assert keys(report) == ["cancelled:2"]


def test_a_race_cancelled_before_its_post_time_is_not_an_odds_fault():
    now = "12:50:00"

    def files():
        data = meeting(now, results=[1])
        old = utc_iso(DAY, "12:20:00")
        data["odds"]["latest"]["2"]["t"] = data["odds"]["pools"]["2"]["last"]["t"] = old
        return data

    assert "odds_stale" in keys(watch.assess(files(), at(now)))
    assert keys(watch.assess(cancel(files(), 2), at(now))) == ["cancelled:2"]


def test_the_summary_names_the_cancelled_race():
    now = "14:41:00"
    report = watch.assess(cancel(meeting(now, results=[1, 3]), 2), at(now))
    assert report["phase"] == "after" and "2/3 場有賽果,取消 1 場" in report["headline"]
    assert "2/3 場有賽果(第2場馬會取消)" in report["summary"]
    assert "運作正常" in report["summary"] and "1 場有備註" in report["summary"]


def test_a_cancelled_race_is_said_once():
    now = "13:40:00"
    report = watch.assess(cancel(meeting(now, results=[1]), 2), at(now))
    state = {"day": DAY, "open": {}, "ready": True, "done": False}
    lines, state = watch.decide(dict(report, done_actions=[]), state, at(now), DAY)
    assert lines == ["賽馬系統備註:第2場馬會宣佈取消,所有注項退款"]
    lines, state = watch.decide(dict(report, done_actions=[]), state, at("14:30:00"), DAY)
    assert lines == []


# ------------------------------------------------------------------ what gets said

def say(report, state, clock):
    report.setdefault("done_actions", [])
    return watch.decide(report, state, at(clock), DAY)


def test_ready_once_then_silence():
    lines, state = say(check("11:35:00"), {}, "11:35:00")
    assert lines == ["11/14 沙田 賽前檢查正常,12:30 開第 1 場。之後有問題先會通知"]
    lines, state = say(check("11:45:00"), state, "11:45:00")
    assert lines == []
    assert say(check("09:00:00"), {}, "09:00:00")[0] == []                   # nothing before the window


def test_a_fault_is_said_once_then_again_after_half_an_hour_then_cleared():
    _, state = say(check("11:35:00"), {}, "11:35:00")
    lines, state = say(check("12:45:00", updated=13), state, "12:45:00")
    assert lines == ["賽馬系統有問題:已經 13 分鐘冇完整更新(正常約 5 分鐘一次)"]
    lines, state = say(check("12:55:00", updated=23), state, "12:55:00")
    assert lines == []
    lines, state = say(check("13:15:00", updated=43, results=[1]), state, "13:15:00")
    assert lines == ["賽馬系統問題未解決:已經 43 分鐘冇完整更新(正常約 5 分鐘一次)"]
    lines, state = say(check("13:25:00", results=[1]), state, "13:25:00")
    assert lines == ["賽馬系統回復正常"] and state["open"] == {}


def test_a_note_about_a_finished_race_is_said_once_only():
    _, state = say(check("11:35:00"), {}, "11:35:00")
    lines, state = say(check("12:45:00", pre={1: None}), state, "12:45:00")
    assert lines == ["賽馬系統備註:第1場冇開跑前賠率(模擬倉值博同測試用唔到呢場)"]
    for clock, done in (("13:20:00", [1]), ("14:00:00", [1, 2]), ("14:30:00", [1, 2, 3])):
        lines, state = say(check(clock, pre={1: None}, results=done), state, clock)
        assert lines == []


def test_a_new_fault_beside_an_old_one_is_said_together_with_it():
    _, state = say(check("11:35:00"), {}, "11:35:00")
    _, state = say(check("12:45:00", updated=13), state, "12:45:00")
    lines, _ = say(check("13:15:00", updated=43), state, "13:15:00")       # now race 1's result is late too
    assert lines == ["賽馬系統有問題:第1場開跑 45 分鐘都未有賽果;已經 43 分鐘冇完整更新(正常約 5 分鐘一次)"]


def test_what_was_restarted_is_part_of_the_message():
    report = check("12:45:00", updated=13)
    report["done_actions"] = ["update"]
    lines, _ = watch.decide(report, {"day": DAY, "open": {}, "ready": True, "done": False}, at("12:45:00"), DAY)
    assert lines == ["賽馬系統有問題:已經 13 分鐘冇完整更新(正常約 5 分鐘一次)。已重新觸發更新"]


def test_not_ready_while_something_is_wrong():
    lines, state = say(check("11:45:00", updated=20), {}, "11:45:00")
    assert len(lines) == 1 and lines[0].startswith("賽馬系統有問題") and state["ready"] is False
    lines, state = say(check("11:55:00"), state, "11:55:00")
    assert lines == ["賽馬系統回復正常", "11/14 沙田 賽前檢查正常,12:30 開第 1 場。之後有問題先會通知"]


def test_summary_is_said_once():
    _, state = say(check("11:35:00"), {}, "11:35:00")
    lines, state = say(check("14:45:00", results=[1, 2, 3]), state, "14:45:00")
    assert len(lines) == 1 and lines[0].startswith("11/14 沙田完成")
    assert say(check("14:55:00", results=[1, 2, 3]), state, "14:55:00")[0] == []


def test_a_new_day_starts_clean_and_long_messages_are_cut():
    old = {"day": "2026-11-11", "open": {"stalled": {"since": "x", "said": "x"}}, "ready": True, "done": True}
    lines, state = say(check("11:35:00"), old, "11:35:00")
    assert state["day"] == DAY and len(lines) == 1 and "賽前檢查正常" in lines[0]
    report = {"phase": "live", "ready": None, "summary": None,
              "problems": [{"key": f"k{i}", "text": "問題" * 20, "lasting": False} for i in range(8)]}
    lines, _ = say(report, {}, "12:00:00")
    assert len(lines[0]) <= 195 and lines[0].endswith("…")


# ------------------------------------------------------------------ what it may restart

@pytest.fixture
def gh(monkeypatch):
    calls = []
    monkeypatch.setattr(watch.shutil, "which", lambda name: "gh")
    monkeypatch.setattr(watch.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, b"", b""))
    return calls


def runs(last_update_minutes_ago, status="completed", now="12:45:00"):
    return {"ticker": OK_RUNS["ticker"],
            "update": [{"status": status, "conclusion": None if status != "completed" else "success",
                        "created": at(now) - timedelta(minutes=last_update_minutes_ago), "url": "u"}]}


def test_stalled_update_is_started_again(gh):
    report = {"actions": ["update"]}
    assert watch.act(report, runs(14), at("12:45:00")) == ["update"]
    assert gh == [["gh", "workflow", "run", "update.yml", "--repo", watch.REPO]]


def test_update_is_not_started_while_one_is_running_or_just_started(gh):
    assert watch.act({"actions": ["update"]}, runs(14, status="in_progress"), at("12:45:00")) == []
    assert watch.act({"actions": ["update"]}, runs(5), at("12:45:00")) == []
    assert gh == []


def test_ticker_is_started_again_and_nothing_else_ever_is(gh):
    assert watch.act({"actions": ["ticker", "ticker", "selfcheck", "pages"]}, runs(3), at("12:45:00")) == ["ticker"]
    assert gh == [["gh", "workflow", "run", "ticker.yml", "--repo", watch.REPO]]


def test_nothing_is_started_without_gh_or_without_run_information(monkeypatch, gh):
    assert watch.act({"actions": ["update"]}, None, at("12:45:00")) == []
    monkeypatch.setattr(watch.shutil, "which", lambda name: None)
    assert watch.act({"actions": ["update"]}, runs(14), at("12:45:00")) == []
    assert gh == []


def test_watch_needs_only_the_standard_library():
    code = "import sys, hkjc.watch; print([m for m in ('numpy', 'requests', 'bs4', 'playwright') if m in sys.modules])"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=watch.store.ROOT)
    assert done.stdout.strip() == "[]", done.stderr
