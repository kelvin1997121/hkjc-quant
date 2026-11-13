"""Daily self-check (hkjc/selfcheck.py): health checks and the record of self-improvement tests."""
from datetime import date, timedelta

import numpy as np
import pytest

from helpers import analysis_doc, hkt, make_card, odds_read, utc_iso
from hkjc import model, selfcheck, store

DAY, VENUE = "2026-11-20", "ST"


# ------------------------------------------------------------------ odds health check

def save_odds(n_races, with_history, with_pre):
    card = make_card(DAY, VENUE, {n: "12:30" for n in range(1, n_races + 1)})
    path = store.DOCS / "data" / f"{DAY}_{VENUE}.json"
    store.write_json(path, analysis_doc(DAY, VENUE, [{"race_no": r["race_no"]} for r in card["races"]]))
    snap = lambda r: odds_read(r, 1, utc_iso(DAY, "12:25:00"))
    store.write_json(store.DATA / "odds" / f"{DAY}_{VENUE}.json", {
        "latest": {},
        "history": {str(r["race_no"]): [{"t": utc_iso(DAY, "12:25:00"), "win": snap(r)["win"]}]
                    for r in card["races"][:with_history]},
        "pools": {str(r["race_no"]): {"pre": snap(r)} for r in card["races"][:with_pre]}})
    return path


def test_odds_check_passes_when_races_have_history_and_a_pre_snapshot(sandbox):
    c = selfcheck.odds_check(save_odds(10, with_history=10, with_pre=9))
    assert c["ok"] is True
    assert "10/10 場賠率紀錄" in c["detail"] and "9/10 場有開跑前賠率" in c["detail"]


def test_odds_check_fails_when_pre_snapshots_are_missing(sandbox):
    c = selfcheck.odds_check(save_odds(10, with_history=10, with_pre=7))     # e.g. quinella page unreadable
    assert c["ok"] is False and "7/10 場有開跑前賠率" in c["detail"]


def test_odds_check_fails_when_odds_were_not_read(sandbox):
    assert selfcheck.odds_check(save_odds(10, with_history=5, with_pre=5))["ok"] is False
    path = save_odds(10, 10, 10)
    (store.DATA / "odds" / f"{DAY}_{VENUE}.json").unlink()
    assert selfcheck.odds_check(path)["ok"] is False


# ------------------------------------------------------------------ live record check

def live(n, z_win, z_place):
    return {"from": "2026-11-11", "races": n, "wins": n // 4, "places": n // 2, "fav_wins": n / 4,
            "fav_places": n / 2, "z_win_vs_fav": z_win, "z_place_vs_fav": z_place}


def test_live_check_waits_for_thirty_races():
    c = selfcheck.live_check(live(29, -5.0, -5.0))
    assert c["ok"] is None and "由 11/11 起" in c["detail"]


def test_live_check_compares_with_the_final_favourite():
    assert selfcheck.live_check(live(40, -1.9, 0.3))["ok"] is True
    assert selfcheck.live_check(live(40, 0.5, -2.1))["ok"] is False
    assert selfcheck.live_check(live(40, None, None))["ok"] is True      # never differed
    assert selfcheck.live_check({})["ok"] is None


# ------------------------------------------------------------------ did the meeting get its updates

def meeting(minutes_between, hole=None):
    """Updates every `minutes_between` minutes from 11:30 to 18:55; hole = (start, end) with none."""
    runs, t = [], hkt(DAY, "11:30:00")
    while t <= hkt(DAY, "18:55:00"):
        if not (hole and hkt(DAY, hole[0]) < t < hkt(DAY, hole[1])):
            runs.append(t.astimezone(selfcheck.timezone.utc).isoformat(timespec="seconds"))
        t += timedelta(minutes=minutes_between)
    return {"meeting": {"date": DAY}, "races": [{"post_time": "12:30"}, {"post_time": "18:10"}], "runs": runs}


def test_meeting_updates_pass_on_the_five_minute_clock():
    c = selfcheck.meeting_updates(meeting(5))
    assert c["ok"] is True and "最長相隔 5 分鐘" in c["detail"]


def test_meeting_updates_catch_a_gap_and_too_few_runs():
    assert selfcheck.meeting_updates(meeting(5, hole=("14:00:00", "14:40:00")))["ok"] is False
    assert selfcheck.meeting_updates(meeting(30))["ok"] is False
    assert selfcheck.meeting_updates({"meeting": {"date": DAY}, "races": [{}], "runs": []})["ok"] is False


# ------------------------------------------------------------------ self-improvement record

def samples(before=35, after=25, runners=8):
    """Races with random factors and winners; `before` of them ahead of the tests' start date."""
    rng = np.random.RandomState(7)
    k = len(model.ALL_FEATURES)
    days = ["2026-10-01"] * before + ["2026-11-20"] * after
    return [{"date": d, "X": rng.normal(size=(runners, k)), "y": int(rng.randint(runners)),
             "odds": np.full(runners, 8.0), "order": None, "nos": list(range(1, runners + 1)), "dividends": None}
            for d in days]


CFG = {"l2": 0.05, "stages": 1}
N_TESTS = len(model.TRIAL_FEATURES) + len(selfcheck.GROUPS)
N_FITS = len(selfcheck.GROUPS) + 2 * len(model.TRIAL_FEATURES)     # an added factor is also run shuffled


@pytest.fixture
def judged(sandbox, monkeypatch):
    """learning() with the model fitting replaced: each test gets the z set here.
    judged.z_add / z_drop / z_placebo, and judged.drop_passes = a group whose removal passes."""
    monkeypatch.setattr(model, "FEATURES", list(model.ALL_FEATURES))
    monkeypatch.setattr(selfcheck, "LOOKS", (10, 50))
    from types import SimpleNamespace
    state = SimpleNamespace(calls=0, z_add=0.0, z_drop=0.0, z_placebo=0.0, drop_passes=None)

    def forward_gain(samples, start, base_cols, cand_cols, cfg, shuffle=None):
        state.calls += 1
        if shuffle:
            return 0.0, state.z_placebo
        if len(cand_cols) > len(base_cols):
            return 0.01, state.z_add
        gone = set(base_cols) - set(cand_cols)
        if state.drop_passes and gone == set(selfcheck.GROUPS[state.drop_passes]):
            return 0.01, 3.0
        return 0.0, state.z_drop

    monkeypatch.setattr(selfcheck, "forward_gain", forward_gain)
    return state


def log():
    return store.read_json(selfcheck.LEARN_PATH)


def config():
    return store.read_json(store.DATA / "model_config.json")


def test_each_look_is_taken_once_and_remembered(judged):
    rows = selfcheck.learning(samples(), CFG)
    assert len(rows) == len(log()["tests"]) == N_TESTS
    assert judged.calls == N_FITS
    assert all(t["races"] == 25 and [lk["at"] for lk in t["looks"]] == [10] for t in log()["tests"].values())
    assert all(r["status"] == "測試中" and r["next"] == 50 for r in rows)

    before = log()["tests"]
    selfcheck.learning(samples(), CFG)                # the next night, same 25 races
    assert judged.calls == N_FITS                     # nothing judged again
    assert log()["tests"] == before


def test_without_the_saved_record_every_test_is_judged_again(judged):
    # what would have happened nightly from 300 races on, while data/learning.json was not committed
    selfcheck.learning(samples(), CFG)
    selfcheck.LEARN_PATH.unlink()
    selfcheck.learning(samples(), CFG)
    assert judged.calls == 2 * N_FITS


def test_no_look_before_the_first_checkpoint(judged, monkeypatch):
    monkeypatch.setattr(selfcheck, "LOOKS", (300, 500, 800))
    rows = selfcheck.learning(samples(), CFG)
    assert judged.calls == 0
    assert all(r["status"] == "測試中" and r["races"] == 25 and r["next"] == 300 and r["last"] is None for r in rows)
    assert config() is None


def test_only_races_after_registration_count(judged):
    selfcheck.learning(samples(before=60, after=4), CFG)       # 64 races in all, 4 of them new
    assert judged.calls == 0
    assert all(t["races"] == 4 for t in log()["tests"].values())


def test_a_factor_is_adopted_at_z_2_5_when_the_shuffled_copy_shows_nothing(judged):
    judged.z_add, judged.z_placebo = 3.0, 0.4
    rows = selfcheck.learning(samples(), CFG)
    assert config()["added"] == list(model.TRIAL_FEATURES)
    assert [r["status"] for r in rows[:len(model.TRIAL_FEATURES)]] == ["已採用"] * len(model.TRIAL_FEATURES)
    assert model.active_features(config())[-len(model.TRIAL_FEATURES):] == list(model.TRIAL_FEATURES)


def test_a_factor_is_not_adopted_when_the_shuffled_copy_gains_too(judged):
    judged.z_add, judged.z_placebo = 3.0, 1.5
    selfcheck.learning(samples(), CFG)
    assert config() is None
    assert all(t["status"] == "testing" for t in log()["tests"].values())


def test_a_factor_is_not_adopted_below_z_2_5(judged):
    judged.z_add, judged.z_placebo = 2.4, 0.0
    selfcheck.learning(samples(), CFG)
    assert config() is None


def test_a_factor_group_is_retired_when_dropping_it_passes(judged):
    judged.drop_passes = "試閘"
    selfcheck.learning(samples(), CFG)
    assert config()["removed"] == selfcheck.GROUPS["試閘"]
    assert log()["tests"]["drop:試閘"]["status"] == "retired"
    assert not set(selfcheck.GROUPS["試閘"]) & set(model.active_features(config()))


def test_the_last_look_closes_a_test_either_way(judged, monkeypatch):
    monkeypatch.setattr(selfcheck, "LOOKS", (10, 20))
    selfcheck.learning(samples(), CFG)                         # 25 races: straight to the last look
    tests = log()["tests"]
    assert all(t["looks"][0]["at"] == 20 for t in tests.values())
    assert {t["status"] for t in tests.values() if t["kind"] == "add"} == {"rejected"}
    assert {t["status"] for t in tests.values() if t["kind"] == "drop"} == {"kept"}
    selfcheck.learning(samples(after=40), CFG)                 # more races later: closed tests stay closed
    assert judged.calls == N_FITS


def test_forward_gain_finds_a_real_factor_and_loses_it_when_shuffled(sandbox, monkeypatch):
    real_fit = model.fit
    monkeypatch.setattr(model, "fit", lambda rows, **kw: real_fit(rows, iters=150, **kw))
    monkeypatch.setattr(model, "FEATURES", ["form", "speed_best"])
    rng = np.random.RandomState(3)
    rows = []
    for _ in range(150):
        X = rng.normal(size=(8, 2))
        p = np.exp(1.5 * X[:, 1])                              # only the second column decides the winner
        rows.append({"date": "2026-11-20", "X": X, "y": int(rng.choice(8, p=p / p.sum())), "order": None})
    gain, z = selfcheck.forward_gain(rows, 60, ["form"], ["form", "speed_best"], CFG)
    _, shuffled = selfcheck.forward_gain(rows, 60, ["form"], ["form", "speed_best"], CFG, shuffle=["speed_best"])
    assert gain > 0 and z > 4
    assert shuffled < 1


# ------------------------------------------------------------------ odds before the next meeting

NEXT, NEXT_VENUE = "2026-11-14", "ST"


def coming_meeting(read, four_pools=None, timed=None, races=11):
    """schedule.json for a meeting on 14 Nov and its odds file: `read` races have a snapshot,
    `four_pools` of them with quinella odds, `timed` of them with their own read time."""
    four_pools = read if four_pools is None else four_pools
    timed = read if timed is None else timed
    card = make_card(NEXT, NEXT_VENUE, {n: "12:30" for n in range(1, races + 1)})
    store.write_json(store.DOCS / "data" / "schedule.json",
                     {"date": NEXT, "venue": NEXT_VENUE, "posts": {str(n): "12:30" for n in range(1, races + 1)}})
    if not read:
        return
    snap = lambda r: odds_read(r, 1, utc_iso("2026-11-13", "13:00:00"))
    rs = card["races"]
    latest = {str(r["race_no"]): {"win": snap(r)["win"], **({"t": snap(r)["t"]} if i < timed else {})}
              for i, r in enumerate(rs[:read])}
    store.write_json(store.DATA / "odds" / f"{NEXT}_{NEXT_VENUE}.json", {
        "latest": latest,
        "history": {str(r["race_no"]): [{"t": snap(r)["t"], "win": snap(r)["win"]}] for r in rs[:read]},
        "pools": {str(r["race_no"]): {"last": snap(r)} for r in rs[:four_pools]}})


def on(day, monkeypatch):
    monkeypatch.setattr(selfcheck, "today_hkt", lambda: date.fromisoformat(day))
    return selfcheck.next_odds_check()


def test_next_odds_check_passes_on_race_day_when_every_race_was_read(sandbox, monkeypatch):
    coming_meeting(read=11)
    c = on("2026-11-14", monkeypatch)
    assert c["ok"] is True and c["name"] == "賽前賠率"
    assert "11/14 已讀到 11/11 場" in c["detail"] and "11 場有讀取時間" in c["detail"]


def test_next_odds_check_fails_on_race_day_when_odds_were_not_read(sandbox, monkeypatch):
    coming_meeting(read=0)
    c = on("2026-11-14", monkeypatch)
    assert c["ok"] is False and "開跑前要檢查" in c["detail"]


def test_next_odds_check_fails_without_quinella_odds_or_read_times(sandbox, monkeypatch):
    coming_meeting(read=11, four_pools=3)                 # the 連贏/位置Q page stopped working
    assert on("2026-11-14", monkeypatch)["ok"] is False
    coming_meeting(read=11, timed=0)                      # odds saved without each race's read time
    assert on("2026-11-14", monkeypatch)["ok"] is False


def test_next_odds_check_waits_until_betting_opens(sandbox, monkeypatch):
    coming_meeting(read=0)
    assert on("2026-11-13", monkeypatch)["ok"] is None    # 03:40 the day before: not open yet
    assert on("2026-11-12", monkeypatch)["ok"] is None
    coming_meeting(read=11)
    assert on("2026-11-13", monkeypatch)["ok"] is True    # read later the day before
    assert on("2026-11-12", monkeypatch)["ok"] is None    # two days ahead: nothing is read yet


def test_next_odds_check_says_nothing_without_a_coming_meeting(sandbox, monkeypatch):
    assert on("2026-11-14", monkeypatch) is None          # no schedule at all
    coming_meeting(read=11)
    assert on("2026-11-15", monkeypatch) is None          # the meeting on the page is over
