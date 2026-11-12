"""New-information test (hkjc/livetest.py): how each race is turned into a sample, how a
candidate is judged, and when the verdict is given. Made-up races only."""
import math
from datetime import date, timedelta

import numpy as np
import pytest

from helpers import code, make_card, make_result, odds_read, utc_iso
from hkjc import livetest, store

DAY, VENUE = "2026-11-14", "ST"


def days_before(n):
    return (date.fromisoformat(DAY) - timedelta(days=n)).isoformat()


# ------------------------------------------------------------------ trackwork and vet records

def test_fast_work_counts_the_fourteen_days_before_race_day():
    horse = {"work": {"快操": [[DAY, "race morning"], [days_before(1), "x"], [days_before(14), "x"],
                             [days_before(15), "too old"]],
                      "踱步": [[days_before(2), "not fast work"]]}, "vet": []}
    assert livetest._work(horse, DAY) == (2.0, 0.0)


def test_vet_record_counts_within_ninety_days():
    assert livetest._work({"vet": [[days_before(90), "跛行", ""]]}, DAY) == (0.0, 1.0)
    assert livetest._work({"vet": [[days_before(91), "跛行", ""]]}, DAY) == (0.0, 0.0)
    assert livetest._work({"vet": [[DAY, "race day itself", ""]]}, DAY) == (0.0, 0.0)
    assert livetest._work({}, DAY) == (0.0, 0.0)


def test_opening_odds_are_the_first_read_on_race_day_in_hong_kong_time():
    history = [{"t": utc_iso(days_before(1), "23:50:00"), "win": {"1": 5.0}},
               {"t": utc_iso(DAY, "00:30:00"), "win": {"1": 4.0}},       # 16:30 UTC the day before
               {"t": utc_iso(DAY, "08:00:00"), "win": {"1": 3.0}}]
    assert livetest._open_win(history, DAY) == {1: 4.0}
    assert livetest._open_win(history[:1], DAY) == {}


# ------------------------------------------------------------------ building the samples

def save_meeting(day=DAY, venue=VENUE):
    """Race 1: everything saved, horse 8 withdrawn. Race 2: no race-day opening odds, vet page
    not read. Race 3: only six priced runners. Race 4: no result yet."""
    card = make_card(day, venue, {1: "12:30", 2: "13:00", 3: "13:55", 4: "14:25"})
    races = {r["race_no"]: r for r in card["races"]}
    open_win = {str(n): 8.0 for n in range(1, 9)}
    pre = {n: odds_read(races[n], fav=3, read_at=utc_iso(day, "12:25:00")) for n in races}
    pre[3]["win"] = {n: o for n, o in pre[3]["win"].items() if n <= 6}
    store.write_json(store.DATA / "odds" / f"{day}_{venue}.json", {
        "latest": {},
        "history": {"1": [{"t": utc_iso(day, "08:00:00"), "win": open_win}],
                    "2": [{"t": utc_iso(days_before(1), "20:00:00"), "win": open_win}]},
        "pools": {str(n): {"pre": pre[n]} for n in races}})
    order = [3, 1, 2, 4, 5, 6, 7]
    store.save_results([make_result(races[1], order, codes={8: "WV"}), make_result(races[2], order + [8]),
                        make_result(races[3], order + [8])])
    horses = lambda n: {code(n, i): {"no": str(i), "horse": "", "reserve": False, "work": {}, "vet": []}
                        for i in range(1, 9)}
    h1 = horses(1)
    h1[code(1, 3)]["work"] = {"快操": [[days_before(2), "x"], [days_before(5), "x"]]}
    h1[code(1, 2)]["vet"] = [[days_before(30), "跛行", days_before(20)]]
    (store.DATA / "trackwork").mkdir(exist_ok=True)
    store.write_json(store.DATA / "trackwork" / f"{day}_{venue}.json", {"date": day, "venue": venue, "races": {
        "1": {"vet_ok": True, "horses": h1}, "2": {"vet_ok": False, "horses": horses(2)}}})


def test_build_makes_one_sample_per_usable_race(sandbox):
    save_meeting()
    out = livetest.build()
    assert [s["key"] for s in out] == [(DAY, VENUE, 1), (DAY, VENUE, 2)]      # race 3 too small, race 4 not run


def test_sample_holds_the_starters_priced_before_the_off(sandbox):
    save_meeting()
    s = livetest.build()[0]
    # runners in finishing order 3, 1, 2, 4, 5, 6, 7; horse 8 was withdrawn
    assert s["y"] == 0 and len(s["logq"]) == 7
    assert np.exp(s["logq"]).sum() == pytest.approx(1)
    q3 = (1 / 1.9) / (1 / 1.9 + sum(1 / (6.0 + n) for n in (1, 2, 4, 5, 6, 7)))
    assert math.exp(s["logq"][0]) == pytest.approx(q3)


def test_steam_is_the_change_in_market_chance_since_the_opening(sandbox):
    save_meeting()
    s = livetest.build()[0]
    q3 = math.exp(s["logq"][0])
    assert s["f"]["steam"][0] == pytest.approx(math.log(q3 / (1 / 7)))        # opened level at 8.0 each
    assert s["f"]["steam"][0] > 0 and all(v < 0 for v in s["f"]["steam"][1:])


def test_trackwork_candidates_come_from_the_file_saved_before_the_race(sandbox):
    save_meeting()
    s = livetest.build()[0]
    assert list(s["f"]["fast_work"]) == [2, 0, 0, 0, 0, 0, 0]                 # horse 3: two gallops
    assert list(s["f"]["vet_recent"]) == [0, 0, 1, 0, 0, 0, 0]                # horse 2 (third home)


def test_a_candidate_is_left_out_of_races_where_it_cannot_be_measured(sandbox):
    save_meeting()
    s = livetest.build()[1]
    assert s["f"]["steam"] is None                                            # no race-day opening odds
    assert s["f"]["fast_work"] is None and s["f"]["vet_recent"] is None       # vet page not read


def test_races_before_the_start_date_are_not_used(sandbox):
    save_meeting(day="2026-11-11", venue="HV")
    assert livetest.build() == []
    assert livetest.FROM == "2026-11-14"


# ------------------------------------------------------------------ judging a candidate

def made_up(n=300, runners=8, effect=1.0):
    """Races where 'steam' really moves the winner's chance and 'fast_work' is noise."""
    rng = np.random.RandomState(11)
    out = []
    for i in range(n):
        logq = np.log(rng.dirichlet(np.full(runners, 3.0)))
        real, noise = rng.normal(size=runners), rng.normal(size=runners)
        p = np.exp(logq + effect * real)
        out.append({"key": i, "y": int(rng.choice(runners, p=p / p.sum())), "logq": logq,
                    "f": {"steam": real, "fast_work": noise, "vet_recent": None}})
    return out


def test_judge_finds_real_information_and_the_shuffled_copy_shows_none():
    rows = made_up()
    real = livetest.judge(rows, "steam")
    assert real["gain"] > 0 and real["z"] > 4
    assert livetest.judge(rows, "steam", placebo=True)["z"] < livetest.Z_PASS


def test_judge_does_not_pass_noise():
    assert livetest.judge(made_up(), "fast_work")["z"] < livetest.Z_PASS


def test_judge_is_repeatable():
    rows = made_up(n=120)
    assert livetest.judge(rows, "steam", placebo=True) == livetest.judge(rows, "steam", placebo=True)


# ------------------------------------------------------------------ when the verdict is given

@pytest.fixture
def looks(monkeypatch):
    """status() over n made-up samples, with judge replaced: returns the rows and what was judged."""
    monkeypatch.setattr(livetest, "LOOKS", (4, 8))

    def run(n, z=0.0, placebo_z=0.0, gain=0.01):
        seen = []
        rows = [{"key": i, "f": {"steam": [0.0], "fast_work": [0.0] if i % 2 == 0 else None, "vet_recent": None}}
                for i in range(n)]
        monkeypatch.setattr(livetest, "build", lambda: rows)

        def judge(samples, cand, placebo=False):
            seen.append((cand, len(samples), placebo))
            return {"gain": gain, "z": placebo_z if placebo else z}
        monkeypatch.setattr(livetest, "judge", judge)
        return livetest.status()["candidates"], seen
    return run


def test_no_verdict_before_the_first_look(looks):
    rows, judged = looks(3, z=9.0)
    assert judged == []
    assert rows["steam"] == {"name": livetest.NAMES_ZH["steam"], "races": 3, "next": 4}


def test_each_candidate_counts_only_its_own_races(looks):
    rows, judged = looks(6, z=9.0)
    assert (rows["steam"]["races"], rows["fast_work"]["races"], rows["vet_recent"]["races"]) == (6, 3, 0)
    assert {c for c, _, _ in judged} == {"steam"}             # fast_work has 3 races: not at its look yet


def test_the_look_uses_exactly_the_first_races_up_to_it(looks):
    _, judged = looks(7, z=9.0)
    assert ("steam", 4, False) in judged and ("steam", 4, True) in judged


def test_pass_needs_z_2_5_and_a_quiet_placebo(looks):
    assert looks(5, z=2.5, placebo_z=0.9)[0]["steam"]["verdict"] == "過關"
    assert looks(5, z=2.49, placebo_z=0.0)[0]["steam"]["verdict"] == "未過關"
    assert looks(5, z=3.0, placebo_z=1.0)[0]["steam"]["verdict"] == "未過關"
    assert looks(5, z=3.0, placebo_z=0.0, gain=-0.01)[0]["steam"]["passed"] is False


def test_the_last_look_closes_the_test(looks):
    row = looks(9, z=1.0)[0]["steam"]
    assert (row["look"], row["next"], row["verdict"]) == (8, None, "未過關,完結")
    assert looks(9, z=3.0)[0]["steam"]["verdict"] == "過關"
