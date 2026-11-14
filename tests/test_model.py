"""Model (hkjc/model.py): what counts as a run, market chances, and no look-ahead."""
import copy
from datetime import date, timedelta

import numpy as np
import pytest

from helpers import make_card, make_result
from hkjc import model


@pytest.mark.parametrize("raw, place, ran", [
    ("1", 1, True), ("12", 12, True), ("3 平頭馬", 3, True),
    ("PU", None, True), ("DNF", None, True), ("UR", None, True), ("FE", None, True),
    ("TNP", None, True), ("DISQ", None, True),
    ("WV", None, False), ("WV-A", None, False), ("WX", None, False), ("WX-A", None, False),
    ("VOID", None, False), ("", None, False),
])
def test_started(raw, place, ran):
    assert model.started({"place_raw": raw, "place": place}) is ran


def test_gear_flags_read_hkjc_suffixes():
    assert model.gear_flags("B1/TT") == {"first": 1, "off": 0, "back": 0}
    assert model.gear_flags("B-/CP2/H1") == {"first": 1, "off": 1, "back": 1}
    assert model.gear_flags("") == model.gear_flags(None) == {"first": 0, "off": 0, "back": 0}


def test_market_probs_remove_the_takeout():
    q = model.market_probs([2.0, 4.0, 8.0])
    assert q.sum() == pytest.approx(1) and q[0] == pytest.approx(2 * q[1])
    assert model.market_probs([2.0, 0, 8.0]) is None          # a missing price: no market view
    assert model.market_probs([]) is None


def test_blend_with_no_model_weight_keeps_the_markets_order():
    q = model.market_probs([6.0, 2.0, 12.0, 4.0])
    p = np.array([0.7, 0.1, 0.1, 0.1])                         # the model loves the first horse
    market_only = model.blend(p, q, 0.0, 1.05)
    assert list(np.argsort(-market_only)) == list(np.argsort(-q))
    assert market_only.sum() == pytest.approx(1)
    assert np.argmax(model.blend(p, q, 1.0, 0.0)) == 0


def test_class_and_band():
    assert model.class_num("第四班") == 4 and model.class_num("一級賽") == -2 and model.class_num("新馬賽") is None
    assert model.band_top({"rating_band": "60-40"}) == 60
    assert model.band_top({"rating_band": "", "class": "第三班"}) == 80


def history(meetings=14, runners=8):
    """Weekly one-race meetings with the same horses and a different finishing order each time."""
    races, day = [], date(2026, 8, 30)
    for i in range(meetings):
        d = (day + timedelta(days=7 * i)).isoformat()
        race = make_card(d, "ST", {1: "12:30"}, n=runners)["races"][0]
        order = [(j * (i + 3)) % runners + 1 for j in range(runners)]
        if len(set(order)) < runners:                          # not a permutation for this i: rotate instead
            order = [(j + i) % runners + 1 for j in range(runners)]
        races.append(make_result(race, order))
    return races


def features_of_last(races):
    return model.build_dataset(races)[1][-1]["X"]


def test_features_of_a_race_do_not_use_its_own_result(sandbox):
    races = history()
    base = features_of_last(races)
    changed = copy.deepcopy(races)
    last = changed[-1]
    flipped = make_result(make_card(last["date"], "ST", {1: "12:30"})["races"][0],
                          [x["no"] for x in reversed(last["runners"])],
                          odds={x["no"]: 99.0 - x["no"] for x in last["runners"]})
    changed[-1] = flipped
    again = model.build_dataset(changed)
    assert np.array_equal(again[1][-1]["X"], base)            # same factors whatever happened in the race
    assert again[1][-1]["y"] != model.build_dataset(races)[1][-1]["y"]


def test_features_do_use_earlier_results(sandbox):
    # the check above could only pass by accident if results never reached the factors at all
    races = history()
    changed = copy.deepcopy(races)
    prev = changed[-2]
    changed[-2] = make_result(make_card(prev["date"], "ST", {1: "12:30"})["races"][0],
                              [x["no"] for x in reversed(prev["runners"])])
    assert not np.array_equal(features_of_last(changed), features_of_last(races))


def test_withdrawn_horses_leave_no_run_behind(sandbox):
    race = make_card("2026-09-06", "ST", {1: "12:30"})["races"][0]
    st = model.State()
    st.update(make_result(race, [1, 2, 3, 4, 5, 6, 7], codes={8: "WV"}))
    assert len(st.horse["A001"]) == 1 and "A008" not in st.horse


# ------------------------------------------------------------------ jockeys and trainers without an id
# A jockey or trainer who has left Hong Kong has no profile page any more, so results fetched
# later carry the name but no id (22% of the 2022/23 season's runners had no jockey id). Until
# 14 Nov 2026 the model kept no record for them.

def test_who_is_the_id_or_failing_that_the_name():
    assert model.who({"jockey": "潘頓", "jockey_id": "PZ"}, "jockey") == "PZ"
    assert model.who({"jockey": "賀銘年", "jockey_id": None}, "jockey") == "name:賀銘年"
    assert model.who({"trainer": "容天鵬"}, "trainer") == "name:容天鵬"
    assert model.who({"jockey": "", "jockey_id": None}, "jockey") is None
    assert model.who({}, "trainer") is None


def without_ids(races, jockey_no=1, trainer_no=1):
    """The same history, with horse `jockey_no`'s jockey and horse `trainer_no`'s trainer known
    by name only, as HKJC shows those who have left."""
    out = copy.deepcopy(races)
    for r in out:
        for x in r["runners"]:
            if x["no"] == jockey_no:
                x["jockey_id"] = None
            if x["no"] == trainer_no:
                x["trainer_id"] = None
    return out


def test_a_jockey_and_trainer_known_by_name_only_keep_their_record(sandbox):
    races = history()
    F = model.FEATURES
    with_ids, by_name = features_of_last(races), features_of_last(without_ids(races))
    for name in ("jockey", "trainer", "jockey_form", "trainer_form", "combo", "trainer_venue"):
        j = F.index(name)
        assert by_name[:, j] == pytest.approx(with_ids[:, j]), name       # the name stands in for the id
    assert len(set(np.round(with_ids[:, F.index("jockey")], 6))) > 1      # and those records do differ by rider


def test_the_record_is_kept_under_the_name(sandbox):
    st = model.State()
    for r in without_ids(history()):
        st.update(r)
    assert len(st.jockey.d["name:騎師1"]) == 14 and len(st.trainer.d["name:練馬師1"]) == 14
    assert None not in st.jockey.d and None not in st.trainer.d
    assert ("name:練馬師1", "ST") in st.trainer_venue.d and (None, "ST") not in st.trainer_venue.d
    assert st.horse["A001"][-1]["jockey"] == "name:騎師1"


def test_nobody_is_recorded_for_a_runner_with_neither_id_nor_name(sandbox):
    races = without_ids(history())
    for r in races:
        for x in r["runners"]:
            if x["no"] == 1:
                x["jockey"] = x["trainer"] = ""
    st = model.State()
    for r in races:
        st.update(r)
    assert all(k for k in st.jockey.d) and all(k[0] for k in st.trainer_venue.d)
    assert len(st.horse["A001"]) == 14                                    # the horse's own runs still count


# ------------------------------------------------------------------ what a course adds to a speed figure
# A course's standard time comes from its winners of every class, so the same horse gets a higher
# figure where the usual winner is a lesser horse (audit of 14 Nov 2026). course_adj, on trial
# from 15 Nov 2026, is that amount for the course of the horse's last figure.

def two_courses(meetings=16, days=7, classes=("第四班", "第四班")):
    """The same eight horses, a race every `days` days, by turns over 1200 m and 1400 m."""
    races, day = [], date(2026, 6, 7)
    for i in range(meetings):
        d = (day + timedelta(days=days * i)).isoformat()
        race = make_card(d, "ST", {1: "12:30"})["races"][0]
        race["distance"], race["class"] = (1200, 1400)[i % 2], classes[i % 2]
        result = make_result(race, [(j + i) % 8 + 1 for j in range(8)])
        result["distance"], result["class"] = race["distance"], race["class"]
        races.append(result)
    return races


def replay(races):
    st = model.State()
    for r in races:
        st.update(r)
    return st


def test_runs_at_two_courses_close_together_in_one_class_are_paired(sandbox):
    st = replay(two_courses())
    pairs = {k: v[0] for k, v in st.course_pairs.items()}
    a, b = ("ST", "TURF", 1200), ("ST", "TURF", 1400)
    assert set(pairs) == {(a, b), (b, a)} and min(pairs.values()) > 20


@pytest.mark.parametrize("kw", [{"days": 50},                                 # too far apart to compare
                                {"classes": ("第四班", "第三班")}])            # not the same class
def test_runs_far_apart_or_in_different_classes_are_not_paired(sandbox, kw):
    assert not replay(two_courses(**kw)).course_pairs


def test_runs_at_one_course_tell_nothing_about_courses(sandbox):
    assert not replay(history()).course_pairs and replay(history()).course_offsets() == {}


def test_offsets_are_the_shrunk_fit_to_the_pairs(sandbox):
    st = model.State()
    a, b, c = ("HV", "TURF", 1200), ("ST", "TURF", 1400), ("ST", "TURF", 2000)
    st.course_pairs[(b, a)] = [200, 200 * 3.0]         # the same horses run 3 points higher at a than at b
    st.course_pairs[(c, b)] = [4, 4 * 10.0]            # four pairs only: next to nothing is made of them
    off = st.course_offsets()
    assert off[a] - off[b] == pytest.approx(2.67, abs=0.1)        # 3.0 shrunk by the ridge
    assert abs(off[c] - off[b]) < 1.0
    st.course_pairs[(b, a)][0] += 200                  # more pairs with no gap: the estimate moves
    assert st.course_offsets()[a] - st.course_offsets()[b] < 2.0


def test_course_adj_is_the_offset_of_the_last_figures_course(sandbox):
    st = replay(two_courses())
    last = st.horse["A001"][-1]
    assert last["speed"] is not None and last["tk"] == ("ST", "TURF", 1400)
    st.course_pairs.clear()
    st.course_pairs[(("ST", "TURF", 1200), ("ST", "TURF", 1400))] = [300, 300 * 2.0]
    off = st.course_offsets()
    assert off[("ST", "TURF", 1400)] > 0.9
    assert st.runner_profile("A001", 1400)["course_adj"] == pytest.approx(-off[("ST", "TURF", 1400)])
    assert st.runner_profile("nobody", 1400)["course_adj"] is None


def test_factors_on_trial_do_not_use_the_races_own_result_either(sandbox, monkeypatch):
    monkeypatch.setattr(model, "FEATURES", list(model.ALL_FEATURES))
    races = two_courses()
    base = features_of_last(races)
    j = model.FEATURES.index("course_adj")
    assert np.abs(base[:, j]).max() > 0                # the factor is alive in this history
    changed = copy.deepcopy(races)
    last = changed[-1]
    flipped = make_result(make_card(last["date"], "ST", {1: "12:30"})["races"][0],
                          [x["no"] for x in reversed(sorted(last["runners"], key=lambda x: x["no"]))])
    flipped["distance"], flipped["class"] = last["distance"], last["class"]
    changed[-1] = flipped
    assert np.array_equal(features_of_last(changed), base)


def test_the_factors_on_trial_and_their_dates():
    assert model.TRIAL_FEATURES == {"jockey_up": "2026-11-08", "awt_pref": "2026-11-08", "course_adj": "2026-11-15"}
    assert model.ALL_FEATURES == model.BASE_FEATURES + ["jockey_up", "awt_pref", "course_adj"]
    assert "course_adj" not in model.BASE_FEATURES     # not used by the live model unless it passes
    from hkjc.analyze import FEATURE_ZH
    assert all(f in FEATURE_ZH for f in model.ALL_FEATURES)


def test_training_needs_enough_history(sandbox):
    state, params, report = model.train(history(meetings=6))
    assert params is None and report["note"] == "not enough history"
    assert isinstance(state, model.State)
