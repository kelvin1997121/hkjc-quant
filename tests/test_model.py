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


def test_training_needs_enough_history(sandbox):
    state, params, report = model.train(history(meetings=6))
    assert params is None and report["note"] == "not enough history"
    assert isinstance(state, model.State)
