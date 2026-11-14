"""Overhead-picture test (hkjc/trips.py): the two measures, which run counts as a horse's last,
how a race becomes a sample, and the registered rules themselves. Made-up races only."""
import numpy as np
import pytest

from helpers import code, make_card, make_result, odds_read, utc_iso
from hkjc import aerial, livetest, store, trips

L = trips.LENGTH


def row(no, x, y, fit=0.95):
    return [no, float(x), float(y), fit, 1.0]


# ------------------------------------------------------------------ the registered rules

def test_the_registered_rules_are_what_was_written_down():
    assert (trips.REGISTERED, trips.FROM) == ("2026-11-14", "2026-11-15")
    assert (trips.LOOKS, trips.BLOCKS, trips.Z_PASS, trips.PLACEBO_MAX, trips.MIN_PRICED) == ((300, 600), 5, 2.5, 1.0, 7)
    assert trips.CANDIDATES == ["wide_last", "boxed_last"]
    assert (trips.LENGTH, trips.METRE, trips.WIDE_WINDOW, trips.WIDE_CAP, trips.MIN_FIELD) == (41.0, 16.4, 2.0, 8.0, 4)


def test_it_is_judged_exactly_like_the_new_information_test():
    assert (trips.LOOKS, trips.BLOCKS, trips.Z_PASS, trips.PLACEBO_MAX, trips.MIN_PRICED) == \
        (livetest.LOOKS, livetest.BLOCKS, livetest.Z_PASS, livetest.PLACEBO_MAX, livetest.MIN_PRICED)


def test_directions_already_registered_do_not_change():
    # new kinds of race may be added; these ten are fixed
    fixed = {("HV", "TURF", 1000): (352, 10), ("HV", "TURF", 1200): (2, 1), ("HV", "TURF", 1650): (358, 2),
             ("HV", "TURF", 1800): (357, 9), ("ST", "AWT", 1200): (357, 156), ("ST", "AWT", 1650): (354, 165),
             ("ST", "TURF", 1200): (38, 185), ("ST", "TURF", 1400): (31, 183), ("ST", "TURF", 1600): (29, 186),
             ("ST", "TURF", 1800): (44, 181)}
    for kind, angles in fixed.items():
        assert trips.DIRECTION[kind] == angles
    assert ("ST", "TURF", 1000) not in trips.DIRECTION            # the straight course has no turn


# ------------------------------------------------------------------ wide

def test_wide_is_metres_outside_the_innermost_horse_alongside():
    # horses running to the left (180): the inside rail is at the top of the picture
    panel = [row(1, 300, 100), row(2, 305, 100 + 16.4), row(3, 310, 100 + 3 * 16.4), row(4, 290, 100)]
    w = trips.wide(panel, 180)
    assert w[1] == pytest.approx(0) and w[4] == pytest.approx(0)
    assert w[2] == pytest.approx(1.0) and w[3] == pytest.approx(3.0)


def test_which_side_is_inside_follows_the_direction_of_travel():
    panel = [row(1, 300, 100), row(2, 300, 141), row(3, 300, 182), row(4, 300, 223)]
    assert trips.wide(panel, 180)[4] == pytest.approx(7.5) and trips.wide(panel, 180)[1] == pytest.approx(0)
    assert trips.wide(panel, 0)[1] == pytest.approx(7.5) and trips.wide(panel, 0)[4] == pytest.approx(0)   # running right


def test_only_horses_within_two_lengths_count_as_alongside():
    panel = [row(1, 300, 100), row(2, 300, 141),                  # 2 is a length outside 1
             row(3, 300 + 2.5 * L, 200), row(4, 300 + 2.5 * L, 241),   # another pair, 2.5 lengths back
             row(5, 300 + 6 * L, 300)]                            # alone at the back
    w = trips.wide(panel, 180)
    assert w[2] == pytest.approx(2.5) and w[4] == pytest.approx(2.5)
    assert w[3] == pytest.approx(0)                               # not measured against horse 1, far ahead
    assert w[5] == pytest.approx(0)                               # nobody alongside: nothing to be outside of


def test_a_slanted_panel_is_measured_along_its_own_direction():
    a = np.radians(38)
    along, inward = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    at = lambda s, out: np.array([400.0, 200.0]) + s * along - out * 16.4 * inward
    panel = [row(n, *at(s, out)) for n, s, out in ((1, 0, 0), (2, -10, 2), (3, -20, 4), (4, -30, 1))]
    w = trips.wide(panel, 38)
    assert [round(w[n], 2) for n in (1, 2, 3, 4)] == [0.0, 2.0, 4.0, 1.0]


def test_a_field_too_thin_to_judge_gives_nothing():
    assert trips.wide([row(1, 300, 100), row(2, 300, 140), row(3, 300, 180)], 180) == {}


# ------------------------------------------------------------------ boxed

POCKET = [row(1, 400, 200),                    # the horse
          row(2, 400 - 1.5 * L, 200),          # one in front of it (to the left)
          row(3, 400 - 0.2 * L, 200 + 0.8 * L),   # one on its outside (lower in the picture)
          row(4, 400 - 0.5 * L, 200 - 0.8 * L)]   # one on its inside


def test_a_horse_with_others_in_front_and_on_both_sides_is_boxed():
    assert trips.boxed(POCKET)[1] is True
    assert trips.boxed(POCKET)[2] is False                        # the one in front has a clear run


@pytest.mark.parametrize("without", [2, 3])
def test_take_away_the_horse_in_front_or_outside_and_it_is_free(without):
    assert trips.boxed([r for r in POCKET if r[0] != without])[1] is False


def test_the_rail_shuts_the_inside_for_the_innermost_horse():
    no_inside_horse = [r for r in POCKET if r[0] != 4]
    assert trips.boxed(no_inside_horse)[1] is True                # nothing inside it: it is on the rail
    off_the_rail = no_inside_horse + [row(5, 400 + 5 * L, 200 - 2 * L)]   # another horse far behind, well inside
    assert trips.boxed(off_the_rail)[1] is False                  # so there is room inside after all


def test_a_horse_too_far_ahead_or_out_of_line_is_not_in_the_way():
    far = [row(2, 400 - 2.5 * L, 200) if r[0] == 2 else r for r in POCKET]
    aside = [row(2, 400 - 1.5 * L, 200 + 0.7 * L) if r[0] == 2 else r for r in POCKET]
    assert trips.boxed(far)[1] is False and trips.boxed(aside)[1] is False


# ------------------------------------------------------------------ one race, one horse's last run

TURN = [row(1, 300, 100), row(2, 300, 100 + 2 * 16.4), row(3, 320, 100 + 4 * 16.4), row(4, 340, 100)]


def race_of(venue="ST", surface="TURF", distance=1400):
    return {"venue": venue, "surface": surface, "distance": distance}


def test_a_race_gives_each_horse_its_two_measures():
    # 1400 m at Sha Tin: the 800 m panel runs at 31 degrees, the 400 m panel at 183
    m = trips.run_measures(race_of(), [[], TURN, POCKET, []])
    assert m[3]["wide"] == pytest.approx(trips.wide(TURN, 183)[3]) and m[3]["wide"] > 3
    assert m[1]["boxed"] is True and m[2]["boxed"] is False


def test_wide_is_the_average_of_the_two_turn_panels_and_is_capped():
    far_out = [row(1, 300, 100), row(2, 300, 100 + 12 * 16.4), row(3, 300, 110), row(4, 300, 120)]
    m = trips.run_measures(race_of(), [[], far_out, [], []])      # 12 metres out at the 400 m panel
    assert trips.wide(far_out, 183)[2] > 11 and m[2]["wide"] == trips.WIDE_CAP
    both = trips.run_measures(race_of(), [TURN, TURN, [], []])
    assert both[2]["wide"] == pytest.approx((trips.wide(TURN, 31)[2] + trips.wide(TURN, 183)[2]) / 2)


@pytest.mark.parametrize("kind", [("ST", "TURF", 1000), ("ST", "TURF", 2400), ("HV", "TURF", 2200)])
def test_no_wide_for_the_straight_course_or_a_kind_of_race_not_yet_measured(kind):
    m = trips.run_measures(race_of(*kind), [TURN, TURN, POCKET, []])
    assert all(v["wide"] is None for v in m.values())
    assert m[1]["boxed"] is True                                  # the straight is the same everywhere


def test_a_number_read_unsurely_gets_no_measure_but_still_stands_in_the_way():
    turn = [row(2, 300, 100 + 2 * 16.4, fit=0.6) if r[0] == 2 else r for r in TURN]
    pocket = [row(3, r[1], r[2], fit=0.6) if r[0] == 3 else r for r in POCKET]
    m = trips.run_measures(race_of(), [[], turn, pocket, []])
    assert m[2]["wide"] is None and m[3]["wide"] is not None
    assert m[3]["boxed"] is None and m[1]["boxed"] is True        # the unsure circle still boxes horse 1 in


def test_a_horse_not_in_a_panel_has_nothing_from_it():
    m = trips.run_measures(race_of(), [[], TURN, [r for r in POCKET if r[0] != 4], []])
    assert m[4]["boxed"] is None and m[4]["wide"] == pytest.approx(0)


# ------------------------------------------------------------------ the test's samples

EARLIER, DAY, VENUE = "2026-11-08", "2026-11-18", "ST"
ORDER = list(range(1, 9))


def save_meeting(day, order=ORDER, pre=True, distance=1400):
    card = make_card(day, VENUE, {1: "13:00"})
    race = card["races"][0]
    race["distance"] = distance
    result = make_result(race, order)
    result["distance"] = distance
    store.save_results([result])
    if pre:
        store.write_json(store.DATA / "odds" / f"{day}_{VENUE}.json", {"latest": {}, "history": {}, "pools": {
            "1": {"pre": odds_read(race, 1, utc_iso(day, "12:58:00"))}}})
    return race


def save_positions(day, panels):
    aerial._save(aerial.FOLDER / f"{day}_{VENUE}.json",
                 {"date": day, "venue": VENUE, "races": {"1": {"read_at": "x", "ran": 8, "panels": panels}}})


@pytest.fixture
def two_meetings(sandbox):
    """Every horse ran on 8 Nov (with a picture) and again on 18 Nov."""
    save_meeting(EARLIER)
    turn = [row(n, 300 + 10 * n, 100 + 16.4 * (n - 1)) for n in ORDER]        # horse n is n - 1 metres out
    save_positions(EARLIER, [[], turn, POCKET, []])
    save_meeting(DAY)


def test_a_race_becomes_one_sample_with_last_runs_measures(two_meetings):
    (s,) = trips.build()
    assert s["key"] == (DAY, VENUE, 1) and s["y"] == 0
    wide = s["f"]["wide_last"]
    measured = trips.wide([row(n, 300 + 10 * n, 100 + 16.4 * (n - 1)) for n in ORDER], 183)   # 1400 m: 183 degrees
    assert wide.sum() == pytest.approx(0)                         # taken from the runners' average
    assert list(wide) == pytest.approx([measured[n] - np.mean(list(measured.values())) for n in ORDER])
    assert np.all(np.diff(wide) > 0.9)                            # about a metre more for each horse out
    assert list(s["f"]["boxed_last"]) == [1.0, 0, 0, 0, 0, 0, 0, 0]   # horse 1 was in the pocket; 5-8 not in that panel


def test_runners_without_a_value_count_as_average(sandbox):
    save_meeting(EARLIER)
    turn = [row(n, 300 + 10 * n, 100 + 16.4 * (n - 1)) for n in (1, 2, 3, 4, 5)]   # 6, 7, 8 out of the picture
    save_positions(EARLIER, [[], turn, [], []])
    save_meeting(DAY)
    (s,) = trips.build()
    measured = trips.wide(turn, 183)
    mean = np.mean(list(measured.values()))
    assert list(s["f"]["wide_last"]) == pytest.approx([measured[n] - mean for n in (1, 2, 3, 4, 5)] + [0, 0, 0])
    assert s["f"]["wide_last"][0] < -1.5 and s["f"]["wide_last"][4] > 1.5
    assert s["f"]["boxed_last"] is None                           # nobody was read at 200 m


def test_a_race_where_fewer_than_half_have_a_value_is_not_used(sandbox):
    save_meeting(EARLIER)
    turn = [row(n, 300 + 10 * n, 100) for n in (1, 2, 3, 4)]
    save_positions(EARLIER, [[], turn, [row(n, 300 + 60 * n, 200) for n in (1, 2, 3)], []])
    save_meeting(DAY)
    (s,) = trips.build()
    assert s["f"]["wide_last"] is not None                        # 4 of 8
    assert s["f"]["boxed_last"] is None                           # 3 of 8


def test_only_the_last_run_before_race_day_counts(sandbox):
    save_meeting("2026-11-01")
    save_positions("2026-11-01", [[], [row(n, 300, 100 + 16.4 * 2 * (n - 1)) for n in ORDER], [], []])
    save_meeting(EARLIER)                                         # a later run, with no picture
    save_meeting(DAY)
    (s,) = trips.build()
    assert s["f"]["wide_last"] is None and s["f"]["boxed_last"] is None


def test_races_before_the_registered_start_are_never_samples(sandbox):
    save_meeting("2026-11-01")
    save_positions("2026-11-01", [[], [row(n, 300, 100 + 16.4 * n) for n in ORDER], [], []])
    save_meeting("2026-11-14")                                    # run before the rules were written
    assert trips.build() == []


def test_a_race_without_pre_odds_or_a_single_winner_is_left_out(sandbox):
    save_meeting(EARLIER)
    save_meeting(DAY, pre=False)
    assert trips.build() == []


def test_status_counts_the_races_and_waits_for_the_first_look(two_meetings):
    st = trips.status()
    assert (st["registered"], st["from"]) == ("2026-11-14", "2026-11-15")
    for name in trips.CANDIDATES:
        assert st["candidates"][name]["races"] == 1 and st["candidates"][name]["next"] == 300
        assert "verdict" not in st["candidates"][name]


def test_the_verdict_comes_from_the_new_information_tests_judge(monkeypatch):
    rng = np.random.default_rng(3)
    samples = [{"key": i, "y": 0, "logq": np.log(np.full(8, 1 / 8)),
                "f": {"wide_last": rng.normal(size=8), "boxed_last": None}} for i in range(300)]
    monkeypatch.setattr(trips, "build", lambda races=None: samples)
    calls = []

    def judge(use, cand, placebo=False):
        calls.append((len(use), cand, placebo))
        return {"gain": 0.01, "z": 0.4 if placebo else 3.1}

    monkeypatch.setattr(livetest, "judge", judge)
    st = trips.status()["candidates"]
    assert calls == [(300, "wide_last", False), (300, "wide_last", True)]
    assert st["wide_last"]["verdict"] == "過關" and st["wide_last"]["passed"] is True
    assert st["boxed_last"]["races"] == 0 and "verdict" not in st["boxed_last"]


def test_meeting_table_lists_every_horse_read(two_meetings):
    rows = trips.meeting_table(EARLIER, VENUE)
    assert len(rows) == 8 and rows[0]["race_no"] == 1
    assert {r["no"]: r["boxed"] for r in rows}[1] is True
    assert code(1, 1) and rows[0]["horse"]
