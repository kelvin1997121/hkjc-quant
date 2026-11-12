"""The live record of frozen first choices beside the final favourite (hkjc/__main__.py)."""
from helpers import analysis_doc, analysis_race, make_card, make_result
from hkjc import __main__ as m, store


def save_meeting(day, venue, races):
    """races: [(finishing order, final odds, pick order, post_hoc)]"""
    card = make_card(day, venue, {i: "12:30" for i in range(1, len(races) + 1)})
    analyses, results = [], []
    for race, (finish, odds, ranking, post_hoc) in zip(card["races"], races):
        res = make_result(race, finish, odds=odds)
        a = analysis_race(race, ranking, post_hoc=post_hoc)
        a["result"] = m._result_summary(res)
        analyses.append(a)
        results.append(res)
    store.write_json(store.DOCS / "data" / f"{day}_{venue}.json", analysis_doc(day, venue, analyses))
    store.save_results(results)


def record():
    m.update_track_record()
    return store.read_json(store.DOCS / "data" / "track_record.json")


NOS = list(range(1, 9))
FLAT = {n: 10.0 + n for n in NOS}


def test_favourites_are_the_lowest_final_odds_among_starters():
    runners = [{"no": 1, "win_odds": 3.0, "place": 2, "place_raw": "2"},
               {"no": 2, "win_odds": 3.0, "place": 1, "place_raw": "1"},
               {"no": 3, "win_odds": 8.0, "place": 3, "place_raw": "3"},
               {"no": 4, "win_odds": 1.5, "place": None, "place_raw": "WV"}]     # withdrawn: not a favourite
    assert m._favourites({"runners": runners}) == [1, 2]
    assert m._favourites({}) == []


def test_paired_z_is_zero_when_the_two_never_differ():
    assert m._paired_z([0, 0, 0, 0]) == 0.0
    assert m._paired_z([1]) == 0.0
    assert m._paired_z([1, 1, 0, 1, 1, 1]) > 0
    assert m._paired_z([-1, -1, 0, -1, -1, -1]) == -m._paired_z([1, 1, 0, 1, 1, 1])


def test_record_scores_the_pick_and_the_final_favourite_of_the_same_race(sandbox):
    save_meeting("2026-11-20", "ST", [
        ([1, 2, 3, 4, 5, 6, 7, 8], {**FLAT, 5: 2.0}, [1, 2, 3, 4, 5, 6, 7, 8], False),   # pick wins, favourite (5) unplaced
        ([4, 5, 6, 1, 2, 3, 7, 8], {**FLAT, 4: 2.0}, [1, 2, 3, 4, 5, 6, 7, 8], False),   # favourite (4) wins, pick 4th
        ([2, 1, 3, 4, 5, 6, 7, 8], {**FLAT, 1: 2.0}, [1, 2, 3, 4, 5, 6, 7, 8], False),   # pick = favourite, placed 2nd
    ])
    tr = record()
    assert (tr["races"], tr["wins"], tr["places"]) == (3, 1, 2)
    assert (tr["fav_wins"], tr["fav_places"]) == (1.0, 2.0)
    assert [r["favourite"] for r in tr["rows"]] == [[5], [4], [1]]
    assert tr["rows"][0]["return"] == FLAT[1] and tr["rows"][1]["return"] == 0
    assert tr["flat_roi"] == round(FLAT[1] / 3 - 1, 4)
    assert tr["z_win_vs_fav"] == 0.0                 # +1, -1, 0: no difference on average


def test_joint_favourites_share_the_race(sandbox):
    save_meeting("2026-11-20", "ST", [([1, 2, 3, 4, 5, 6, 7, 8], {**FLAT, 1: 2.0, 4: 2.0}, NOS, False)])
    row = record()["rows"][0]
    assert row["favourite"] == [1, 4]
    assert (row["fav_won"], row["fav_placed"]) == (0.5, 0.5)


def test_meetings_before_the_freeze_and_post_hoc_races_are_left_out(sandbox):
    save_meeting("2026-11-08", "ST", [(NOS, FLAT, NOS, False)])          # re-analysed after the meeting
    save_meeting("2026-11-20", "ST", [(NOS, FLAT, NOS, True), (NOS, FLAT, NOS, False)])
    tr = record()
    assert tr["from"] == m.LIVE_FROM == "2026-11-11"
    assert tr["races"] == 1
    assert (tr["rows"][0]["date"], tr["rows"][0]["race_no"]) == ("2026-11-20", 2)


def test_empty_record(sandbox):
    tr = record()
    assert tr["races"] == 0 and tr["rows"] == [] and tr["flat_roi"] is None and tr["z_win_vs_fav"] is None
