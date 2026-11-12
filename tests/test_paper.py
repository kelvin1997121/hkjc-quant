"""模擬倉 (hkjc/paper.py): which bets each strategy places, how they are settled, when it switches."""
import pytest

from helpers import analysis_doc, analysis_race, hkt, make_card, make_result
from hkjc import paper, pools, store

DAY, VENUE = "2026-12-02", "HV"


def save(card, analyses, results):
    store.save_card(card)
    store.write_json(store.DOCS / "data" / f"{DAY}_{VENUE}.json", analysis_doc(DAY, VENUE, analyses))
    if results:
        store.save_results(results)


def book(doc, key):
    return next(b for b in doc["books"] if b["key"] == key)


def race_row(doc, no):
    return next(r for r in doc["meetings"][0]["races"] if r["race_no"] == no)


# ------------------------------------------------------------------ settling one bet

RESULT = {"runners": [{"no": 1, "place": 1, "place_raw": "1"}, {"no": 2, "place": 2, "place_raw": "2"},
                      {"no": 3, "place": 3, "place_raw": "3"}, {"no": 4, "place": None, "place_raw": "WV-A"},
                      {"no": 5, "place": None, "place_raw": "PU"}, {"no": 6, "place": 4, "place_raw": "4"}],
          "dividends": {"獨贏": [["1", 35.0]], "位置": [["1", 15.0], ["2", 22.0], ["3", 40.0]],
                        "連贏": [["1,2", 88.0]], "位置Q": [["1,2", 30.0], ["1,3", 55.0], ["2,3", 70.0]]}}
CARD_RACE = {"runners": [{"no": n, "scratched": False} for n in range(1, 8)]}   # horse 7 is not in the result


def test_withdrawn_horses_are_found_from_the_result_and_the_card():
    assert paper._withdrawn(RESULT, CARD_RACE) == {4, 7}        # code WV-A; missing from the result
    marked = {"runners": [{"no": n, "scratched": n == 5} for n in range(1, 7)]}
    assert 5 in paper._withdrawn(RESULT, marked)                # the card said scratched and it never placed
    assert paper._withdrawn(RESULT, None) == {4}                # no card: the result alone


def test_a_bet_on_a_withdrawn_horse_is_refunded():
    out = paper._withdrawn(RESULT, CARD_RACE)
    assert paper.settle("pla", 4, RESULT, out) == (10.0, True)
    assert paper.settle("qin", (1, 4), RESULT, out) == (10.0, True)
    assert paper.settle("qpl", (7, 2), RESULT, out) == (10.0, True)


def test_a_bet_on_a_horse_that_ran_wins_or_loses():
    out = paper._withdrawn(RESULT, CARD_RACE)
    assert paper.settle("pla", 3, RESULT, out) == (40.0, False)
    assert paper.settle("qin", (2, 1), RESULT, out) == (88.0, False)
    assert paper.settle("pla", 5, RESULT, out) == (0.0, False)   # pulled up: it started, so the bet lost
    assert paper.settle("qin", (1, 3), RESULT, out) == (0.0, False)


def test_a_bet_in_a_pool_that_was_not_paid_is_refunded():
    unpaid = dict(RESULT, dividends={**RESULT["dividends"], "連贏": [["-", None]]})
    assert paper.settle("qin", (1, 2), unpaid, set()) == (10.0, True)
    assert paper.settle("pla", 1, unpaid, set()) == (15.0, False)
    no_qpl = dict(RESULT, dividends={k: v for k, v in RESULT["dividends"].items() if k != "位置Q"})
    assert paper.settle("qpl", (1, 2), no_qpl, set()) == (10.0, True)


# ------------------------------------------------------------------ choosing the bets

def test_first_three_strategies_follow_the_frozen_ranking():
    card = make_card(DAY, VENUE, {1: "18:40"})
    race = analysis_race(card["races"][0], order=[5, 2, 7, 1, 3, 4, 6, 8], longshots=(7, 3))
    picks = paper.pick_bets(race, None)
    assert [(p, s) for p, s, _, _ in picks["fav_place"]] == [("pla", 5)]
    assert [(p, s) for p, s, _, _ in picks["top2_qin"]] == [("qin", (5, 2))]
    assert [(p, s) for p, s, _, _ in picks["shot_place"]] == [("pla", 7), ("pla", 3)]
    assert picks["value"] == []                                  # no odds snapshot, no value bets


def test_value_strategy_takes_the_most_likely_of_the_value_selections():
    win = {1: 2.5, 2: 4.0, 3: 6.0, 4: 8.0, 5: 12.0, 6: 20.0, 7: 30.0}
    ch = pools.chances(win)
    # chance x odds: horse 1 = 1.3, horse 2 = 1.25, horse 7 = 3.0 (the biggest "value"), others 0.5
    value = {1: 1.3, 2: 1.25, 7: 3.0}
    shot = {"win": {str(n): o for n, o in win.items()},
            "place": {str(n): value.get(n, 0.5) / ch["pla"][n] for n in win}}
    race = {"runners": [{"no": n, "horse": str(n), "final_rank": n} for n in win]}
    picks = paper.pick_bets(race, shot)["value"]
    assert [(pool, sel) for pool, sel, _, _ in picks] == [("pla", 1), ("pla", 2)]
    assert len(picks) == paper.VALUE_MAX


def test_value_strategy_needs_seven_priced_runners():
    win = {n: 3.0 + n for n in range(1, 7)}
    shot = {"win": {str(n): o for n, o in win.items()}, "place": {str(n): 99.0 for n in win}}
    race = {"runners": [{"no": n, "horse": str(n), "final_rank": n} for n in win]}
    assert paper.pick_bets(race, shot)["value"] == []


# ------------------------------------------------------------------ the whole account

def test_account_over_a_meeting_with_every_kind_of_race(sandbox):
    card = make_card(DAY, VENUE, {1: "18:40", 2: "19:10", 3: "19:40", 4: "20:10", 5: "20:40", 6: "21:10"})
    by_no = {r["race_no"]: r for r in card["races"]}
    order = list(range(1, 9))
    save(card,
         [analysis_race(by_no[n], order, post_hoc=(n == 5)) for n in by_no],
         [make_result(by_no[1], order),                                          # 1 wins, 2 second
          make_result(by_no[2], [2, 3, 4, 5, 6, 7, 8], codes={1: "WV"}),         # first choice withdrawn
          make_result(by_no[5], order),                                          # analysed after the off
          make_result(by_no[6], order, dividends=None)])                         # result, no dividends
    doc = paper.build(now=hkt(DAY, "19:45:00"))

    assert [race_row(doc, n)["state"] for n in range(1, 7)] == \
        ["settled", "settled", "placed", "planned", "post_hoc", "void"]
    assert doc["races"] == 2 and doc["next_look"] == 300

    fav = book(doc, "fav_place")       # race 1 paid $15, race 2 refunded, races 3 and 4 not settled
    assert (fav["bets"], fav["hits"], fav["pending"]) == (4, 1, 2)
    assert (fav["staked"], fav["settled_staked"], fav["returned"], fav["net"]) == (40.0, 20.0, 25.0, 5.0)
    assert fav["balance"] == 1000 - 40 + 25
    assert fav["roi"] == 0.25

    qin = book(doc, "top2_qin")        # race 1 paid $88, race 2 refunded (one of the pair withdrawn)
    assert (qin["bets"], qin["hits"], qin["returned"], qin["net"]) == (4, 1, 98.0, 78.0)

    main = doc["main"]                 # follows 首選+次選連贏 from the start
    assert main["following"] == "top2_qin"
    assert (main["balance"], main["net"], main["bets"], main["hits"]) == (qin["balance"], 78.0, 4, 1)

    r1, r2 = race_row(doc, 1), race_row(doc, 2)
    assert r1["result"] == [1, 2, 3]
    assert (r1["main"][0]["sel"], r1["main"][0]["back"], r1["main"][0]["refund"]) == ("1-2", 88.0, False)
    assert (r2["main"][0]["back"], r2["main"][0]["refund"]) == (10.0, True)
    assert race_row(doc, 5)["main"] == []                                         # no bets after the off
    assert all(not b["placed"] for b in race_row(doc, 6)["main"])                 # void: nothing staked


def meeting_of_six(place_backs):
    """Six settled races: the first choice (1) runs second and its place bet pays place_backs[i]
    (0 = unplaced that day); the first two choices never fill the quinella."""
    card = make_card(DAY, VENUE, {n: f"18:{n:02d}" for n in range(1, 8)})
    by_no = {r["race_no"]: r for r in card["races"]}
    order = list(range(1, 9))
    results = []
    for n, back in zip(range(1, 7), place_backs):
        finish = [3, 1, 4, 2, 5, 6, 7, 8] if back else [3, 4, 5, 1, 2, 6, 7, 8]
        placed = [["3", 20.0], ["1", back], ["4", 30.0]] if back else [["3", 20.0], ["4", 30.0], ["5", 40.0]]
        results.append(make_result(by_no[n], finish, dividends={
            "獨贏": [["3", 60.0]], "位置": placed, "連贏": [["1,3" if back else "3,4", 50.0]],
            "位置Q": [["1,3", 20.0], ["3,4", 25.0], ["1,4", 30.0]]}))
    save(card, [analysis_race(by_no[n], order) for n in by_no], results)


def test_main_account_switches_only_when_another_strategy_is_clearly_better(sandbox, monkeypatch):
    monkeypatch.setattr(paper, "LOOK_EVERY", 6)
    meeting_of_six([22.0, 23.0, 24.0, 25.0, 26.0, 27.0])       # 首選位置 wins every race, the quinella never
    doc = paper.build(now=hkt(DAY, "18:30:00"))
    look = doc["main"]["looks"][0]
    assert look["at"] == 6 and look["switched_to"] == "fav_place"
    assert doc["main"]["following"] == "fav_place"
    assert doc["main"]["switches"][0]["from"] == "top2_qin" and doc["main"]["switches"][0]["z"] >= paper.Z_PASS
    # the six races it had already bet stay as they were; race 7 is bet the new way
    assert [race_row(doc, n)["following"] for n in (6, 7)] == ["top2_qin", "fav_place"]
    assert race_row(doc, 7)["main"][0]["pool"] == "位置"


def test_main_account_does_not_switch_on_one_lucky_race(sandbox, monkeypatch):
    monkeypatch.setattr(paper, "LOOK_EVERY", 6)
    meeting_of_six([150.0, 0, 0, 0, 0, 0])                      # better on average, far from z 2.5
    doc = paper.build(now=hkt(DAY, "18:30:00"))
    look = doc["main"]["looks"][0]
    compare = {c["strategy"]: c for c in look["compare"]}
    assert compare["fav_place"]["diff_per_race"] > 0 and compare["fav_place"]["z"] < paper.Z_PASS
    assert look["switched_to"] is None and doc["main"]["following"] == "top2_qin"
    assert doc["main"]["switches"] == []


def test_a_book_stops_betting_when_it_cannot_cover_a_stake(sandbox, monkeypatch):
    monkeypatch.setattr(paper, "BANKROLL", 25.0)
    meeting_of_six([0, 0, 0, 0, 0, 0])
    doc = paper.build(now=hkt(DAY, "18:30:00"))
    fav = book(doc, "fav_place")
    assert fav["bets"] == 2 and fav["balance"] == 5.0           # $25: two $10 bets, then not enough


def test_meetings_before_the_start_are_ignored(sandbox):
    card = make_card("2026-11-08", "ST", {1: "12:30"})
    store.save_card(card)
    store.write_json(store.DOCS / "data" / "2026-11-08_ST.json",
                     analysis_doc("2026-11-08", "ST", [analysis_race(card["races"][0], list(range(1, 9)))]))
    store.save_results([make_result(card["races"][0], list(range(1, 9)))])
    doc = paper.build(now=hkt(DAY, "12:00:00"))
    assert doc["races"] == 0 and doc["meetings"] == []
    assert book(doc, "fav_place")["balance"] == pytest.approx(1000.0)
