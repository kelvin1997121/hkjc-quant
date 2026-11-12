"""Pool test (hkjc/pools.py): Harville chances, dividends, refunds, and when it gives a verdict."""
import pytest

from helpers import make_card, make_result, odds_read, utc_iso
from hkjc import pools, store

WIN7 = {1: 2.5, 2: 4.0, 3: 6.0, 4: 8.0, 5: 12.0, 6: 20.0, 7: 30.0}
DIVIDENDS = {"獨贏": [["1", 35.0]], "位置": [["1", 15.0], ["2", 22.0], ["3", 40.0]],
             "連贏": [["1,2", 88.0]], "位置Q": [["1,2", 30.0], ["1,3", 55.0], ["2,3", 70.0]]}


def result(codes=None, dividends=DIVIDENDS):
    """Seven horses finish 1, 2, 3... by number, except those given a code instead."""
    codes = codes or {}
    runners, place = [], 0
    for no in range(1, 8):
        if no in codes:
            runners.append({"no": no, "place": None, "place_raw": codes[no]})
        else:
            place += 1
            runners.append({"no": no, "place": place, "place_raw": str(place)})
    return {"runners": runners, "dividends": dividends}


def shot(win=WIN7):
    nos = sorted(win)
    return {"win": {str(n): o for n, o in win.items()}, "place": {str(n): 9.0 for n in nos},
            "qin": {f"{a}-{b}": 99.0 for a in nos for b in nos if a < b},
            "qpl": {f"{a}-{b}": 60.0 for a in nos for b in nos if a < b}}


def bets(v):
    return (v["above"][0] + v["below"][0]) // 10


def back(v):
    return v["above"][1] + v["below"][1]


def test_harville_chances_add_up():
    ch = pools.chances(WIN7)
    assert sum(ch["pla"].values()) == pytest.approx(3)    # three placed horses
    assert sum(ch["qin"].values()) == pytest.approx(1)    # one pair fills the first two
    assert sum(ch["qpl"].values()) == pytest.approx(3)    # three pairs among the first three
    assert max(ch["pla"], key=ch["pla"].get) == 1
    assert max(ch["qin"], key=ch["qin"].get) == (1, 2)
    assert len(ch["qin"]) == len(ch["qpl"]) == 21


def test_harville_matches_a_hand_calculation():
    ch = pools.chances({1: 2.0, 2: 1 / 0.3, 3: 5.0})      # win chances 0.5, 0.3, 0.2
    assert ch["qin"][(1, 2)] == pytest.approx(0.5 * 0.3 / 0.5 + 0.3 * 0.5 / 0.7)
    assert ch["pla"][3] == pytest.approx(1.0)             # three runners: all are placed


def test_chances_accept_the_string_keys_of_a_saved_snapshot():
    assert pools.chances({str(n): o for n, o in WIN7.items()}) == pools.chances(WIN7)


def test_paid_reads_real_dividends_in_either_order():
    assert pools._paid(DIVIDENDS, "pla", 2) == 22.0
    assert pools._paid(DIVIDENDS, "pla", 5) == 0
    assert pools._paid(DIVIDENDS, "qin", (2, 1)) == 88.0
    assert pools._paid(DIVIDENDS, "qpl", (3, 1)) == 55.0
    spaced = {"連贏": [["1, 2", 88.0]]}
    assert pools._paid(spaced, "qin", (1, 2)) == 88.0


def test_paid_out_is_false_for_a_pool_hkjc_did_not_pay():
    assert pools.paid_out(DIVIDENDS, "qin")
    assert not pools.paid_out({**DIVIDENDS, "連贏": [["-", None]]}, "qin")
    assert not pools.paid_out({k: v for k, v in DIVIDENDS.items() if k != "位置Q"}, "qpl")


def test_every_selection_is_a_bet_and_is_paid_the_real_dividend():
    rb = pools.race_bets(shot(), result())
    assert (bets(rb["pla"]), bets(rb["qin"]), bets(rb["qpl"])) == (7, 21, 21)
    assert back(rb["pla"]) == 15.0 + 22.0 + 40.0
    assert back(rb["qin"]) == 88.0
    assert back(rb["qpl"]) == 30.0 + 55.0 + 70.0


def test_selection_is_above_or_below_by_chance_times_shown_odds():
    ch = pools.chances(WIN7)
    s = shot()
    s["place"] = {str(n): (1.3 if n == 1 else 1.1) / ch["pla"][n] for n in WIN7}   # only horse 1 is value
    rb = pools.race_bets(s, result())
    assert rb["pla"]["above"] == (10, 15.0)
    assert rb["pla"]["below"] == (60, 22.0 + 40.0)


def test_a_horse_that_did_not_start_is_no_bet():
    rb = pools.race_bets(shot(), result(codes={4: "WV-A"}))
    assert bets(rb["pla"]) == 6            # not the withdrawn horse
    assert bets(rb["qin"]) == 15           # nor the 6 pairs with it
    assert bets(rb["qpl"]) == 15


def test_a_horse_that_started_but_did_not_finish_is_a_losing_bet():
    rb = pools.race_bets(shot(), result(codes={7: "PU"}))
    assert bets(rb["pla"]) == 7 and bets(rb["qin"]) == 21


def test_a_pool_that_was_not_paid_is_left_out_of_the_race():
    rb = pools.race_bets(shot(), result(dividends={**DIVIDENDS, "連贏": [["-", None]]}))
    assert sorted(rb) == ["pla", "qpl"]


def test_small_fields_and_races_without_dividends_are_skipped():
    six = {n: o for n, o in WIN7.items() if n != 7}
    assert pools.race_bets(shot(six), result()) == {}
    assert pools.race_bets(shot(), result(dividends={})) == {}


def save_meeting(day, races, pre=True):
    card = make_card(day, "ST", {n: "12:30" for n in races})
    store.save_card(card)
    store.save_results([make_result(r, order=[x["no"] for x in r["runners"]]) for r in card["races"]])
    books = {}
    for r in card["races"]:
        snap = odds_read(r, fav=1, read_at=utc_iso(day, "12:25:00"))
        snap.pop("scratched")
        books[str(r["race_no"])] = {"pre": snap} if pre else {"last": snap}
    store.write_json(store.DATA / "odds" / f"{day}_ST.json", {"latest": {}, "history": {}, "pools": books})


def test_status_counts_races_that_have_a_pre_snapshot(sandbox):
    save_meeting("2026-11-20", [1, 2, 3])
    save_meeting("2026-11-21", [1, 2], pre=False)       # odds read, but never before a post time
    save_meeting("2026-11-01", [1, 2])                  # before the test was registered
    st = pools.status()
    assert st["races"] == 3
    assert all(p["races"] == 3 and p["next"] == 300 and "verdict" not in p for p in st["pools"].values())


def test_status_gives_a_verdict_only_at_a_look(sandbox, monkeypatch):
    monkeypatch.setattr(pools, "LOOKS", (2, 4))
    save_meeting("2026-11-20", [1, 2, 3])
    row = pools.status()["pools"]["pla"]
    assert row["look"] == 2 and row["next"] == 4
    assert row["verdict"] in ("過關", "未過關")            # the first look can't close the test
    save_meeting("2026-11-21", [1, 2])
    row = pools.status()["pools"]["pla"]
    assert row["look"] == 4 and row["next"] is None
    assert row["verdict"] in ("過關", "未過關,完結")
