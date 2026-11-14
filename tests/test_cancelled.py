"""A race HKJC cancels: every single-race pool refunded, and no result, ever.

It first happened to the system on 14 Nov 2026 (Sha Tin race 6, declared cancelled about 13
minutes before its post time). The system kept waiting for a result: the race stayed open on
the page, its paper bets stayed out, every update asked for the result again and tried to read
the closed odds page, and the cloud clock stayed on its 2-minute pace to the end of the meeting.

Here: recognising HKJC's notice, keeping the record apart from the results, the paper account
and the pool test. The race-day updates are in test_live_updates.py, the watch in
test_watch.py, the page in test_page.py.
"""
from datetime import date

import pytest
from bs4 import BeautifulSoup

from conftest import FIXTURES
from helpers import analysis_doc, analysis_race, hkt, make_card, make_result, odds_read, utc_iso
from hkjc import paper, pools, scraper, store

PAGES = FIXTURES / "pages"
D11 = date(2026, 11, 11)                      # the made-up meeting of the page fixtures: HV, race 2 cancelled
DAY, VENUE = "2026-12-02", "HV"
POSTS = {1: "18:40", 2: "19:10", 3: "19:40"}
ORDER = list(range(1, 9))


def html(name):
    return (PAGES / name).read_text(encoding="utf-8")


def soup(name="cancelled.html", swap=()):
    text = html(name)
    for old, new in swap:
        assert old in text
        text = text.replace(old, new)
    return BeautifulSoup(text, "html.parser")


# ------------------------------------------------------------------ HKJC's notice

def test_the_notice_page_holds_no_result_and_is_recognised():
    page = soup()
    assert scraper._parse_result_race(page, "HV", D11, 2) is None
    assert scraper._cancelled_race(page, "HV", D11, 2) is True


@pytest.mark.parametrize("venue, day, race_no", [
    ("HV", D11, 1), ("HV", D11, 3),            # another race of the meeting
    ("ST", D11, 2),                            # another course
    ("HV", date(2026, 11, 14), 2),             # another meeting's page served for the same number
])
def test_the_notice_counts_only_for_the_race_asked_for(venue, day, race_no):
    assert scraper._cancelled_race(soup(), venue, day, race_no) is False


WIN_REFUND = '<td>獨贏</td><td class="f_fs14">-</td><td class="f_fs14 f_tar">退款</td>'


@pytest.mark.parametrize("swap", [
    [("此場賽事宣佈取消", "此場賽事現正進行研訊")],                                              # no notice
    [(WIN_REFUND, '<td>獨贏</td><td class="f_fs14">4</td><td class="f_fs14 f_tar">52.50</td>')],   # the win pool paid
    [("racecard_rt_2_o.gif", "racecard_rt_2.gif")],                                              # no race on show
    [("Racecourse=HV", "Racecourse=S1")],                                                        # a simulcast's row
])
def test_a_page_missing_any_one_sign_is_not_taken_as_cancelled(swap):
    assert scraper._cancelled_race(soup(swap=swap), "HV", D11, 2) is False


def test_a_page_with_a_result_is_not_a_cancelled_race():
    assert scraper._cancelled_race(soup("result.html"), "HV", D11, 1) is False


def test_the_refunded_pools_read_as_not_paid():
    div = scraper._parse_dividends(soup())
    for pool in ("獨贏", "位置", "連贏", "位置Q"):
        assert div[pool] == [["-", None]]
    assert div["第一口孖寶"] == [["5/F", 321.0]]          # a pool over several races is still paid
    assert not any(pools.paid_out(div, pool) for pool in pools.POOLS)


def test_fetching_a_meeting_tells_the_cancelled_race_apart(monkeypatch):
    pages = {1: html("result.html"), 2: html("cancelled.html")}
    monkeypatch.setattr(scraper, "get", lambda path, **params: pages[params["RaceNo"]])
    void = set()
    races = scraper.fetch_meeting_results(D11, with_sectionals=False, venue="HV", race_nos=[1, 2], void=void)
    assert [r["race_no"] for r in races] == [1] and void == {2}
    # a caller that does not ask gets the results alone, as before
    assert [r["race_no"] for r in scraper.fetch_meeting_results(
        D11, with_sectionals=False, venue="HV", race_nos=[1, 2])] == [1]


# ------------------------------------------------------------------ the record

def test_cancelled_races_are_kept_per_meeting(sandbox):
    assert store.cancelled(DAY, VENUE) == set()
    assert store.set_cancelled(DAY, VENUE, {3, 2}) is True
    assert store.set_cancelled(DAY, VENUE, [2, 3]) is False                    # nothing new: no rewrite
    assert store.set_cancelled("2026-11-14", "ST", {6}) is True
    assert store.read_json(store.CANCELLED) == {"2026-11-14_ST": [6], f"{DAY}_{VENUE}": [2, 3]}
    assert store.cancelled(DAY, VENUE) == {2, 3} and store.cancelled(DAY, "ST") == set()
    assert store.set_cancelled(DAY, VENUE, set()) is True
    assert store.read_json(store.CANCELLED) == {"2026-11-14_ST": [6]}


def test_the_record_is_not_among_the_results(sandbox):
    store.set_cancelled(DAY, VENUE, {2})
    assert store.load_all_results() == [] and not store.have_results(DAY)


# ------------------------------------------------------------------ the paper account and the pool test

@pytest.fixture
def meeting(sandbox):
    """Three races analysed and bet on before the off; races 1 and 3 were run, race 2 has no
    result. Whether race 2 is known to be cancelled is up to each test."""
    card = make_card(DAY, VENUE, POSTS)
    races = {r["race_no"]: r for r in card["races"]}
    store.save_card(card)
    win = {n: (1.9 if n == 1 else 6.0 + n) for n in ORDER}
    store.write_json(store.DOCS / "data" / f"{DAY}_{VENUE}.json",
                     analysis_doc(DAY, VENUE, [analysis_race(races[n], ORDER, win_odds=win) for n in POSTS]))
    store.save_results([make_result(races[1], ORDER), make_result(races[3], ORDER)])
    store.write_json(store.DATA / "odds" / f"{DAY}_{VENUE}.json", {"latest": {}, "history": {}, "pools": {
        str(n): {"pre": odds_read(races[n], 1, utc_iso(DAY, POSTS[n][:3] + f"{int(POSTS[n][3:]) - 2:02d}:00"))}
        for n in POSTS}})
    return races


AFTER = hkt("2026-12-03", "12:00:00")


def rows(doc):
    return {r["race_no"]: r for r in doc["meetings"][0]["races"]}


def test_until_the_race_is_known_to_be_cancelled_its_bets_stay_out(meeting):
    doc = paper.build(now=AFTER)
    assert rows(doc)[2]["state"] == "placed" and doc["main"]["pending"] == 1
    assert doc["main"]["staked"] == 30.0


def test_nothing_is_bet_on_a_cancelled_race(meeting):
    store.set_cancelled(DAY, VENUE, {2})
    doc = paper.build(now=AFTER)
    row = rows(doc)[2]
    assert row["state"] == "void" and row["cancelled"] is True
    assert row["main"] and all(b["placed"] is False and b["void"] is True and b["back"] is None for b in row["main"])
    assert all(b["placed"] is False for bets in row["books"].values() for b in bets)
    assert doc["main"]["pending"] == 0 and doc["main"]["bets"] == 2 and doc["main"]["staked"] == 20.0
    assert all(b["pending"] == 0 for b in doc["books"])
    assert doc["races"] == 2                                                   # not one of the 300
    assert "cancelled" not in rows(doc)[1] and rows(doc)[1]["state"] == "settled"


def test_the_account_ends_where_it_would_without_that_race(meeting):
    store.set_cancelled(DAY, VENUE, {2})
    with_void = paper.build(now=AFTER)
    path = store.DOCS / "data" / f"{DAY}_{VENUE}.json"
    doc = store.read_json(path)
    doc["races"] = [r for r in doc["races"] if r["race_no"] != 2]
    store.write_json(path, doc)
    without = paper.build(now=AFTER)
    assert with_void["main"] == without["main"] and with_void["books"] == without["books"]


def test_a_result_wins_over_a_stale_cancelled_mark(meeting):
    store.set_cancelled(DAY, VENUE, {1})                                       # race 1 has a result
    doc = paper.build(now=AFTER)
    assert rows(doc)[1]["state"] == "settled" and "cancelled" not in rows(doc)[1]


def test_the_pool_test_holds_no_bet_on_a_cancelled_race(meeting):
    store.set_cancelled(DAY, VENUE, {2})
    assert pools.status()["races"] == 2                                        # races 1 and 3 only
