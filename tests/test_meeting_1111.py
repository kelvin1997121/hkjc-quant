"""The 11 Nov 2026 Happy Valley meeting, replayed from the files saved that night.

It is the first meeting whose picks were frozen at each post time and the first in the paper
account and the pool test. The numbers below were settled on real dividends; a change to the
code must not move them. If one of these fails, either the change is wrong or it rewrites
history that the registered rules say is fixed.
"""
from datetime import datetime, timedelta, timezone

from hkjc import __main__ as m, livetest, paper, pools, store

NOW = datetime(2026, 11, 12, 12, 0, tzinfo=timezone(timedelta(hours=8)))


def test_paper_account(meeting_1111):
    doc = paper.build(now=NOW)
    assert doc["races"] == 9
    books = {b["key"]: (b["balance"], b["net"], b["bets"], b["hits"], b["pending"]) for b in doc["books"]}
    assert books == {"fav_place": (977.5, -22.5, 9, 4, 0),
                     "top2_qin": (910.0, -90.0, 9, 0, 0),
                     "shot_place": (1113.0, 113.0, 14, 8, 0),
                     "value": (885.0, -115.0, 18, 1, 0)}
    main = doc["main"]
    assert (main["following"], main["balance"], main["net"], main["bets"], main["hits"]) == \
        ("top2_qin", 910.0, -90.0, 9, 0)
    assert main["switches"] == [] and main["looks"] == []


def test_paper_account_bets(meeting_1111):
    races = paper.build(now=NOW)["meetings"][0]["races"]
    assert [r["state"] for r in races] == ["settled"] * 9
    assert [r["main"][0]["sel"] for r in races] == ["6-12", "4-3", "11-7", "6-8", "10-1", "4-7", "5-1", "2-3", "5-1"]
    value = [(b["pool"], b["sel"], b["odds"], b["back"]) for b in races[0]["books"]["value"]]
    assert value == [("位置Q", "6-11", 10.0, 65.0), ("位置Q", "6-9", 18.0, 0.0)]
    assert not any(b["refund"] for r in races for bets in r["books"].values() for b in bets)


def test_track_record(meeting_1111):
    m.update_track_record()
    tr = store.read_json(store.DOCS / "data" / "track_record.json")
    assert (tr["races"], tr["wins"], tr["places"]) == (9, 1, 4)
    assert (tr["fav_wins"], tr["fav_places"]) == (1.0, 4.0)
    assert (tr["z_win_vs_fav"], tr["z_place_vs_fav"], tr["flat_roi"]) == (0.0, 0.0, -0.5667)
    assert [(r["pick"], r["winner"], r["favourite"]) for r in tr["rows"]] == [
        (6, 5, [6]), (4, 7, [10]), (11, 3, [11]), (6, 11, [6]), (10, 8, [10]),
        (4, 11, [4]), (5, 5, [5]), (2, 1, [12]), (5, 3, [12])]


def test_pool_test(meeting_1111):
    st = pools.status()
    assert st["races"] == 9
    assert all(p["races"] == 9 and p["next"] == 300 and "verdict" not in p for p in st["pools"].values())
    odds = store.read_json(store.DATA / "odds" / "2026-11-11_HV.json")["pools"]
    results = {r["race_no"]: r for r in store.read_json(store.result_path("2026-11-11", "HV"))}
    total = {pool: [0, 0.0, 0, 0.0] for pool in pools.POOLS}      # staked and paid, above and below 1.20
    for k, book in odds.items():
        for pool, v in pools.race_bets(book["pre"], results[int(k)]).items():
            for i, x in enumerate((*v["above"], *v["below"])):
                total[pool][i] += x
    assert total == {"pla": [10, 0.0, 1060, 913.0], "qin": [890, 0.0, 4940, 2881.5], "qpl": [860, 307.5, 4970, 4770.5]}


def test_new_information_test_can_read_a_real_meeting(meeting_1111, monkeypatch):
    # this meeting is before the test's start date; moved here only to check the files are read
    assert livetest.build() == []
    monkeypatch.setattr(livetest, "FROM", "2026-11-11")
    samples = livetest.build()
    assert len(samples) == 9
    assert all(s["f"]["steam"] is not None and len(s["f"]["steam"]) == len(s["logq"]) >= 7 for s in samples)
    assert all(s["f"]["fast_work"] is None for s in samples)       # no trackwork file in this fixture
