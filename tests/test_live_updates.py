"""Race-day updates (hkjc/__main__.py): odds snapshots, freezing a pick at its post time,
and getting a result onto the page. HKJC, the odds pages, the clock and the model are replaced."""
import copy
from types import SimpleNamespace

import pytest

from helpers import hkt, make_card, make_result, odds_read, utc_iso
from hkjc import __main__ as m, model, odds as odds_mod, reports, store

DAY, VENUE = "2026-11-14", "ST"
POSTS = {1: "12:30", 2: "13:00", 3: "13:55"}


@pytest.fixture
def raceday(sandbox, monkeypatch):
    card = make_card(DAY, VENUE, POSTS)
    world = SimpleNamespace(card=card, races={r["race_no"]: r for r in card["races"]},
                            now=hkt(DAY, "08:00:00"), reads={}, stamp=None, finished=set(), blocked=False)

    def fetch_odds(date_iso, venue, race_nos, log=print):
        if world.blocked:
            raise RuntimeError("odds pages blocked")
        out = {n: copy.deepcopy(world.reads[n]) for n in race_nos if n in world.reads}
        out["_fetched_at"] = world.stamp
        return out

    def train(races, cfg=None):    # a model that knows nothing: the ranking is the market's
        k = len(model.FEATURES)
        blend = {"a": 0.0, "b": 1.0}
        return model.State(), {"w": [0.0] * k, "mu": [0.0] * k, "sd": [1.0] * k,
                               "features": model.FEATURES, "blend": blend}, {"blend": blend}

    monkeypatch.setattr(odds_mod, "fetch_odds", fetch_odds)
    monkeypatch.setattr(m, "today_hkt", lambda: world.now.date())
    monkeypatch.setattr(m, "now_hkt", lambda: world.now)
    monkeypatch.setattr(m, "backfill", lambda *a, **k: 0)
    monkeypatch.setattr(m, "complete_recent_results", lambda *a, **k: None)
    monkeypatch.setattr(reports, "refresh_recent", lambda *a, **k: None)
    monkeypatch.setattr(m.scraper, "fetch_racecard", lambda *a, **k: copy.deepcopy(card))
    monkeypatch.setattr(m, "live_results", lambda c, race_nos=None: set(world.finished))
    monkeypatch.setattr(model, "train", train)
    return world


def set_reads(world, clock, reads, stamp, day=DAY):
    """reads: {race_no: (clock it was read at, favourite's number)}; stamp: when the batch ended."""
    world.now = hkt(day, clock)
    world.reads = {n: odds_read(world.races[n], fav, utc_iso(DAY, at)) for n, (at, fav) in reads.items()}
    world.stamp = utc_iso(DAY, stamp)


def update(world, clock, reads, stamp):
    set_reads(world, clock, reads, stamp)
    m.run(use_agent=False)


def analysis(n):
    doc = store.read_json(store.DOCS / "data" / f"{DAY}_{VENUE}.json")
    return next(r for r in doc["races"] if r["race_no"] == n)


def pick(n):
    return analysis(n)["runners"][0]["no"]


def odds_file():
    return store.read_json(store.DATA / "odds" / f"{DAY}_{VENUE}.json")


def book(n):
    return odds_file()["pools"][str(n)]


def all_at(clock, fav=1):
    return {n: (clock, fav) for n in POSTS}


# ------------------------------------------------------------------ small pieces

def test_started_races_follow_the_clock():
    card = make_card(DAY, VENUE, POSTS)
    assert m.started_races(card, hkt(DAY, "12:29:59")) == set()
    assert m.started_races(card, hkt(DAY, "12:30:00")) == {1}
    assert m.started_races(card, hkt(DAY, "13:54:00")) == {1, 2}
    assert m.started_races(card, hkt("2026-11-13", "23:00:00")) == set()
    assert m.started_races(card, hkt("2026-11-15", "00:01:00")) == {1, 2, 3}


def test_read_after_post_compares_each_read_with_its_own_post_time():
    card = make_card(DAY, VENUE, POSTS)
    late = m.read_after_post(card, {1: {"t": utc_iso(DAY, "12:29:59")},      # a second before the post
                                    2: {"t": utc_iso(DAY, "13:00:00")},      # on it
                                    3: {}})                                  # an old file: no read time
    assert late == {2}
    late = m.read_after_post(card, {1: {"t": utc_iso("2026-11-13", "23:00:00")},
                                    2: {"t": utc_iso("2026-11-15", "00:30:00")}})
    assert late == {2}


def test_schedule_lists_post_times_and_finished_races():
    s = m.race_schedule(make_card(DAY, VENUE, POSTS), {2, 1})
    assert (s["first_post"], s["last_post"], s["resulted"]) == ("12:30", "13:55", [1, 2])
    assert s["posts"] == {"1": "12:30", "2": "13:00", "3": "13:55"}


def test_write_if_changed_ignores_the_timestamp(sandbox):
    path = sandbox / "docs" / "data" / "x.json"
    assert m.write_if_changed(path, {"updated_at": "a", "v": 1})
    assert not m.write_if_changed(path, {"updated_at": "b", "v": 1})
    assert m.write_if_changed(path, {"updated_at": "c", "v": 2})


# ------------------------------------------------------------------ odds snapshots

def test_pre_snapshot_goes_by_the_races_own_read_time_not_the_batch_end(raceday):
    # race 1 (post 12:30) read at 12:29:59, race 2 (post 13:00) at 13:00:00; the batch ends 13:05
    set_reads(raceday, "12:29:00", {1: ("12:29:59", 4), 2: ("13:00:00", 5), 3: ("13:00:10", 6)}, "13:05:00")
    out = m.live_odds(raceday.card)
    assert book(1)["pre"]["t"] == utc_iso(DAY, "12:29:59")     # kept, though the batch ended after 12:30
    assert "pre" not in book(2)                                # read on the post time: not pre
    assert book(3)["pre"]["t"] == utc_iso(DAY, "13:00:10")
    assert [h["t"] for h in odds_file()["history"]["1"]] == [utc_iso(DAY, "12:29:59")]
    assert out[1]["t"] == utc_iso(DAY, "12:29:59") and out[1]["win"][4] == 1.9
    assert set(book(1)["pre"]) == {"t", "updated", "win", "place", "qin", "qpl"}


def test_a_later_read_after_the_post_time_moves_last_but_not_pre(raceday):
    set_reads(raceday, "12:24:00", all_at("12:24:20", fav=1), "12:25:00")
    m.live_odds(raceday.card)
    set_reads(raceday, "12:31:00", all_at("12:31:20", fav=2), "12:32:00")
    m.live_odds(raceday.card)
    assert book(1)["pre"]["t"] == utc_iso(DAY, "12:24:20") and book(1)["last"]["t"] == utc_iso(DAY, "12:31:20")
    assert book(1)["first"]["t"] == utc_iso(DAY, "12:24:20")
    assert book(2)["pre"]["t"] == utc_iso(DAY, "12:31:20")     # race 2 is still before its post


def test_the_day_before_is_never_a_pre_snapshot(raceday):
    raceday.now = hkt("2026-11-13", "20:00:00")
    raceday.reads = {n: odds_read(raceday.races[n], 1, utc_iso("2026-11-13", "20:00:05")) for n in POSTS}
    raceday.stamp = utc_iso("2026-11-13", "20:00:30")
    out = m.live_odds(raceday.card)
    assert all("pre" not in book(n) and "last" in book(n) for n in POSTS)
    assert out[1]["open"] is None                              # "open" is the first read on race day itself


def test_history_keeps_a_snapshot_only_when_the_odds_moved(raceday):
    set_reads(raceday, "09:00:00", all_at("09:00:10", fav=1), "09:00:30")
    m.live_odds(raceday.card)
    set_reads(raceday, "09:30:00", all_at("09:30:10", fav=1), "09:30:30")     # same prices
    m.live_odds(raceday.card)
    set_reads(raceday, "10:00:00", all_at("10:00:10", fav=3), "10:00:30")
    out = m.live_odds(raceday.card)
    assert [h["t"] for h in odds_file()["history"]["1"]] == [utc_iso(DAY, "09:00:10"), utc_iso(DAY, "10:00:10")]
    assert out[1]["open"]["1"] == 1.9 and out[1]["win"][3] == 1.9             # opened with 1, now 3


def test_no_odds_outside_race_day_and_the_day_before(raceday):
    raceday.now = hkt("2026-11-12", "20:00:00")
    assert m.live_odds(raceday.card) == {}
    assert odds_file() is None


def test_blocked_odds_pages_fall_back_to_what_was_saved(raceday):
    set_reads(raceday, "09:00:00", all_at("09:00:10", fav=2), "09:00:30")
    m.live_odds(raceday.card)
    raceday.blocked = True
    raceday.now = hkt(DAY, "09:30:00")
    out = m.live_odds(raceday.card)
    assert out[1]["win"]["2"] == 1.9


# ------------------------------------------------------------------ freezing the pick

def test_pick_follows_the_odds_until_the_post_time_then_stays(raceday):
    update(raceday, "12:20:00", all_at("12:20:10", fav=1), "12:20:40")
    assert pick(1) == 1
    update(raceday, "12:25:00", all_at("12:25:10", fav=2), "12:25:40")
    assert pick(1) == 2 and pick(2) == 2
    frozen = analysis(1)
    update(raceday, "12:31:00", all_at("12:31:10", fav=3), "12:31:40")        # race 1 is off
    assert analysis(1) == frozen
    assert pick(2) == 3                                                        # race 2 still follows
    assert book(1)["pre"]["t"] == utc_iso(DAY, "12:25:10")
    assert not analysis(1).get("post_hoc")


def test_a_read_just_after_the_post_time_does_not_replace_the_pick(raceday):
    update(raceday, "12:24:00", all_at("12:24:20", fav=1), "12:25:00")
    frozen = analysis(1)
    # the clock said 12:29:50 when the update decided what was off; the page was read at 12:30:05
    update(raceday, "12:29:50", {1: ("12:30:05", 2), 2: ("12:30:12", 5), 3: ("12:30:20", 5)}, "12:31:00")
    assert analysis(1) == frozen and pick(1) == 1
    assert book(1)["pre"]["t"] == utc_iso(DAY, "12:24:20") and book(1)["last"]["t"] == utc_iso(DAY, "12:30:05")
    assert pick(2) == 5 and book(2)["pre"]["t"] == utc_iso(DAY, "12:30:12")


def test_a_read_before_the_post_time_counts_even_if_the_batch_ends_after_it(raceday):
    # 11 Nov 2026 race 2: read at 19:08 for a 19:10 post, batch stamped 19:10:01
    update(raceday, "12:50:00", all_at("12:50:10", fav=1), "12:50:40")
    update(raceday, "12:58:30", {1: ("12:58:45", 1), 2: ("12:58:50", 3), 3: ("12:59:40", 1)}, "13:00:20")
    pre = book(2)["pre"]
    assert pick(2) == 3
    assert pre["t"] == utc_iso(DAY, "12:58:50")
    assert min(pre["win"], key=pre["win"].get) == "3"
    assert all(pre["win"][str(x["no"])] == x["win_odds"] for x in analysis(2)["runners"])   # same odds as the pick


def test_a_race_first_analysed_after_its_post_time_is_marked_and_not_scored(raceday):
    update(raceday, "12:35:00", all_at("12:35:10", fav=1), "12:35:40")
    assert analysis(1)["post_hoc"] is True
    assert not analysis(2).get("post_hoc")
    raceday.finished = {1}
    store.save_results([make_result(raceday.races[1], list(range(1, 9)))])
    update(raceday, "12:45:00", all_at("12:45:10", fav=1), "12:45:40")
    record = store.read_json(store.DOCS / "data" / "track_record.json")
    assert analysis(1)["result"]["placings"] == [1, 2, 3]
    assert record["races"] == 0                                                # shown, but not in the record


def test_a_finished_race_keeps_its_pick_and_enters_the_record(raceday):
    update(raceday, "12:24:00", all_at("12:24:20", fav=2), "12:25:00")
    frozen = analysis(1)["runners"]
    raceday.finished = {1}
    odds = {no: 9.0 + no for no in range(1, 9)}
    odds[3] = 2.0                                                              # final favourite: horse 3
    store.save_results([make_result(raceday.races[1], [2, 3, 1, 4, 5, 6, 7, 8], odds=odds)])
    update(raceday, "12:45:00", all_at("12:45:10", fav=5), "12:45:40")
    assert analysis(1)["runners"] == frozen
    assert analysis(1)["result"]["placings"] == [2, 3, 1]
    record = store.read_json(store.DOCS / "data" / "track_record.json")
    row = record["rows"][0]
    assert (record["races"], record["wins"], record["places"]) == (1, 1, 1)
    assert (row["pick"], row["winner"], row["favourite"], row["fav_won"], row["fav_placed"]) == (2, 2, [3], 0.0, 1.0)


def test_latest_json_is_the_meetings_own_file(raceday):
    update(raceday, "12:24:00", all_at("12:24:20"), "12:25:00")
    data = store.DOCS / "data"
    assert store.read_json(data / "latest.json") == store.read_json(data / f"{DAY}_{VENUE}.json")
    assert store.read_json(data / "schedule.json")["date"] == DAY


# ------------------------------------------------------------------ results onto the page first

def result_arrives(world, monkeypatch, race_no=1):
    def live_results(card, race_nos=None):
        store.write_json(store.result_path(DAY, VENUE), [make_result(world.races[race_no], list(range(1, 9)))])
        return {race_no}
    monkeypatch.setattr(m, "live_results", live_results)


def test_quick_results_put_the_result_in_the_file_the_page_reads(raceday, monkeypatch):
    update(raceday, "12:24:00", all_at("12:24:20"), "12:25:00")
    frozen = analysis(1)["runners"]
    latest = store.DOCS / "data" / "latest.json"
    assert "result" not in store.read_json(latest)["races"][0]
    result_arrives(raceday, monkeypatch)
    raceday.now = hkt(DAY, "12:45:00")
    assert m.quick_results(now=raceday.now) is True
    shown = store.read_json(latest)
    assert shown["races"][0]["result"]["placings"] == [1, 2, 3]
    assert shown["races"][0]["runners"] == frozen                              # the pick is untouched
    assert shown == store.read_json(store.DOCS / "data" / f"{DAY}_{VENUE}.json")
    assert store.read_json(store.DOCS / "data" / "schedule.json")["resulted"] == [1]
    paper = store.read_json(store.DOCS / "data" / "paper.json")
    assert paper["meetings"][0]["races"][0]["state"] == "settled"


def test_quick_results_leave_latest_json_alone_when_it_shows_another_meeting(raceday, monkeypatch):
    update(raceday, "12:24:00", all_at("12:24:20"), "12:25:00")
    other = {"meeting": {"date": "2026-11-18", "venue": "HV"}, "races": []}
    store.write_json(store.DOCS / "data" / "latest.json", other)
    result_arrives(raceday, monkeypatch)
    raceday.now = hkt(DAY, "12:45:00")
    assert m.quick_results(now=raceday.now) is True
    assert store.read_json(store.DOCS / "data" / "latest.json") == other
    assert analysis(1)["result"]["placings"] == [1, 2, 3]


def test_quick_results_ask_only_for_races_that_are_due(raceday, monkeypatch):
    update(raceday, "12:24:00", all_at("12:24:20"), "12:25:00")
    asked = []
    monkeypatch.setattr(m, "live_results", lambda card, race_nos=None: asked.append(race_nos) or set())
    assert m.quick_results(now=hkt(DAY, "12:32:00")) is False                  # 2 minutes after the post
    assert asked == []
    assert m.quick_results(now=hkt(DAY, "13:05:00")) is False                  # not out yet
    assert asked == [[1, 2]]


def test_quick_results_do_nothing_on_other_days(raceday):
    update(raceday, "12:24:00", all_at("12:24:20"), "12:25:00")
    raceday.now = hkt("2026-11-15", "12:45:00")
    assert m.quick_results(now=raceday.now) is False
