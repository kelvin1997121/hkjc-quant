"""Reading the odds pages with a real browser (hkjc/odds.py).

bet.hkjc.com is replaced by a local page with the same cell ids, so this runs the real
Playwright steps: open the page, wait for the table, wait for the numbers, read them. Whether
HKJC's own page still looks like this is what the 賽前賠率 and 即時賠率 health checks watch.
"""
import http.server
import socket
from datetime import datetime, timedelta, timezone

import pytest

from hkjc import odds

pytestmark = pytest.mark.usefixtures("chrome")

DAY, VENUE = "2026-11-14", "ST"
RACES = {
    1: {"win": {1: "3.5", 2: "12", 3: "99"}, "place": {1: "1.6", 2: "3.2", 3: "15"},
        "qin": {"1-2": "18", "1-3": "150", "2-3": "480"}, "qpl": {"1-2": "7.5", "1-3": "42", "2-3": "-"}},
    2: {"win": {1: "2.1", 2: "退出", 3: "8.4", 4: "21"}, "place": {1: "1.2", 2: "", 3: "2.4", 4: "5.5"},
        "qin": {"1-3": "9.5", "1-4": "30", "3-4": "66"}, "qpl": {"1-3": "4.1", "1-4": "11", "3-4": "25"}},
}


def wpq_page(race, book, delay_ms):
    """The 獨贏/位置/連贏/位置Q page: the table is there at once, the numbers are filled in later."""
    rows = "".join(f'<tr><td id="runnerNo_{race}_{no}">{no}</td><td>馬{no}</td>'
                   f'<td id="odds_WIN_{race}_{no}"></td><td id="odds_PLA_{race}_{no}"></td></tr>' for no in book["win"])
    pairs = "".join(f'<td id="qb_{pool.upper()}_{pair.replace("-", "_")}"></td>'
                    for pool in ("qin", "qpl") for pair in book[pool])
    fill = "".join(f'document.getElementById("odds_WIN_{race}_{no}").innerText = "{book["win"][no]}";'
                   f'document.getElementById("odds_PLA_{race}_{no}").innerText = "{book["place"][no]}";'
                   for no in book["win"])
    fill += "".join(f'document.getElementById("qb_{pool.upper()}_{pair.replace("-", "_")}").innerText = "{text}";'
                    for pool in ("qin", "qpl") for pair, text in book[pool].items())
    return (f'<html><meta charset="utf-8"><body><div>更新時間: 14/11/2026 12:25</div>'
            f'<table border="1">{rows}</table><table border="1"><tr>{pairs}</tr></table>'
            f'<script>setTimeout(() => {{ {fill} }}, {delay_ms});</script></body></html>')


def wp_page(race, book):
    """The plain 獨贏/位置 page: odds in td.rc-odds cells, no ids on them."""
    rows = "".join(f'<tr><td id="runnerNo_{race}_{no}">{no}</td><td>馬{no}</td>'
                   f'<td class="rc-odds">{book["win"][no]}</td><td class="rc-odds">{book["place"][no]}</td></tr>'
                   for no in book["win"])
    return (f'<html><meta charset="utf-8"><body><div>更新時間：14/11/2026 12:26</div>'
            f'<table border="1">{rows}</table></body></html>')


@pytest.fixture
def site(serve, monkeypatch):
    """A stand-in for bet.hkjc.com; site.delay_ms = how long the numbers take to appear."""
    class Site(http.server.BaseHTTPRequestHandler):
        delay_ms = 0
        seen = []

        def do_GET(self):
            parts = self.path.strip("/").split("/")
            book = RACES.get(int(parts[3])) if len(parts) == 4 and parts[0] in ("wpq", "wp") else None
            if not book:                           # an unknown race, or the browser asking for /favicon.ico
                self.send_error(404)
                return
            kind, race = parts[0], parts[3]
            Site.seen.append((kind, int(race)))
            body = (wpq_page(int(race), book, Site.delay_ms) if kind == "wpq" else wp_page(int(race), book)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    base = serve(Site)
    monkeypatch.setattr(odds, "URL_WPQ", base + "/wpq/{date}/{venue}/{race}")
    monkeypatch.setattr(odds, "URL", base + "/wp/{date}/{venue}/{race}")
    return Site


def closed_port():
    """An address nothing listens on: the browser gets 'connection refused' at once."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}"


def when(text):
    return datetime.fromisoformat(text)


def test_reads_all_four_pools(site):
    got = odds.fetch_odds(DAY, VENUE, [1, 2])
    assert got[1]["win"] == {1: 3.5, 2: 12.0, 3: 99.0}
    assert got[1]["place"] == {1: 1.6, 2: 3.2, 3: 15.0}
    assert got[1]["qin"] == {"1-2": 18.0, "1-3": 150.0, "2-3": 480.0}
    assert got[1]["qpl"] == {"1-2": 7.5, "1-3": 42.0}                 # "-" = no odds shown for that pair
    assert got[1]["updated"] == "14/11/2026 12:25"
    assert got[1]["scratched"] == []


def test_scratched_horse_is_listed_and_has_no_price(site):
    got = odds.fetch_odds(DAY, VENUE, [2])[2]
    assert got["scratched"] == [2]
    assert got["win"] == {1: 2.1, 3: 8.4, 4: 21.0} and 2 not in got["place"]


def test_each_race_carries_the_time_it_was_read(site):
    before = datetime.now(timezone.utc) - timedelta(seconds=1)
    got = odds.fetch_odds(DAY, VENUE, [1, 2])
    after = datetime.now(timezone.utc) + timedelta(seconds=1)
    t1, t2, end = when(got[1]["t"]), when(got[2]["t"]), when(got["_fetched_at"])
    assert before <= t1 <= t2 <= end <= after
    assert t1.tzinfo is not None                                      # stored with its time zone


def test_waits_for_numbers_that_arrive_after_the_table(site):
    site.delay_ms = 600
    got = odds.fetch_odds(DAY, VENUE, [1])
    assert got[1]["win"][1] == 3.5 and got[1]["qin"]["1-2"] == 18.0


def test_falls_back_to_the_win_place_page_when_the_quinella_page_is_down(site, monkeypatch):
    monkeypatch.setattr(odds, "URL_WPQ", closed_port() + "/wpq/{date}/{venue}/{race}")
    log = []
    got = odds.fetch_odds(DAY, VENUE, [1, 2], log=log.append)
    assert got[1]["win"] == {1: 3.5, 2: 12.0, 3: 99.0} and got[1]["place"][1] == 1.6
    assert "qin" not in got[1] and "qpl" not in got[1]
    assert got[2]["scratched"] == [2] and "t" in got[2]
    assert got[1]["updated"] == "14/11/2026 12:26"
    assert sum("quinella page unreadable" in line for line in log) == 1
    assert sum("(wpq)" in line for line in log) == 1                  # tried once, not again for race 2


def test_a_site_that_is_down_gives_no_odds_and_no_error(monkeypatch):
    down = closed_port()
    monkeypatch.setattr(odds, "URL_WPQ", down + "/wpq/{date}/{venue}/{race}")
    monkeypatch.setattr(odds, "URL", down + "/wp/{date}/{venue}/{race}")
    log = []
    got = odds.fetch_odds(DAY, VENUE, [1, 2], log=log.append)
    assert list(got) == ["_fetched_at"]
    assert len(log) == 4                                              # both pages, both races


def test_stops_when_the_time_budget_is_used_up(site, monkeypatch):
    monkeypatch.setattr(odds, "BUDGET_SECONDS", -1)
    log = []
    got = odds.fetch_odds(DAY, VENUE, [1, 2], log=log.append)
    assert list(got) == ["_fetched_at"] and site.seen == []
    assert any("out of time" in line for line in log)
