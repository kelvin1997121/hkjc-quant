"""Reading HKJC's pages: times, class lines, dividends, odds cells, trackwork dates."""
from datetime import date

import pytest
from bs4 import BeautifulSoup

from hkjc import odds, reports, scraper, trackwork


@pytest.mark.parametrize("text, seconds", [
    ("1:09.68", 69.68), ("1.09.68", 69.68), ("57.30", 57.3), ("(2:01.05)", 121.05), ("", None), ("---", None),
])
def test_parse_time(text, seconds):
    assert scraper.parse_time(text) == seconds


def test_class_line():
    assert scraper._parse_class_line("第五班 - 1200米 - (40-0)") == ("第五班", 1200, "40-0")
    assert scraper._parse_class_line("一級賽 - 1600米") == ("一級賽", 1600, "")


def test_track():
    assert scraper._parse_track('草地 - "C+3" 賽道') == ("TURF", "C+3")
    assert scraper._parse_track("全天候跑道") == ("AWT", "")


def test_race_number_in_a_heading():
    assert scraper._race_no_in("第 7 場 (123)") == 7
    assert scraper._race_no_in("沒有場次") is None


def test_dividends_table():
    html = """<table>
      <tr><td>彩池</td><td>勝出組合</td><td>派彩 (HK$)</td></tr>
      <tr><td>獨贏</td><td>2</td><td>172.50</td></tr>
      <tr><td>位置</td><td>2</td><td>49.00</td></tr>
      <tr><td>7</td><td>26.00</td></tr>
      <tr><td>11</td><td>1,234.50</td></tr>
      <tr><td>連贏</td><td>2,7</td><td>-</td></tr>
    </table>"""
    assert scraper._parse_dividends(BeautifulSoup(html, "html.parser")) == {
        "獨贏": [["2", 172.5]],
        "位置": [["2", 49.0], ["7", 26.0], ["11", 1234.5]],      # one pool over several rows
        "連贏": [["2,7", None]],                                 # listed but not paid
    }
    assert scraper._parse_dividends(BeautifulSoup("<p>no table yet</p>", "html.parser")) == {}


def test_odds_cells():
    page = {"updated": "14/11/2026 12:25",
            "rows": [{"no": "1", "win": "3.5", "place": "1.6"}, {"no": "2", "win": "退出", "place": ""},
                     {"no": "3", "win": "SCR", "place": "-"}, {"no": "4", "win": "", "place": ""},
                     {"no": "", "win": "9", "place": "9"}],
            "qin": {"1-3": "12", "1-4": "-"}, "qpl": {}}
    out = odds._parse(page)
    assert out["win"] == {1: 3.5} and out["place"] == {1: 1.6}
    assert out["scratched"] == [2, 3]
    assert out["qin"] == {"1-3": 12.0}
    assert "qpl" not in out                                     # nothing readable: the pool is left out


def test_trackwork_dates_get_the_right_year():
    assert trackwork._full_date("05/11", date(2026, 11, 12)) == "2026-11-05"
    assert trackwork._full_date("12/11", date(2026, 11, 12)) == "2026-11-12"
    assert trackwork._full_date("28/12", date(2027, 1, 3)) == "2026-12-28"      # last year, not next December


def test_trackwork_entries():
    td = BeautifulSoup("<td>05/11: 沙田 全天候 29.9 26.8 (56.7)<br>02/11: 沙田 草地 快踱</td>", "html.parser")
    assert trackwork._entries(td, date(2026, 11, 12)) == [["2026-11-05", "沙田 全天候 29.9 26.8 (56.7)"],
                                                          ["2026-11-02", "沙田 草地 快踱"]]


def test_trial_margins():
    assert reports.margin_lengths("1-1/4L") == 1.25
    assert reports.margin_lengths("Neck") == 0.3
    assert reports.margin_lengths("12L") == 12.0
    assert reports.margin_lengths("") == reports.margin_lengths("---") == 0.0


def test_ran_wide_reads_comments_and_incident_reports():
    assert reports.ran_wide({"comment": "全程走三疊,沒有遮擋"})
    assert reports.ran_wide({"incident": "走外疊"})
    assert not reports.ran_wide({"comment": "沿內欄競跑"})
