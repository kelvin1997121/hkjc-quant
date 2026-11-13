"""Reading HKJC's result, sectional-time and race card pages (hkjc/scraper.py), and the nightly
check that reads one stored race again (selfcheck.page_check).

The pages in tests/fixtures/pages are miniatures: the tags, classes, ids and column headings
the real pages had on 13 Nov 2026, with made-up horses. They pin down what the parsers do with
that structure. Whether HKJC still serves that structure is what page_check watches every night.
"""
from datetime import date

import pytest
from bs4 import BeautifulSoup

from conftest import FIXTURES
from hkjc import scraper, selfcheck, store

PAGES = FIXTURES / "pages"
DAY = date(2026, 11, 11)


def page(name):
    return (PAGES / name).read_text(encoding="utf-8")


def parse_result(html=None, venue="HV", day=DAY, race_no=1):
    return scraper._parse_result_race(BeautifulSoup(html or page("result.html"), "html.parser"), venue, day, race_no)


# ------------------------------------------------------------------ result page

def test_result_race_details():
    r = parse_result()
    assert (r["date"], r["venue"], r["race_no"], r["race_id"]) == ("2026-11-11", "HV", 1, 178)
    assert (r["class"], r["distance"], r["rating_band"], r["going"]) == ("第五班", 1650, "40-0", "好地")
    assert (r["race_name"], r["surface"], r["course"]) == ("測試讓賽", "TURF", "B")
    assert r["cum_times"] == [28.07, 53.08, 76.63, 100.73]
    assert r["sectionals"] == [28.07, 25.01, 23.55, 24.1]          # not the 200 m splits printed beside them


def test_result_runner_columns():
    first = parse_result()["runners"][0]
    assert first == {"place_raw": "1", "place": 1, "no": 5, "horse": "甲馬", "horse_code": "K101",
                     "jockey": "騎師甲", "jockey_id": "AAA", "trainer": "練馬師甲", "trainer_id": "TA",
                     "weight": 133, "body_weight": 1142, "draw": 6, "lbw": "---", "running": [6, 5, 5, 1],
                     "time": 100.73, "win_odds": 10.0}


def test_result_rows_without_a_plain_place():
    runners = {x["no"]: x for x in parse_result()["runners"]}
    assert len(runners) == 5
    assert (runners[11]["place_raw"], runners[11]["place"]) == ("3", 3)              # a photo icon in the cell
    pulled_up = runners[2]
    assert (pulled_up["place"], pulled_up["place_raw"], pulled_up["time"]) == (None, "PU", None)
    assert pulled_up["running"] == [1, 1, 12] and pulled_up["win_odds"] == 15.0      # it ran: it has a price
    withdrawn = runners[9]
    assert (withdrawn["place"], withdrawn["place_raw"]) == (None, "WV")
    assert (withdrawn["draw"], withdrawn["body_weight"], withdrawn["win_odds"], withdrawn["running"]) == \
        (None, None, None, [])


def test_result_dividends():
    assert parse_result()["dividends"] == {
        "獨贏": [["5", 109.0]], "位置": [["5", 33.0], ["6", 17.5], ["11", 26.5]],
        "連贏": [["5,6", 1208.0]], "位置Q": [["5,6", 65.0], ["5,11", 129.5], ["6,11", 71.0]],
        "三T": [["-", None]]}


def test_result_of_another_race_or_meeting_is_refused():
    # HKJC answers a request it can't serve with whatever race it has
    assert parse_result(race_no=2) is None
    assert parse_result(day=date(2026, 11, 14)) is None


def test_result_page_without_its_tables_gives_nothing():
    assert parse_result("<html><body><p>沒有相關資料</p></body></html>") is None
    assert parse_result(page("result.html").replace('class="performance"', 'class="perf_new"')) is None


def test_sectional_times(monkeypatch):
    monkeypatch.setattr(scraper, "get", lambda *a, **k: page("sectional.html"))
    assert scraper.fetch_sectionals(DAY, 1) == {5: [28.79, 24.69, 23.67, 23.58], 6: [29.51, 24.73, 23.51, 23.02],
                                                11: [29.03, 24.73, 23.59, 23.54]}       # not the horse that pulled up


# ------------------------------------------------------------------ race card page

def card(html=None, race_no=1):
    return scraper._parse_card_race(html or page("card.html"), date(2026, 11, 14), "ST", race_no)


def test_card_race_details():
    c = card()
    assert {k: v for k, v in c.items() if k != "runners"} == {
        "date": "2026-11-14", "venue": "ST", "race_no": 1, "race_name": "測試讓賽", "post_time": "12:30",
        "distance": 1200, "surface": "TURF", "course": "A", "going": "好地", "rating_band": "60-40", "class": "第四班"}


def test_card_going_is_empty_until_hkjc_announces_it():
    assert card(page("card.html").replace("1200米, 好地", "1200米"))["going"] == ""


def test_card_runner_columns():
    one = card()["runners"][0]
    assert (one["no"], one["horse"], one["horse_code"], one["horse_id"]) == (1, "甲馬", "L245", "HK_2024_L245")
    assert (one["jockey"], one["jockey_id"], one["trainer"], one["trainer_id"]) == ("騎師甲", "AAA", "練馬師甲", "TA")
    assert (one["draw"], one["weight"], one["rating"], one["rating_change"]) == (2, 133, 60, 0)
    assert (one["body_weight"], one["body_weight_change"], one["days_since"], one["age"]) == (928, 7, 20, 3)
    assert (one["last6"], one["gear"], one["sex"], one["import_type"]) == ("3/13/1/2/5", "XB", "閹", "PPG")
    assert one["scratched"] is False


def test_card_apprentice_claim_comes_off_the_weight():
    two = card()["runners"][1]
    assert (two["jockey"], two["allowance"]) == ("騎師乙", -2)
    assert (two["handicap_weight"], two["weight"]) == (133, 131)       # results show the weight carried


def test_card_horse_without_a_draw_is_scratched():
    three = card()["runners"][2]
    assert three["draw"] is None and three["scratched"] is True


def test_card_of_another_race_is_refused():
    assert card(race_no=4) is None
    assert card("<html><body></body></html>") is None


def test_race_links_find_the_meeting_and_ignore_overseas_races():
    links = scraper._race_links(page("card.html"), 'RaceCard|localtrackwork|racereportext|"')
    assert links == {("2026/11/14", "ST"): {1, 2, 3}}                  # S1 (a simulcast) is left out


# ------------------------------------------------------------------ the nightly re-read

@pytest.fixture
def stored(sandbox, monkeypatch):
    """The race in result.html saved as the system would have it; HKJC answers with `served`."""
    served = {"LocalResults.aspx": page("result.html"), "displaysectionaltime": page("sectional.html")}

    def get(path, **params):
        key = next(k for k in served if k in path)
        if served[key] is None:
            raise RuntimeError(f"failed to fetch {path}")
        return served[key]

    monkeypatch.setattr(scraper, "get", get)
    monkeypatch.setattr(selfcheck, "today_hkt", lambda: date(2026, 11, 13))
    race = parse_result()
    scraper.attach_sectionals(race)
    race["runners"][0].update(rating=40, gear="B", comment="留守中間位置")       # added later from other pages
    store.save_results([race])
    return served


def check():
    return selfcheck.page_check(store.load_all_results())


def test_page_check_passes_when_the_page_reads_as_stored(stored):
    c = check()
    assert c["ok"] is True and c["name"] == "馬會頁面"
    assert "11/11 第 1 場" in c["detail"] and "一致" in c["detail"]


def test_page_check_reads_the_first_race_of_the_newest_finished_meeting(stored, monkeypatch):
    asked = []
    real = scraper.get
    monkeypatch.setattr(scraper, "get", lambda path, **params: asked.append((path, params)) or real(path, **params))
    older = dict(parse_result(), date="2026-11-08", venue="ST", has_sect=True)
    today = dict(parse_result(), date="2026-11-13", has_sect=True)             # still being run: not used
    store.save_results([older])
    store.save_results([today])
    check()
    assert asked[0] == ("LocalResults.aspx", {"RaceDate": "2026/11/11", "Racecourse": "HV", "RaceNo": 1})
    assert len(asked) == 2


def test_page_check_fails_when_the_layout_changes(stored):
    stored["LocalResults.aspx"] = page("result.html").replace('class="performance"', 'class="perf_new"')
    c = check()
    assert c["ok"] is False and "頁面可能改咗版" in c["detail"]


def test_page_check_fails_when_a_column_moves(stored):
    # 檔位 and 排位體重 swap places: every row still parses, with the wrong numbers
    html = page("result.html").replace("<td>排位<br>體重</td>", "<td>X</td>") \
                              .replace("<td>檔位</td>", "<td>排位<br>體重</td>").replace("<td>X</td>", "<td>檔位</td>")
    stored["LocalResults.aspx"] = html
    c = check()
    assert c["ok"] is False and "名次、賠率或走位" in c["detail"]


def test_page_check_names_what_differs(stored):
    stored["LocalResults.aspx"] = page("result.html").replace("1,208.00", "1,280.00")
    assert "派彩" in check()["detail"]
    stored["LocalResults.aspx"] = page("result.html")
    stored["displaysectionaltime"] = page("sectional.html").replace("28.79", "28.97")
    c = check()
    assert c["ok"] is False and "分段時間" in c["detail"] and "派彩" not in c["detail"]


def test_page_check_fails_when_the_sectional_page_is_empty(stored):
    stored["displaysectionaltime"] = "<html><body></body></html>"
    assert check()["ok"] is False


def test_page_check_reports_a_site_that_cannot_be_reached(stored):
    stored["LocalResults.aspx"] = None
    c = check()
    assert c["ok"] is False and "連接唔到馬會網站" in c["detail"]


def test_page_check_waits_when_nothing_is_stored(sandbox):
    assert selfcheck.page_check([])["ok"] is None
