"""The web page (docs/index.html), opened in a real browser.

The page is served locally with data files made by the real code from the saved 11 Nov 2026
meeting, so a change on either side that the other does not expect shows up here: a renamed
field, a script error, a section that no longer draws. GitHub is blocked, so the page falls
back to the data/ folder beside it, as it does on a network that blocks GitHub's raw files.
"""
import http.server
import json
import shutil
from datetime import datetime, timedelta, timezone

import pytest

from conftest import FIXTURES
from hkjc import __main__ as m, paper, store

NOW = datetime(2026, 11, 12, 12, 0, tzinfo=timezone(timedelta(hours=8)))
STRONG = {"date": "2026-10-21", "venue": "HV", "race_no": 3, "place": None, "place_raw": "PU",
          "faster": 0.42, "placed": 3}


@pytest.fixture(scope="module")
def browser(chrome):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        try:
            b = pw.chromium.launch(channel="chrome")
        except Exception:
            b = pw.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def site(meeting_1111, serve):
    """docs/ as the cloud would leave it after the 11 Nov meeting. site.latest can be edited
    and saved again with site.save() before a page is opened."""
    docs = meeting_1111 / "docs"
    data = docs / "data"
    shutil.copy(store.ROOT / "docs" / "index.html", docs / "index.html")
    for name in ("selfcheck.json", "oldbank.json"):
        shutil.copy(FIXTURES / "site" / name, data / name)
    m.update_track_record()
    paper.update(now=NOW)

    class Docs(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(docs), **kwargs)

        def log_message(self, *args):
            pass

    class Site:
        base = serve(Docs)
        latest = store.read_json(data / "2026-11-11_HV.json")

        def save(self):
            store.write_json(data / "latest.json", self.latest)
            return self

    return Site().save()


@pytest.fixture
def view(browser, site):
    """view() -> the loaded page. Any script error on it fails the test when it ends."""
    pages, errors = [], []

    def load():
        page = browser.new_page(locale="zh-HK", timezone_id="Asia/Hong_Kong")
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.route("**/*", lambda route: route.continue_() if route.request.url.startswith(site.base)
                   else route.abort())
        page.goto(site.base + "/index.html")
        page.wait_for_selector("#tabs button")
        pages.append(page)
        return page

    yield load
    for page in pages:
        page.close()
    assert errors == []


def text(page, selector):
    return page.inner_text(selector)


def test_race_page_shows_the_meeting(view):
    page = view()
    assert "2026-11-11 跑馬地" in text(page, "#sub") and "9 場" in text(page, "#sub")
    tabs = page.locator("#tabs button").all_inner_texts()
    assert tabs == [f"第{n}場 ✓" for n in range(1, 10)]
    assert "第1場" in text(page, "#view h2") and "爵祿讓賽" in text(page, "#view h2")
    assert page.locator("#view tbody tr").count() == 12                       # one row per runner
    assert "賽果:5 - 6 - 11" in text(page, "#view .res")
    assert page.locator("#view tbody tr.won").count() == 1
    assert page.locator("#view tbody tr.won td").nth(1).inner_text().startswith("5.")


def test_ranking_note_says_the_model_has_no_weight(view):
    note = text(view(), "#view")
    assert "模型 0 / 市場" in note and "唔會標「值博」" in note


def test_choosing_a_race_shows_that_race_and_its_paper_bet(view):
    page = view()
    page.locator("#tabs button").nth(1).click()
    assert "第2場" in text(page, "#view h2")
    assert page.locator("#tabs button.on").inner_text() == "第2場 ✓"
    assert "模擬倉(首選+次選連贏)" in text(page, "#view") and "連贏 4-3" in text(page, "#view")
    assert "已結算" in text(page, "#view")


def test_the_first_race_without_a_result_is_shown_first(view, site):
    for race in site.latest["races"][7:]:
        del race["result"]
    site.save()
    page = view()
    assert "第8場" in text(page, "#view h2")
    assert page.locator("#tabs button").all_inner_texts()[6:] == ["第7場 ✓", "第8場", "第9場"]
    assert page.locator("#view .res").count() == 0


def test_a_race_analysed_after_the_off_says_so(view, site):
    site.latest["races"][0]["post_hoc"] = True
    site.save()
    assert "此場預測於開跑後產生,不計入往績" in text(view(), "#view .res")


def test_strong_race_mark_and_a_pulled_up_run(view, site):
    site.latest["races"][0]["runners"][0]["strong"] = [STRONG]
    site.save()
    page = view()
    assert page.locator("#view .tag.strong").count() == 1
    assert "出自強組" in text(page, "#view")
    assert "10/21 跑馬地第3場 收停" in text(page, "#view")                      # PU is spelled out
    assert "每公里 0.42 秒" in text(page, "#view") and "只作參考" in text(page, "#view")


def test_live_record_is_shown_beside_the_final_favourite(view):
    perf = text(view(), "#perf")
    assert "實戰往績" in perf and "由 11/11 起" in perf
    assert "同場最後熱門勝出(對照)" in perf and "同場最後熱門入位(對照)" in perf
    assert "11.1%" in perf and "44.4%" in perf                                 # 1 win and 4 placed from 9


def test_self_check_card_lists_every_health_check(view):
    card = text(view(), "#selfcheck")
    checks = json.loads((FIXTURES / "site" / "selfcheck.json").read_text(encoding="utf-8"))["health"]["checks"]
    assert "系統自我檢測" in card and len(checks) >= 8
    for check in checks:
        assert check["name"] in card
    assert "自我學習" in card and "舊季度一次性測試" in card and "真實派彩回測" in card


def test_paper_account_page(view):
    page = view()
    page.click('#pages button[data-p="paper"]')
    assert page.locator("#racepage").is_hidden() and page.locator("#paperpage").is_visible()
    body = text(page, "#paperpage")
    assert "$910" in body and "-$90" in body                                   # main account after 9 races
    assert "首選+次選連贏" in body and "← 模擬倉跟緊" in body
    for name in ("首選位置", "冷門馬位置", "彩池值博"):
        assert name in body
    assert "$1,113" in body and "+$113" in body                                # 冷門馬位置, the only book ahead
    assert "9 / 300" in body                                                   # settled races / next review
    assert "2026-11-11 跑馬地 · 9 注" in body
    assert page.locator("#paperpage details.mt").first.get_attribute("open") is not None
    assert "連贏 6-12" in body and "賽果 5-6-11" in body


def test_paper_page_opens_from_its_own_address(browser, site, view):
    page = browser.new_page()
    page.route("**/*", lambda route: route.continue_() if route.request.url.startswith(site.base) else route.abort())
    page.goto(site.base + "/index.html#paper")
    page.wait_for_selector("#paperpage details.mt")
    assert page.locator("#racepage").is_hidden()
    page.close()


def test_page_still_draws_when_optional_files_are_missing(view, site):
    data = store.DOCS / "data"
    for name in ("selfcheck.json", "paper.json", "oldbank.json", "track_record.json"):
        (data / name).unlink()
    page = view()
    assert "第1場" in text(page, "#view h2")
    page.click('#pages button[data-p="paper"]')
    assert "模擬倉暫時未有數據" in text(page, "#paperpage")


def test_a_page_with_no_data_says_so(browser, site):
    (store.DOCS / "data" / "latest.json").unlink()
    page = browser.new_page()
    page.route("**/*", lambda route: route.continue_() if route.request.url.startswith(site.base) else route.abort())
    page.goto(site.base + "/index.html")
    page.wait_for_function("document.querySelector('#sub').innerText.includes('暫時攞唔到數據')")
    page.close()
