"""Live odds read from the public bet.hkjc.com race pages.

A headless browser opens each race's odds page the same way a viewer
would and reads the numbers shown. One page load per race, only on (or
the day before) a race day. The 獨贏/位置/連贏/位置Q page carries all four
pools, so quinella and quinella-place odds come at no extra cost; if that
page can't be read, the plain 獨贏/位置 page is tried instead.
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone

URL_WPQ = "https://bet.hkjc.com/ch/racing/wpq/{date}/{venue}/{race}"
URL = "https://bet.hkjc.com/ch/racing/wp/{date}/{venue}/{race}"
# A slow or blocked site must not use up the update's 15 minutes and lose the results with
# it. Races run in order, so the next race to run is read first.
BUDGET_SECONDS = 240

# cells carry stable ids on both the wide and the narrow layout:
# odds_WIN_<race>_<no>, odds_PLA_<race>_<no>, qb_QIN_<a>_<b>, qb_QPL_<a>_<b>
_EXTRACT_WPQ = """
(race) => {
  const out = {rows: [], qin: {}, qpl: {}, updated: null};
  const txt = id => { const e = document.getElementById(id); return e ? e.innerText.trim() : ''; };
  for (const td of document.querySelectorAll(`td[id^="runnerNo_${race}_"]`)) {
    const no = td.id.split('_').pop();
    out.rows.push({no: td.innerText.trim() || no,
                   win: txt(`odds_WIN_${race}_${no}`), place: txt(`odds_PLA_${race}_${no}`)});
  }
  for (const pool of ['QIN', 'QPL']) {
    for (const td of document.querySelectorAll(`[id^="qb_${pool}_"]`)) {
      const [a, b] = td.id.split('_').slice(2);
      out[pool.toLowerCase()][`${a}-${b}`] = td.innerText.trim();
    }
  }
  const m = document.body.innerText.match(/更新時間[:：]\\s*([0-9/]+\\s+[0-9:]+)/);
  out.updated = m ? m[1] : null;
  return out;
}
"""

_EXTRACT = """
(race) => {
  const out = {rows: [], updated: null};
  for (const td of document.querySelectorAll(`td[id^="runnerNo_${race}_"]`)) {
    const tr = td.closest('tr');
    const odds = [...tr.querySelectorAll('td.rc-odds')].map(c => c.innerText.trim());
    out.rows.push({no: td.innerText.trim(), win: odds[0] || '', place: odds[1] || ''});
  }
  const m = document.body.innerText.match(/更新時間[:：]\\s*([0-9/]+\\s+[0-9:]+)/);
  out.updated = m ? m[1] : null;
  return out;
}
"""


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _parse(data):
    """page rows -> win, place, scratched (+ qin, qpl when the page had them)"""
    win, place, scratched = {}, {}, []
    for r in data["rows"]:
        no = int(re.sub(r"\D", "", r["no"]) or 0)
        if not no:
            continue
        w, p = _num(r["win"]), _num(r["place"])
        if w:
            win[no] = w
        if p:
            place[no] = p
        if "退出" in r["win"] or "SCR" in r["win"].upper():
            scratched.append(no)
    out = {"updated": data["updated"], "win": win, "place": place, "scratched": scratched}
    for pool in ("qin", "qpl"):
        got = {k: v for k, v in ((k, _num(s)) for k, s in (data.get(pool) or {}).items()) if v}
        if got:
            out[pool] = got
    return out


def fetch_odds(date_iso: str, venue: str, race_nos: list[int], log=print) -> dict:
    """{race_no: {"t", "updated", "win": {no: odds}, "place": {no: odds}, "scratched": [no],
    "qin": {"a-b": odds}, "qpl": {"a-b": odds}}} (qin/qpl only when the page showed them).
    "t" is when we read that race's page; "updated" is the page's own update time."""
    from playwright.sync_api import Error as PlaywrightError  # includes timeouts
    from playwright.sync_api import sync_playwright

    out = {}
    started = time.time()
    with sync_playwright() as pw:
        try:  # GitHub runners ship Google Chrome: no 5-minute browser install needed
            browser = pw.chromium.launch(channel="chrome")
        except Exception:
            browser = pw.chromium.launch()
        # cloud runners are on UTC; the page prints its update time in local time
        page = browser.new_page(locale="zh-HK", timezone_id="Asia/Hong_Kong")
        use_wpq = True
        for n in race_nos:
            if time.time() - started > BUDGET_SECONDS:
                log(f"   odds: out of time, skipped from R{n}")
                break
            got = None
            if use_wpq:
                try:
                    page.goto(URL_WPQ.format(date=date_iso, venue=venue, race=n), timeout=30000)
                    page.wait_for_selector(f'[id^="odds_WIN_{n}_"]', timeout=20000)
                    page.wait_for_function(  # the table renders before the numbers arrive
                        "(r) => [...document.querySelectorAll(`[id^=\"odds_WIN_${r}_\"]`)]"
                        ".some(e => /\\d/.test(e.innerText))", arg=n, timeout=15000)
                    got = _parse(page.evaluate(_EXTRACT_WPQ, n))
                except PlaywrightError as e:
                    log(f"   odds R{n} (wpq): {type(e).__name__}")
            if not (got and got["win"]) and time.time() - started <= BUDGET_SECONDS:
                try:  # the plain win/place page, as before
                    page.goto(URL.format(date=date_iso, venue=venue, race=n), timeout=30000)
                    page.wait_for_selector(f'td[id^="runnerNo_{n}_"]', timeout=20000)
                    page.wait_for_function(
                        "(r) => [...document.querySelectorAll(`td[id^=\"runnerNo_${r}_\"]`)]"
                        ".some(td => /\\d/.test(td.closest('tr').querySelector('td.rc-odds')?.innerText || ''))",
                        arg=n, timeout=15000)
                    got = _parse(page.evaluate(_EXTRACT, n))
                except PlaywrightError as e:
                    log(f"   odds R{n}: {type(e).__name__}")
                    continue
                if got["win"] and use_wpq:  # the plain page works and the other doesn't:
                    use_wpq = False         # don't wait 20 s on it for every race
                    log("   odds: quinella page unreadable, win/place page for the rest")
            if got and got["win"]:
                # when this race's page was read: the caller needs it per race, as the whole
                # batch can end a minute or more after the next race to run was read
                got["t"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                out[n] = got
        browser.close()
    out["_fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return out


if __name__ == "__main__":
    import json
    import sys
    d, v = sys.argv[1], sys.argv[2]
    print(json.dumps(fetch_odds(d, v, [int(x) for x in sys.argv[3:]] or [1]), ensure_ascii=False))
