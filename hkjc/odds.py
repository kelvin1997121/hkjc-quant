"""Live win / place odds read from the public bet.hkjc.com race pages.

A headless browser opens each race's odds page the same way a viewer
would and reads the numbers shown in the odds table. One page load per
race, only on (or the day before) a race day.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

URL = "https://bet.hkjc.com/ch/racing/wp/{date}/{venue}/{race}"

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


def fetch_odds(date_iso: str, venue: str, race_nos: list[int], log=print) -> dict:
    """{race_no: {"updated", "win": {no: odds}, "place": {no: odds}, "scratched": [no]}}"""
    from playwright.sync_api import TimeoutError as PWTimeout
    from playwright.sync_api import sync_playwright

    out = {}
    with sync_playwright() as pw:
        try:  # GitHub runners ship Google Chrome: no 5-minute browser install needed
            browser = pw.chromium.launch(channel="chrome")
        except Exception:
            browser = pw.chromium.launch()
        # cloud runners are on UTC; the page prints its update time in local time
        page = browser.new_page(locale="zh-HK", timezone_id="Asia/Hong_Kong")
        for n in race_nos:
            try:
                page.goto(URL.format(date=date_iso, venue=venue, race=n), timeout=45000)
                page.wait_for_selector(f'td[id^="runnerNo_{n}_"]', timeout=30000)
                # the table renders before the numbers arrive
                page.wait_for_function(
                    "(r) => [...document.querySelectorAll(`td[id^=\"runnerNo_${r}_\"]`)]"
                    ".some(td => /\\d/.test(td.closest('tr').querySelector('td.rc-odds')?.innerText || ''))",
                    arg=n, timeout=20000)
                data = page.evaluate(_EXTRACT, n)
            except PWTimeout:
                log(f"   odds R{n}: page did not load in time")
                continue
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
            if win:
                out[n] = {"updated": data["updated"], "win": win, "place": place,
                          "scratched": scratched}
        browser.close()
    out["_fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return out


if __name__ == "__main__":
    import json
    import sys
    d, v = sys.argv[1], sys.argv[2]
    print(json.dumps(fetch_odds(d, v, [int(x) for x in sys.argv[3:]] or [1]), ensure_ascii=False))
