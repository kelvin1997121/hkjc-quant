"""Scraper for the public HKJC racing information pages (racing.hkjc.com).

Only the public information site is used: race cards, results and the
race-day selector. Requests are throttled so we stay a polite client.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

import requests
from bs4 import BeautifulSoup

BASE = "https://racing.hkjc.com/racing/information/Chinese/Racing/"
MAIN_VENUES = ("ST", "HV")
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36 hkjc-quant/1.0"
)

_local = threading.local()


def _session() -> requests.Session:
    """One session per thread (requests.Session isn't guaranteed thread-safe)."""
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": UA, "Accept-Language": "zh-HK,zh;q=0.9"})
        # Optional proxy (e.g. a Hong Kong egress) for runners whose IP gets blocked.
        if os.environ.get("HKJC_PROXY"):
            s.proxies.update({"http": os.environ["HKJC_PROXY"], "https": os.environ["HKJC_PROXY"]})
        _local.session = s
    return s


# Politeness: a global cap on request *starts* per second, shared by all threads.
# Live runs use 1/s; backfills raise HKJC_RATE and use a few threads so the ~1.5 s
# page latency overlaps instead of serialising.
RATE = float(os.environ.get("HKJC_RATE", "1.0"))
CONCURRENCY = int(os.environ.get("HKJC_CONCURRENCY", "1"))
_rate_lock = threading.Lock()
_next_start = 0.0


def _pace():
    global _next_start
    with _rate_lock:
        now = time.time()
        start = max(now, _next_start)
        _next_start = start + 1.0 / RATE
    if start > now:
        time.sleep(start - now)


def get(path: str, **params) -> str:
    url = ("https://racing.hkjc.com" + path) if path.startswith("/") else BASE + path
    for attempt in range(4):
        _pace()
        try:
            r = _session().get(url, params=params or None, timeout=30)
            if r.status_code == 200:
                r.encoding = "utf-8"
                return r.text
        except requests.RequestException:
            pass
        time.sleep(2 + attempt * 3)
    raise RuntimeError(f"failed to fetch {url} {params}")


def _txt(el) -> str:
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip() if el else ""


def _int(s):
    m = re.search(r"-?\d+", s or "")
    return int(m.group()) if m else None


def _float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse_time(s: str):
    """'1:09.68' -> 69.68, '57.30' -> 57.3"""
    s = (s or "").strip().strip("()")
    m = re.fullmatch(r"(?:(\d+)[:.])?(\d{1,2})\.(\d{2})", s)
    if not m:
        return None
    mins = int(m.group(1) or 0)
    return round(mins * 60 + int(m.group(2)) + int(m.group(3)) / 100, 2)


def _code_from_href(a, key):
    if not a or not a.get("href"):
        return None
    m = re.search(key + r"=([A-Za-z0-9_]+)", a["href"])
    return m.group(1) if m else None


def _race_links(html: str, page: str):
    """Return (date 'YYYY/MM/DD', venue, [race numbers]) for main venue links."""
    links = re.findall(
        r"(?:" + page + r")\?racedate=(\d{4}/\d{2}/\d{2})&(?:amp;)?Racecourse=(\w+)&(?:amp;)?RaceNo=(\d+)",
        html, flags=re.I,
    )
    races = {}
    for d, v, n in links:
        if v in MAIN_VENUES:
            races.setdefault((d, v), set()).add(int(n))
    return races


# --------------------------------------------------------------------------- results

def list_result_dates() -> list[date]:
    """All race dates offered by the results page selector (newest first)."""
    html = get("LocalResults.aspx")
    out = []
    for d in re.findall(r'"date":"(\d{2}/\d{2}/\d{4})"', html):
        out.append(datetime.strptime(d, "%d/%m/%Y").date())
    return sorted(set(out), reverse=True)


def _parse_result_race(soup: BeautifulSoup, venue: str, race_date: date, race_no: int):
    tab = soup.select_one("div.race_tab table")
    perf = soup.select_one("div.performance table")
    if not tab or not perf:
        return None
    rows = tab.select("tbody tr")
    info = {"date": race_date.isoformat(), "venue": venue, "race_no": race_no}
    header = _txt(tab.select_one("thead td"))
    info["race_id"] = _int(re.sub(r"第\s*\d+\s*場", "", header))
    for tr in rows:
        tds = tr.find_all("td", recursive=False)
        if not tds:
            continue
        label = _txt(tds[1]) if len(tds) > 1 else ""
        first = _txt(tds[0])
        if "場地狀況" in label:
            info["class"], info["distance"], info["rating_band"] = _parse_class_line(first)
            info["going"] = _txt(tds[2])
        elif "賽道" in label:
            info["race_name"] = first
            info["surface"], info["course"] = _parse_track(_txt(tds[2]))
        elif label.startswith("時間"):
            info["prize"] = first
            info["cum_times"] = [parse_time(_txt(td)) for td in tds[2:]]
        elif "分段時間" in label:
            secs = []
            for td in tds[2:]:
                first_tok = (td.find(string=True, recursive=False) or "").strip()
                secs.append(_float(first_tok))
            info["sectionals"] = secs

    heads = [_txt(td) for td in perf.select("thead td")]
    col = {h.replace(" ", ""): i for i, h in enumerate(heads)}
    runners = []
    for tr in perf.select("tbody tr"):
        tds = tr.find_all("td", recursive=False)
        if len(tds) < 8:
            continue
        def c(name):
            i = col.get(name)
            return tds[i] if i is not None and i < len(tds) else None
        name_td = c("馬名")
        horse_a = name_td.find("a") if name_td else None
        code_m = re.search(r"\(([A-Z]\d{3})\)", _txt(name_td))
        pos_td = c("沿途走位")
        running = [int(x) for x in re.findall(r"\d+", _txt(pos_td))] if pos_td else []
        place_raw = _txt(c("名次"))
        runners.append({
            "place_raw": place_raw,
            "place": _int(place_raw) if re.match(r"^\d", place_raw) else None,
            "no": _int(_txt(c("馬號"))),
            "horse": horse_a.get_text(strip=True) if horse_a else _txt(name_td),
            "horse_code": code_m.group(1) if code_m else None,
            "jockey": _txt(c("騎師")),
            "jockey_id": _code_from_href(c("騎師").find("a") if c("騎師") else None, "jockeyid"),
            "trainer": _txt(c("練馬師")),
            "trainer_id": _code_from_href(c("練馬師").find("a") if c("練馬師") else None, "trainerid"),
            "weight": _int(_txt(c("實際負磅"))),
            "body_weight": _int(_txt(c("排位體重"))),
            "draw": _int(_txt(c("檔位"))),
            "lbw": _txt(c("頭馬距離")),
            "running": running,
            "time": parse_time(_txt(c("完成時間"))),
            "win_odds": _float(_txt(c("獨贏賠率"))),
        })
    info["runners"] = runners
    return info


def _parse_class_line(s: str):
    # '第五班 - 1200米 - (40-0)' / '一級賽 - 1600米' / '新馬賽 - 1000米'
    parts = [p.strip() for p in s.split(" - ")]
    cls = parts[0] if parts else ""
    dist = None
    band = ""
    for p in parts[1:]:
        if "米" in p:
            dist = _int(p)
        elif "(" in p:
            band = p.strip("()")
    return cls, dist, band


def _parse_track(s: str):
    # '草地 - "C+3" 賽道' / '全天候跑道'
    surface = "AWT" if "全天候" in s else "TURF"
    m = re.search(r'"([^"]+)"', s.replace("&quot;", '"'))
    return surface, (m.group(1) if m else "")


def fetch_sectionals(race_date: date, race_no: int) -> dict:
    """Per-horse sectional times: {horse_no: [section seconds, ...]} (empty if unavailable)."""
    html = get("/zh-hk/local/information/displaysectionaltime",
               racedate=race_date.strftime("%d/%m/%Y"), RaceNo=race_no)
    soup = BeautifulSoup(html, "html.parser")
    table = next((t for t in soup.select("table.race_table") if "馬名" in t.get_text()), None)
    out = {}
    if not table:
        return out
    for tr in table.find_all("tr"):
        tds = tr.find_all("td", recursive=False)
        if len(tds) < 5 or not re.search(r"\([A-Z]\d{3}\)", _txt(tds[2])):
            continue
        no = _int(_txt(tds[1]))
        secs = []
        for td in tds[3:]:
            ps = td.find_all("p", recursive=False)
            if len(ps) < 2:
                continue
            first = (ps[1].find(string=True, recursive=False) or "").strip()
            secs.append(_float(first))
        if no is not None and secs and all(s for s in secs):
            out[no] = secs
    return out


def attach_sectionals(race: dict):
    try:
        sect = fetch_sectionals(date.fromisoformat(race["date"]), race["race_no"])
    except RuntimeError:
        return
    for x in race["runners"]:
        if x.get("no") in sect:
            x["sect"] = sect[x["no"]]
    race["has_sect"] = bool(sect)


def fetch_meeting_results(race_date: date, with_sectionals: bool = True):
    """Fetch every race result for a meeting. Returns list of race dicts (maybe empty)."""
    ds = race_date.strftime("%Y/%m/%d")
    html = get("LocalResults.aspx", RaceDate=ds)
    meetings = _race_links(html, "localresults")
    venue = None
    nos = set()
    for (d, v), n in meetings.items():
        if d == ds:
            venue, nos = v, n
    if not venue:
        return []
    nos.add(1)

    def one(n):
        page = html if n == 1 else get("LocalResults.aspx", RaceDate=ds, Racecourse=venue, RaceNo=n)
        r = _parse_result_race(BeautifulSoup(page, "html.parser"), venue, race_date, n)
        if r and r["runners"]:
            if with_sectionals:
                attach_sectionals(r)
            return r
        return None

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        races = [r for r in pool.map(one, sorted(nos)) if r]
    return races


# --------------------------------------------------------------------------- race card

_CARD_COLS = {
    "馬匹編號": "no", "6次近績": "last6", "馬名": "horse", "烙號": "horse_code",
    "負磅": "weight", "騎師": "jockey", "檔位": "draw", "練馬師": "trainer",
    "評分": "rating", "評分+/-": "rating_change", "排位體重": "body_weight",
    "排位體重+/-": "body_weight_change", "最佳時間": "best_time", "馬齡": "age",
    "性別": "sex", "上賽距今日數": "days_since", "配備": "gear", "優先參賽次序": "priority",
    "今季獎金": "season_stakes", "父系": "sire",
}


def next_meeting():
    """(date, venue, [race_nos]) of the meeting currently shown on the race card page."""
    html = get("RaceCard.aspx")
    meetings = _race_links(html, 'RaceCard|localtrackwork|racereportext|"')
    if not meetings:
        return None
    (d, v), nos = sorted(meetings.items())[0]
    return datetime.strptime(d, "%Y/%m/%d").date(), v, sorted(nos | {1}), html


def _parse_card_race(html: str, race_date: date, venue: str, race_no: int):
    soup = BeautifulSoup(html, "html.parser")
    info = {"date": race_date.isoformat(), "venue": venue, "race_no": race_no}
    for div in soup.select("div.f_fs13"):
        t = div.get_text("\n", strip=True)
        if re.search(r"第\s*\d+\s*場", t) and "米" in t:
            lines = [l.strip() for l in t.split("\n") if l.strip()]
            info["race_name"] = re.sub(r"^第\s*\d+\s*場\s*-\s*", "", lines[0])
            joined = " ".join(lines)
            m = re.search(r"(\d{1,2}:\d{2})", joined)
            info["post_time"] = m.group(1) if m else ""
            info["distance"] = _int(re.search(r"(\d{3,4})米", joined).group(1))
            info["surface"] = "AWT" if "全天候" in joined else "TURF"
            m = re.search(r'"([^"]+)"\s*賽道', joined)
            info["course"] = m.group(1) if m else ""
            m = re.search(r"\d{3,4}米,\s*([^\s,]+)", joined)
            info["going"] = m.group(1) if m else ""
            m = re.search(r"評分:\s*([\d\-]+)", joined)
            info["rating_band"] = m.group(1) if m else ""
            m = re.search(r"(第.班|[一二三]級賽|新馬賽|四歲馬賽|條件賽)", joined)
            info["class"] = m.group(1) if m else ""
            break
    table = soup.select_one("#racecardlist table.starter")
    if not table:
        return None
    heads = [_txt(td) for td in table.select("thead td")]
    runners = []
    for tr in table.select("tbody tr"):
        tds = tr.find_all("td", recursive=False)
        if len(tds) < len(heads) - 2:
            continue
        r = {}
        for h, td in zip(heads, tds):
            key = _CARD_COLS.get(h)
            if not key:
                continue
            val = _txt(td)
            if key == "jockey":
                r["jockey_id"] = _code_from_href(td.find("a"), "jockeyid")
                m = re.search(r"\(([-+]?\d+)\)", val)
                r["allowance"] = int(m.group(1)) if m else 0
                val = re.sub(r"\s*\([-+]?\d+\)", "", val)
            elif key == "trainer":
                r["trainer_id"] = _code_from_href(td.find("a"), "trainerid")
            elif key == "horse":
                r["horse_id"] = _code_from_href(td.find("a"), "horseid")
            if key in ("no", "weight", "draw", "rating", "rating_change", "body_weight",
                       "body_weight_change", "age", "days_since"):
                val = _int(val)
            r[key] = val
        # results report the weight actually carried, so apply the apprentice claim here too
        if r.get("weight") is not None:
            r["handicap_weight"] = r["weight"]
            r["weight"] = r["weight"] + (r.get("allowance") or 0)
        # scratched horses stay on the card; HKJC greys them out / marks them 退出
        row_txt = _txt(tr)
        style = (tr.get("style") or "") + " ".join(tr.get("class") or [])
        r["scratched"] = ("退出" in row_txt or "line-through" in style
                          or "grey" in style.lower() or r.get("draw") is None)
        if r.get("no") is not None and r.get("horse"):
            runners.append(r)
    info["runners"] = runners
    return info


def fetch_racecard(race_date: date | None = None, venue: str | None = None):
    """Fetch the full race card. Defaults to the next meeting shown by HKJC."""
    nm = next_meeting()
    if not nm:
        return None
    d, v, nos, first_html = nm
    if race_date and (race_date != d or (venue and venue != v)):
        ds = race_date.strftime("%Y/%m/%d")
        first_html = get("RaceCard.aspx", RaceDate=ds, Racecourse=venue or v, RaceNo=1)
        d, v = race_date, venue or v
        found = _race_links(first_html, 'RaceCard|localtrackwork|racereportext|"').get((ds, v))
        nos = sorted((found or set()) | {1})
    races = []
    for n in nos:
        html = first_html if n == 1 else get(
            "RaceCard.aspx", RaceDate=d.strftime("%Y/%m/%d"), Racecourse=v, RaceNo=n)
        r = _parse_card_race(html, d, v, n)
        if r and r["runners"]:
            races.append(r)
    return {"date": d.isoformat(), "venue": v, "races": races,
            "fetched_at": datetime.utcnow().isoformat() + "Z"}


if __name__ == "__main__":
    card = fetch_racecard()
    print(json.dumps(card["races"][0], ensure_ascii=False, indent=1)[:2000])
