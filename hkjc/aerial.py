"""Where every horse was at four points of a race, read from HKJC's overhead pictures.

    python -m hkjc.aerial                # recent meetings whose pictures have not been read yet
    python -m hkjc.aerial folder <dir>   # pictures already saved as <date>_<venue>_R<n>.jpg

Since the 2026/27 season HKJC publishes, for each race, a picture drawn from its tracking data
(模擬鳥瞰圖): four views from above, one each at about 800 m, 400 m and 200 m to go and at the
finish, with a numbered blue circle for every horse. It shows what the result sheet does not:
how far from the rail each horse raced and which horses had others in front and beside them.
The picture appears the day after the meeting.

This module only reads the pictures and stores the positions (data/aerial/<meeting>.json).
It judges nothing. Each race holds four panels of [horse number, x, y, fit, seen]: x and y are
the circle's centre in the 800 x 450 panel, fit how well the number matched (0 to 1), seen the
share of the circle not hidden behind another. The pictures themselves are HKJC's and are not
kept.

How it reads, with numpy and Pillow only (no AI service): a circle's dark edge votes for its
centre, so circles lying over one another are still found; the white number inside is compared
with the fourteen shapes in aerial_digits.json, only over the part that can be seen; and each
panel gives every runner at most one circle.

Checked on all 186 races of 6 Sep to 11 Nov 2026 (tools/aerial_check.py prints this): 96% of
horse positions read (99% at 400 m and 200 m, 91% at 800 m, 95% at the finish); the horse
nearest the line was the official winner in 186 of 186, and 92% of horses stood at exactly
their official place. What is not read is not in the picture: back-markers outside the frame,
or a circle behind the corner label. HKJC notes that the positions can be off where phone
signals disturb the tracking.
"""
from __future__ import annotations

import io
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from . import scraper, store
from .backfill import today_hkt
from .model import started

FOLDER = store.DATA / "aerial"
DIGITS_FILE = Path(__file__).with_name("aerial_digits.json")
HOST = "https://consvc.hkjc.com/general/-/media/Sites/JCRW/RaceResult"
FIRST = "2026-09-06"   # first meeting with these pictures
RETRY_DAYS = 10        # a meeting whose pictures never came is left alone after this

WIDTH, PANEL, PANELS = 800, 450, 4
R = 20                 # circle radius in pixels
G = 15                 # half-size of the square holding the number (inside the circle's edge)
PAD = 16               # a circle cut by the panel's edge can have its centre this far outside
MIN_EDGE, MIN_BODY, MIN_IN_FRAME, MIN_GAP = 0.30, 0.80, 0.45, 9.0
WEAK_FIT = 0.85        # below this the number is a guess


# ------------------------------------------------------------------ the picture

def picture_url(day: str, race_no: int) -> str:
    y = int(day[:4])
    first = y if day[5:7] >= "09" else y - 1      # a season runs September to July
    ymd = day.replace("-", "")
    return f"{HOST}/{first % 100:02d}_{(first + 1) % 100:02d}/{ymd}/{ymd}R{race_no}_Aerial_L.jpg"


def fetch_picture(day: str, race_no: int) -> bytes | None:
    """The picture's bytes, or None while HKJC does not serve it (it answers with a redirect
    to an error page). Network faults raise RuntimeError, like scraper.get."""
    import requests
    scraper._pace()
    try:
        r = scraper._session().get(picture_url(day, race_no), timeout=30, allow_redirects=False)
    except requests.RequestException as e:
        raise RuntimeError(f"picture of {day} race {race_no}: {type(e).__name__}") from e
    if r.status_code == 200 and r.headers.get("Content-Type", "").startswith("image") and r.content[:2] == b"\xff\xd8":
        return r.content
    return None


def decode(data: bytes) -> np.ndarray | None:
    """The picture as an 800 x 1800 RGB array, or None if it is not one of these pictures.
    Two meetings (6 Sep and 25 Oct 2026) were published at half size; those are enlarged."""
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None
    w, h = im.size
    if w * PANEL * PANELS != h * WIDTH or w < WIDTH // 2:
        return None
    if (w, h) != (WIDTH, PANEL * PANELS):
        im = im.resize((WIDTH, PANEL * PANELS), Image.BICUBIC)
    return np.asarray(im).astype(np.int16)


def masks(img):
    """(circle blue, the circle's darker edge, white of a number) as 0/1 pictures."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    fill = (b > 185) & (r < 70) & (b - g > 60)
    edge = (b > 95) & (b <= 185) & (r < 60) & (b - g > 28) & (b - r > 70)
    white = (r > 120) & (g > 150) & (b > 190)
    return fill, edge, white


# ------------------------------------------------------------------ finding the circles

def _shape(lo, hi):
    n = int(hi) + 2
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    d = np.hypot(yy, xx)
    return ((d >= lo) & (d <= hi)).astype(np.float32)


RING = _shape(R - 2.0, R + 0.8)      # where a circle's edge lies
DISK = _shape(0, R - 3.0)            # its inside


def _match(picture: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """For every pixel, how much of `kernel` centred there falls on `picture` (both 0/1)."""
    h, w = picture.shape
    kh, kw = kernel.shape
    size = (h + kh - 1, w + kw - 1)
    f = np.fft.rfft2(picture.astype(np.float32), size) * np.conj(np.fft.rfft2(kernel, size))
    return np.roll(np.fft.irfft2(f, size), (kh // 2, kw // 2), axis=(0, 1))[:h, :w]


def find_circles(fill, edge, white) -> list[tuple]:
    """Circles of one panel, strongest edge first: [(x, y, edge share)].

    Shares are of the part inside the panel, so a circle cut by the frame still counts. A
    centre can lie up to PAD outside the panel."""
    pad = lambda m: np.pad(m, PAD)
    frame = pad(np.ones(fill.shape, dtype=bool))
    ring_in, disk_in = _match(frame, RING), _match(frame, DISK)
    votes = _match(pad(edge), RING) / np.maximum(ring_in, 1)
    body = _match(pad(fill | edge | white), DISK) / np.maximum(disk_in, 1)
    score = np.where((body >= MIN_BODY) & (disk_in >= MIN_IN_FRAME * DISK.sum()), votes, 0.0)
    yy, xx = np.mgrid[0:score.shape[0], 0:score.shape[1]]
    found = []
    while len(found) < 40:
        y, x = divmod(int(score.argmax()), score.shape[1])
        if score[y, x] < MIN_EDGE:
            break
        found.append((float(x - PAD), float(y - PAD), float(score[y, x])))
        y0, x0 = max(0, y - 40), max(0, x - 40)
        near = np.hypot(yy[y0:y + 41, x0:x + 41] - y, xx[y0:y + 41, x0:x + 41] - x) < MIN_GAP
        score[y0:y + 41, x0:x + 41][near] = 0
    return found


# ------------------------------------------------------------------ reading the numbers

def digit_shapes() -> dict[int, np.ndarray]:
    doc = json.loads(DIGITS_FILE.read_text(encoding="utf-8"))
    return {int(no): np.array([[int(c) for c in row] for row in rows], dtype=np.float32) / 9
            for no, rows in doc["digits"].items()}


_YY, _XX = np.mgrid[-G:G + 1, -G:G + 1]
_INSIDE = np.hypot(_YY, _XX) <= G
_SHIFTS = [(dy, dx) for dy in (-2, -1, 0, 1, 2) for dx in (-2, -1, 0, 1, 2)]


def _square(white, x, y):
    """The number of the circle at (x, y): its white pixels, and which pixels are in the panel."""
    xi, yi = int(round(x)), int(round(y))
    got = np.zeros((2 * G + 1, 2 * G + 1), dtype=bool)
    there = np.zeros_like(got)
    y0, x0 = yi - G, xi - G
    ya, yb, xa, xb = max(0, y0), min(white.shape[0], yi + G + 1), max(0, x0), min(white.shape[1], xi + G + 1)
    if ya < yb and xa < xb:
        got[ya - y0: yb - y0, xa - x0: xb - x0] = white[ya:yb, xa:xb]
        there[ya - y0: yb - y0, xa - x0: xb - x0] = True
    return got & _INSIDE, there & _INSIDE


def _covers(edge, a, b) -> float:
    """Share of circle a's edge, where it runs across circle b, that is drawn: high when a
    lies on top of b."""
    ang = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    hit = tot = 0
    for rad in (R - 1.5, R - 0.5):
        xs, ys = a[0] + rad * np.cos(ang), a[1] + rad * np.sin(ang)
        over = np.hypot(xs - b[0], ys - b[1]) < R - 4
        xi, yi = np.round(xs[over]).astype(int), np.round(ys[over]).astype(int)
        ok = (xi >= 0) & (yi >= 0) & (xi < edge.shape[1]) & (yi < edge.shape[0])
        tot += int(ok.sum())
        hit += int(edge[yi[ok], xi[ok]].sum())
    return hit / tot if tot else 0.0


def assign(cost: list[list[float]]) -> list[int]:
    """Give each row its own column at the least total cost (rows <= columns): the column of
    each row. The Hungarian method; small enough here not to need a library."""
    n, m = len(cost), len(cost[0])
    inf = float("inf")
    u, v, p, way = [0.0] * (n + 1), [0.0] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        least, used = [inf] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < least[j]:
                        least[j], way[j] = cur, j0
                    if least[j] < delta:
                        delta, j1 = least[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    least[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    out = [0] * n
    for j in range(1, m + 1):
        if p[j]:
            out[p[j] - 1] = j - 1
    return out


def read_panel(fill, edge, white, starters: list[int], shapes: dict) -> list[list]:
    """One panel -> [[horse number, x, y, fit, seen], ...] for the circles found in it."""
    circles = find_circles(fill, edge, white)[:len(starters)]
    if not circles:
        return []
    cost = []
    seen_share = []
    for i, c in enumerate(circles):
        got, visible = _square(white, c[0], c[1])
        for j, o in enumerate(circles):
            if i != j and np.hypot(o[0] - c[0], o[1] - c[1]) < 2 * R and _covers(edge, o, c) > _covers(edge, c, o):
                visible = visible & (np.hypot(_XX + c[0] - o[0], _YY + c[1] - o[1]) > R + 1)
        n, whole = int(visible.sum()), int(_INSIDE.sum())
        seen_share.append(n / whole)
        row = []
        for no in starters:
            shape = shapes.get(no)
            if shape is None or not n:
                row.append(0.25)
                continue
            best = max(1 - float(np.abs(got[visible] - np.roll(np.roll(shape, dy, 0), dx, 1)[visible]).sum()) / n
                       for dy, dx in _SHIFTS)
            # the hidden part says nothing: count it as a so-so match, so a circle seen whole
            # is trusted before one that is mostly covered
            row.append(1 - (best * n + 0.75 * (whole - n)) / whole)
        cost.append(row)
    cols = assign(cost)
    return [[starters[k], round(c[0], 1), round(c[1], 1), round(1 - cost[i][k], 3), round(seen_share[i], 3)]
            for i, (c, k) in enumerate(zip(circles, cols))]


def read_picture(img: np.ndarray, starters: list[int], shapes: dict | None = None) -> list[list[list]]:
    """The four panels of one picture. starters: the numbers of the horses that ran."""
    shapes = shapes or digit_shapes()
    fill, edge, white = masks(img)
    starters = sorted(starters)
    return [read_panel(fill[p * PANEL:(p + 1) * PANEL], edge[p * PANEL:(p + 1) * PANEL],
                       white[p * PANEL:(p + 1) * PANEL], starters, shapes) for p in range(PANELS)]


# ------------------------------------------------------------------ keeping what was read

def _ran(race: dict) -> list[int]:
    return sorted(x["no"] for x in race.get("runners", []) if x.get("no") is not None and started(x))


def _entry(img, race, shapes) -> dict:
    panels = read_picture(img, _ran(race), shapes)
    return {"read_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "ran": len(_ran(race)),
            "panels": panels}


def _save(path: Path, doc: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def load(day: str, venue: str) -> dict:
    """{race number: [four panels]} for a meeting (empty if none read)."""
    doc = store.read_json(FOLDER / f"{day}_{venue}.json", {}) or {}
    return {int(k): v["panels"] for k, v in (doc.get("races") or {}).items()}


def update(log=print, fetch=None) -> tuple[int, int]:
    """Read the pictures of finished meetings of the last RETRY_DAYS days that are not stored
    yet. Returns (races read, races whose picture is not there yet). Never raises for one race."""
    fetch = fetch or fetch_picture
    today = today_hkt()
    shapes = digit_shapes()
    read = waiting = 0
    for path in sorted(store.RESULTS.glob("20*_*.json")):
        day, venue = path.stem.split("_")
        if venue == "NONE" or day < FIRST or not 0 < (today - date.fromisoformat(day)).days <= RETRY_DAYS:
            continue
        out = FOLDER / path.name
        doc = store.read_json(out) or {"date": day, "venue": venue, "races": {}}
        before = len(doc["races"])
        for race in store.read_json(path, []) or []:
            key = str(race["race_no"])
            if key in doc["races"]:
                continue
            try:
                data = fetch(day, race["race_no"])
            except RuntimeError as e:
                log(f"  ! {e}")
                data = None
            img = decode(data) if data else None
            if img is None:
                waiting += 1
                continue
            doc["races"][key] = _entry(img, race, shapes)
            read += 1
        if len(doc["races"]) > before:
            doc["races"] = dict(sorted(doc["races"].items(), key=lambda kv: int(kv[0])))
            _save(out, doc)
            got = sum(len(p) for r in doc["races"].values() for p in r["panels"])
            want = sum(r["ran"] * PANELS for r in doc["races"].values())
            log(f"  {path.stem}: {len(doc['races'])} races, {got}/{want} positions read")
    return read, waiting


def read_folder(folder: Path, log=print) -> int:
    """Read pictures saved by hand as <date>_<venue>_R<n>.jpg (used once, for the meetings
    before this module existed). Stored reads are kept."""
    shapes = digit_shapes()
    by_meeting: dict = {}
    for p in sorted(folder.glob("20*_*_R*.jpg")):
        day, venue, rn = p.stem.split("_")
        by_meeting.setdefault((day, venue), []).append((int(rn[1:]), p))
    done = 0
    for (day, venue), pictures in sorted(by_meeting.items()):
        races = {r["race_no"]: r for r in store.read_json(store.result_path(day, venue), []) or []}
        out = FOLDER / f"{day}_{venue}.json"
        doc = store.read_json(out) or {"date": day, "venue": venue, "races": {}}
        for n, p in sorted(pictures):
            if str(n) in doc["races"] or n not in races:
                continue
            img = decode(p.read_bytes())
            if img is None:
                log(f"  ! {p.name}: not an overhead picture")
                continue
            doc["races"][str(n)] = _entry(img, races[n], shapes)
            done += 1
        doc["races"] = dict(sorted(doc["races"].items(), key=lambda kv: int(kv[0])))
        _save(out, doc)
        log(f"  {day}_{venue}: {len(doc['races'])} races")
    return done


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "folder":
        print("read", read_folder(Path(sys.argv[2])), "pictures")
        sys.exit()
    got, wait = update()
    print(f"read {got} races, {wait} pictures not there yet")
