"""Overhead-picture test, registered 2026-11-14 before any of its races were run.

Question: HKJC's overhead pictures (hkjc/aerial.py) show two things about a run that the
result sheet does not, and that the system could only guess from the wording of comments:
how far out from the rail a horse raced round the turn, and whether it was shut in behind
other horses in the straight. Does either, measured on a horse's last run, say more about its
next race than the win odds at our freeze?

Rules, fixed now and not to be changed after seeing results:
  * Measures, taken from the stored positions of one race (pixels; a length is 41 px and a
    metre 16.4 px, from the finish margins of 186 races):
      wide    metres outside the innermost horse among those within 2 lengths in front or
              behind, averaged over the panels at 800 m and 400 m to go where the horse was
              read. Each panel has a direction of travel that depends only on the course and
              distance (DIRECTION, found from the official running order of the races of 6 Sep
              to 11 Nov 2026; the inside rail is on the right of it, both courses being
              right-handed). A kind of race not in the table gives no value, nor does the
              1000 m straight course at Sha Tin, which has no turn. Capped at WIDE_CAP.
      boxed   at the 200 m panel (always the home straight, horses running to the left): a
              horse within 2 lengths in front and within half a length to either side of its
              line, AND a horse on its outside (0.3 to 1.5 lengths out, from 1 length behind
              to 2 in front), AND the same on its inside or, failing that, the horse is itself
              within half a length of the innermost of the field (the rail).
    A horse's own measure needs its number read surely (fit >= aerial.WEAK_FIT); every circle
    counts as a horse in the way of others. wide needs at least 4 horses in the panel.
  * Candidates, each judged alone: wide_last and boxed_last = the measure on the horse's most
    recent run before race day (that run only: if it has no picture, or the horse was not in
    it, the horse has no value). wide_last is taken as the difference from the average of the
    runners that have a value; a runner without one counts as average (0). boxed_last is 1 or
    0, a runner without a value 0. A race is used for a candidate only if at least half of its
    runners have a value.
  * Races: every race from FROM (2026-11-15) with a result, exactly one winner and a "pre" odds
    snapshot of at least 7 priced runners, as in livetest.py. Base, cross-validation, looks at
    LOOKS (300, 600) races, pass mark (z >= 2.5 with a shuffled placebo below 1.0) and the
    closing at the last look are those of livetest.py, whose code does the judging.
  * A pass is reported for the user to decide on. It never changes the ranking by itself.
  * DIRECTION may gain a line for a kind of race once 5 pictures of it exist, found the same
    way; lines already there never change.

What was checked before registering, on the races of 6 Sep to 11 Nov 2026 and without looking
at how any horse ran afterwards: the measures against what the race caller and the stewards
wrote about the same runs. Runs written up as wide (走外疊, 沒有遮擋, 三疊...) measured a
median 4.5 m against 1.7 m for the rest; of the runs flagged boxed (11.4% of all), 27.7% were
written up as 未能望空, 受困 or 受阻 against 6.9% of the rest (tools/trips_check.py prints
this). The thresholds were set on that agreement and on how often they fire, never on results.

Limits. A picture is four moments, so trouble between them is missed. The direction is one
number per panel although the track bends, which can misplace a back-marker on a tight turn by
about a metre. HKJC says the positions themselves can be off. And the "wide" told by comments
is already a model factor (run_wide) whose weight beside the market is nil, so the likeliest
result is that these two show nothing either; the test is how to find out.
"""
from __future__ import annotations

import math

import numpy as np

from . import aerial, livetest, store
from .model import started

REGISTERED = "2026-11-14"
FROM = "2026-11-15"
LOOKS, BLOCKS, Z_PASS, PLACEBO_MAX, MIN_PRICED = (300, 600), 5, 2.5, 1.0, 7
CANDIDATES = ["wide_last", "boxed_last"]
NAMES_ZH = {"wide_last": "上仗轉彎蝕位(鳥瞰圖)", "boxed_last": "上仗直路受困(鳥瞰圖)"}

LENGTH = 41.0                 # pixels
METRE = LENGTH / 2.5
WIDE_WINDOW = 2.0             # lengths in front or behind that count as alongside
WIDE_CAP = 8.0                # metres
MIN_FIELD = 4
STRAIGHT = 180.0              # the 200 m panel: horses run to the left

# (venue, surface, distance): direction of travel in the picture at the 800 m and 400 m panels,
# in degrees (0 = to the right, 90 = down the picture)
DIRECTION = {
    ("HV", "TURF", 1000): (352, 10), ("HV", "TURF", 1200): (2, 1),
    ("HV", "TURF", 1650): (358, 2), ("HV", "TURF", 1800): (357, 9),
    ("ST", "AWT", 1200): (357, 156), ("ST", "AWT", 1650): (354, 165),
    ("ST", "TURF", 1200): (38, 185), ("ST", "TURF", 1400): (31, 183),
    ("ST", "TURF", 1600): (29, 186), ("ST", "TURF", 1800): (44, 181),
}


# ------------------------------------------------------------------ the measures

def _frame(panel, angle) -> dict:
    """{horse: (s, u)} in pixels: s grows towards the front, u towards the inside rail."""
    a = math.radians(angle)
    return {r[0]: (r[1] * math.cos(a) + r[2] * math.sin(a), -r[1] * math.sin(a) + r[2] * math.cos(a))
            for r in panel}


def wide(panel, angle) -> dict:
    """{horse: metres outside the innermost horse among those alongside}; {} for a thin field."""
    f = _frame(panel, angle)
    if len(f) < MIN_FIELD:
        return {}
    return {h: (max(u2 for s2, u2 in f.values() if abs(s2 - s) <= WIDE_WINDOW * LENGTH) - u) / METRE
            for h, (s, u) in f.items()}


def boxed(panel) -> dict:
    """{horse: True if shut in at the 200 m panel}."""
    f = _frame(panel, STRAIGHT)
    if not f:
        return {}
    innermost = max(u for _, u in f.values())
    out = {}
    for h, (s, u) in f.items():
        others = [((s2 - s) / LENGTH, (u2 - u) / LENGTH) for k, (s2, u2) in f.items() if k != h]
        ahead = any(0 < ds <= 2.0 and abs(du) <= 0.5 for ds, du in others)
        outside = any(-1.5 <= du < -0.3 and -1.0 <= ds <= 2.0 for ds, du in others)
        inside = any(0.3 < du <= 1.5 and -1.0 <= ds <= 2.0 for ds, du in others) \
            or u >= innermost - 0.5 * LENGTH
        out[h] = ahead and outside and inside
    return out


def run_measures(race: dict, panels: list) -> dict:
    """{horse number: {"wide": metres or None, "boxed": True/False or None}} for one race."""
    sure = [{r[0] for r in p if r[3] >= aerial.WEAK_FIT} for p in panels]
    angles = DIRECTION.get((race.get("venue"), race.get("surface"), race.get("distance")))
    turn = [wide(panels[p], angles[p]) for p in (0, 1)] if angles else [{}, {}]
    shut = boxed(panels[2])
    out = {}
    for h in {r[0] for p in panels for r in p}:
        w = [turn[p][h] for p in (0, 1) if h in turn[p] and h in sure[p]]
        out[h] = {"wide": min(WIDE_CAP, sum(w) / len(w)) if w else None,
                  "boxed": bool(shut[h]) if h in shut and h in sure[2] else None}
    return out


# ------------------------------------------------------------------ a horse's last run

def runs_by_horse(races: list[dict]) -> dict:
    """{horse code: [(date, venue, race number, horse number), ...]} oldest first."""
    out: dict = {}
    for r in sorted(races, key=lambda r: (r["date"], r["race_no"])):
        for x in r["runners"]:
            if x.get("horse_code") and x.get("no") is not None and started(x):
                out.setdefault(x["horse_code"], []).append((r["date"], r["venue"], r["race_no"], x["no"]))
    return out


class LastRun:
    """Measures of a horse's most recent run before a day, read from data/aerial as needed."""

    def __init__(self, races: list[dict]):
        self.runs = runs_by_horse(races)
        self.races = {(r["date"], r["venue"], r["race_no"]): r for r in races}
        self._meetings: dict = {}

    def _measures(self, day, venue, race_no) -> dict:
        key = (day, venue)
        if key not in self._meetings:
            self._meetings[key] = {n: run_measures(self.races[(day, venue, n)], p)
                                   for n, p in aerial.load(day, venue).items() if (day, venue, n) in self.races}
        return self._meetings[key].get(race_no, {})

    def before(self, code: str, day: str) -> dict:
        """{"wide": ..., "boxed": ...} of the horse's last run before `day`; both None if that
        run has no picture or the horse is not in it."""
        earlier = [r for r in self.runs.get(code, []) if r[0] < day]
        if not earlier:
            return {"wide": None, "boxed": None}
        d, venue, race_no, no = earlier[-1]
        return self._measures(d, venue, race_no).get(no, {"wide": None, "boxed": None})


# ------------------------------------------------------------------ the test

def _candidate(values: list, centre: bool):
    known = [v for v in values if v is not None]
    if len(known) * 2 < len(values) or not known:
        return None
    mid = sum(float(v) for v in known) / len(known) if centre else 0.0
    return np.array([float(v) - mid if v is not None else 0.0 for v in values])


def build(races: list[dict] | None = None) -> list[dict]:
    """One sample per usable race, shaped like livetest.build: {"key", "y", "logq", "f"}."""
    races = store.load_all_results() if races is None else races
    last = LastRun(races)
    out = []
    for path in sorted((store.DATA / "odds").glob("20*_*.json")):
        day, venue = path.stem.split("_")
        if day < FROM:
            continue
        saved = store.read_json(path, {}) or {}
        for k, book in sorted((saved.get("pools") or {}).items(), key=lambda kv: int(kv[0])):
            res = last.races.get((day, venue, int(k)))
            pre = {int(n): o for n, o in ((book.get("pre") or {}).get("win") or {}).items() if o}
            if not res or len(pre) < MIN_PRICED:
                continue
            runners = [x for x in res["runners"] if started(x) and x.get("no") in pre]
            winners = [i for i, x in enumerate(runners) if x.get("place") == 1]
            if len(winners) != 1:
                continue
            q = livetest._probs(pre, [x["no"] for x in runners])
            seen = [last.before(x.get("horse_code"), day) for x in runners]
            out.append({"key": (day, venue, int(k)), "y": winners[0], "logq": np.log(q),
                        "f": {"wide_last": _candidate([m["wide"] for m in seen], centre=True),
                              "boxed_last": _candidate([m["boxed"] for m in seen], centre=False)}})
    return out


def status(races: list[dict] | None = None) -> dict:
    """Races collected per candidate and, once a look is reached, its verdict."""
    samples = build(races)
    out = {"registered": REGISTERED, "from": FROM, "candidates": {}}
    for c in CANDIDATES:
        use = [s for s in samples if s["f"][c] is not None]
        n = len(use)
        look = max([x for x in LOOKS if n >= x], default=None)
        row = {"name": NAMES_ZH[c], "races": n, "next": next((x for x in LOOKS if x > n), None)}
        if look:
            res = livetest.judge(use[:look], c)
            res["placebo_z"] = livetest.judge(use[:look], c, placebo=True)["z"]
            passed = res["gain"] > 0 and res["z"] >= Z_PASS and res["placebo_z"] < PLACEBO_MAX
            row.update(look=look, **res, passed=passed,
                       verdict="過關" if passed else ("未過關,完結" if look == LOOKS[-1] else "未過關"))
        out["candidates"][c] = row
    return out


def meeting_table(day: str, venue: str) -> list[dict]:
    """Every horse's measures in one stored meeting, for a look by eye:
    [{"race_no", "no", "horse", "wide", "boxed"}]."""
    races = {r["race_no"]: r for r in store.read_json(store.result_path(day, venue), []) or []}
    rows = []
    for n, panels in sorted(aerial.load(day, venue).items()):
        if n not in races:
            continue
        m = run_measures(races[n], panels)
        for x in races[n]["runners"]:
            if x.get("no") in m:
                rows.append({"race_no": n, "no": x["no"], "horse": x.get("horse"), **m[x["no"]]})
    return rows


if __name__ == "__main__":
    import json
    print(json.dumps(status(), ensure_ascii=False, indent=1))
