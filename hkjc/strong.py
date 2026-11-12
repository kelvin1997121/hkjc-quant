"""強組 marks for the page: horses whose last 2 runs include a strong race. Reference only.

A strong race, for the page:
- it was run fast: the winner's speed figure, as known before that race, at least 3 points
  (0.3 s per km) above the mean winner figure of up to the previous 200 races of its class;
- and at least 3 of its beaten horses (not its winner, not the horse being marked) finished
  in the first 3 in their very next start before today.

The registered test (hkjc/oldbank.py) used a looser version: 2 beaten horses within their next 3
starts, last 3 runs. That marked about 42% of runners, so the user chose this stricter bar for
the page (about 13%). The test found horses from such races win about what their odds say
(533 wins against 526.9 expected), so the mark changes no ranking or bet. A look (not a test)
at this stricter mark on the same 2020-03..2022-08 races: 2,735 marked runners won 220 races
against 230.4 expected from their odds.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

FAST, MIN_PLACED, AFTER_STARTS, LAST_N = 3.0, 3, 1, 2   # the user's bar for the page
CLASS_RACES, CLASS_MIN = 200, 20                        # race figure as in hkjc/oldbank.py


class Tracker:
    """Kept on model.State; note() runs at the start of State.update, before the race is added."""

    def __init__(self):
        self.races = {}                    # race key -> {"rel": figure minus class par, "runners"}
        self.run_key = {}                  # (horse code, date) -> race key
        self.class_figs = defaultdict(list)
        self.raw = {}                      # (horse code, date) -> HKJC code when no place (PU, DNF...)

    def note(self, state, race):
        from .model import class_num, started
        starters = [x for x in race["runners"] if x.get("horse_code") and started(x)]
        win = next((x for x in starters if x.get("place") == 1), None)
        fig = state.speed_fig(race, win.get("time"), win.get("weight")) if win else None
        cls = class_num(race.get("class"))
        par = self.class_figs[cls][-CLASS_RACES:] if cls is not None else []
        rel = fig - float(np.mean(par)) if fig is not None and len(par) >= CLASS_MIN else None
        if fig is not None and cls is not None:
            self.class_figs[cls].append(fig)
        key = (race["date"], race["venue"], race["race_no"])
        self.races[key] = {"rel": rel, "runners": [
            (x["horse_code"], x.get("place") if isinstance(x.get("place"), int) else None) for x in starters]}
        for x in starters:
            self.run_key[(x["horse_code"], race["date"])] = key
            if not isinstance(x.get("place"), int) and x.get("place_raw"):
                self.raw[(x["horse_code"], race["date"])] = x["place_raw"]


def marks(state, code, today) -> list[dict]:
    """Strong races among the horse's last LAST_N runs before `today`, newest first."""
    tr = getattr(state, "strong", None)
    if tr is None or not code:
        return []
    out = []
    runs = [r for r in state.horse.get(code, []) if r["date"] < today][-LAST_N:]
    for run in reversed(runs):
        key = tr.run_key.get((code, run["date"]))
        info = tr.races.get(key) if key else None
        if not info or info["rel"] is None or info["rel"] < FAST:
            continue
        placed = 0
        for c, place in info["runners"]:
            if c == code or place == 1:
                continue
            later = [h for h in state.horse.get(c, []) if run["date"] < h["date"] < today][:AFTER_STARTS]
            placed += any(h.get("place") and h["place"] <= 3 for h in later)
        if placed >= MIN_PLACED:
            out.append({"date": run["date"], "venue": key[1], "race_no": key[2],
                        "place": run.get("place"), "faster": round(info["rel"] / 10, 2),
                        "placed": placed, "place_raw": tr.raw.get((code, run["date"]))})
    return out
