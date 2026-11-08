"""Quantitative model.

Races are replayed in date order. Before each race every runner gets a
feature vector built only from information known at that moment (no
look-ahead); afterwards the result updates the running state. The same
state then scores today's race card, so live predictions use exactly the
feature pipeline that the weights were trained on.

Factors
  speed      going/weight adjusted speed figures vs. self-calibrated standard times
  form       recency-weighted finishing positions
  style      running style (early position) from 沿途走位
  pace       projected pace of today's field x horse's style
  draw       long-run draw bias for venue / track / course / distance
  today      live bias from races already run at this meeting (style + draw)
  connections jockey / trainer strike rates
  weight, layoff, distance suitability

Weights are fitted with a conditional logit (softmax within each race).
"""
from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from datetime import date
from statistics import median

import numpy as np

from . import reports

FEATURES = [
    "speed_best", "speed_last", "speed_missing", "form", "dist_form",
    "early", "early_x_pace", "early_x_today", "early_x_course",
    "draw_bias", "draw_x_today", "jockey", "trainer", "weight_rel",
    "layoff", "debut", "mkt_rating", "vs_mkt", "closing", "class_drop", "bw_change",
    "perf_best", "perf_last", "handicap_edge", "late_speed", "early_speed",
    "jockey_form", "trainer_form", "combo",
    "rating_move", "trainer_venue", "form_trend",
    "gear_first", "gear_off", "gear_back",
    "run_wide", "trial_score", "trial_count",
]
BASE_FEATURES = list(FEATURES)
# Factors on trial, with the date they were registered. The self-check judges each one only on
# races run after that date (selfcheck.learning) and adds it to FEATURES only if it passes.
TRIAL_FEATURES = {
    "jockey_up": "2026-11-08",   # today's jockey's win rate against the horse's last jockey's
    "awt_pref": "2026-11-08",    # all-weather form against overall form, in all-weather races
}
ALL_FEATURES = BASE_FEATURES + list(TRIAL_FEATURES)


def active_features(cfg):
    """Base factors, plus those the self-check adopted, minus those it retired."""
    cfg = cfg or {}
    return [f for f in BASE_FEATURES if f not in cfg.get("removed", [])] + \
        [f for f in cfg.get("added", []) if f in TRIAL_FEATURES]
# From running comments, incident reports and trials, these three carried the whole gain
# (walk-forward log loss 2.2049 -> 2.1866 on 1,291 races; shuffling trials among horses in
# a race removes it). No gain, dropped: blocked, bad start, pulled, strong or weak finish,
# health and excuse tags from the last run; trial win, margin, jockey, comment tone, debut.
# Also tested on official-rating data and dropped for no walk-forward gain: sire record
# overall and by distance, rating gap to the field, apprentice claim, first-time blinkers.
# Tried with the 4 Nov review and dropped (no walk-forward gain on 1,291 races):
# jockey_venue, first_up, quick_back.
# venue_form (horse's form at this venue/surface) was tried and dropped: it worsened
# walk-forward log loss (2.2199 without vs 2.2220 with).
# Same-day sectional reads (track variant, closing advantage) are shown to the user
# but kept out of the model: as features they worsened walk-forward log loss
# (2.2251 -> 2.2303 on 452 races).
# Nov 2026, 968 development races, dropped: track-variant speed, age, sex, imported with
# experience, distance change, wet-track and all-weather record, weight x distance, beaten
# margin, career record, mark vs last win, body weight; training on only the last 1-2 seasons.
# jockey_up (log win rate of today's jockey minus the horse's last jockey) passed there (+0.0028,
# z 3.35) and vanished when shuffled, but missed on the 323 locked races from 2026-05-20 (+0.0039,
# z 1.60, needed 1.64). Not adopted: model weight stays 0, so the first choice wouldn't change.
# Those 323 races are now used up; retest it on races after 2026-11-08.
SEC_PER_LENGTH = 0.17

_CLASS = {"一級賽": -2, "二級賽": -1, "三級賽": 0, "第一班": 1, "第二班": 2,
          "第三班": 3, "第四班": 4, "第五班": 5}


_CLASS_TOP = {"一級賽": 130, "二級賽": 118, "三級賽": 110, "第一班": 110, "第二班": 100,
              "第三班": 80, "第四班": 60, "第五班": 40}


def band_top(race):
    """Upper rating of the race band, e.g. '60-40' -> 60."""
    m = re.match(r"(\d+)", race.get("rating_band") or "")
    if m:
        return int(m.group(1))
    for k, v in _CLASS_TOP.items():
        if k in (race.get("class") or ""):
            return v
    return None


def est_ratings(race, runners):
    """Official rating per runner; for the few without one, band top minus weight below top."""
    top = band_top(race)
    wts = [x.get("weight") or 0 for x in runners]
    maxw = max(wts) if wts else 0
    out = []
    for x, w in zip(runners, wts):
        if x.get("rating"):
            out.append(float(x["rating"]))
        elif top is not None and w:
            out.append(float(top - (maxw - w)))
        else:
            out.append(None)
    return out


def gear_flags(gear):
    """HKJC gear codes: suffix 1 = first time on, 2 = back on after a break, - = taken off."""
    parts = [g.strip() for g in (gear or "").split("/") if g.strip()]
    return {"first": sum(g.endswith("1") for g in parts),
            "off": sum(g.endswith("-") for g in parts),
            "back": sum(g.endswith("2") for g in parts)}


STYLES = ("領放", "前列", "中游", "後上")


def style_label(early):
    """Running style from average early position (0 = leads, 1 = last)."""
    if early is None:
        return "未知"
    if early <= 0.2:
        return "領放"
    if early <= 0.4:
        return "前列"
    if early <= 0.65:
        return "中游"
    return "後上"


def pace_projection(profs, field):
    """Pre-race pace from how many runners usually lead: (score, n_leaders); >0 = contested."""
    n_lead = sum(1 for p in profs if p["early"] is not None and p["early"] <= 0.25)
    return (n_lead - 0.2 * field) / max(field, 1) * 4, n_lead


def pace_bucket(pace):
    return "快" if pace > 0.5 else "慢" if pace < -0.4 else "中"


def started(x) -> bool:
    """False for withdrawn / void entries, which must not count as a run."""
    if x.get("place"):
        return True
    code = (x.get("place_raw") or "").upper()
    return bool(code) and not code.startswith("W") and code != "VOID"


def class_num(s):
    """Grade as a number, higher = lower grade (easier). None if unknown."""
    for k, v in _CLASS.items():
        if s and k in s:
            return v
    return None


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None

LB_SEC_PER_KM = 0.07          # 1 lb ~ 0.07 s per 1000 m
WARMUP_DAYS = 60
GOOD = {"好地", "快地", "好至快地", "好地至快地"}


def going_key(g: str) -> str:
    return g or "?"


def track_key(r) -> tuple:
    return (r["venue"], r.get("surface", "TURF"), r.get("distance"))


def draw_key(r) -> tuple:
    return (r["venue"], r.get("surface", "TURF"), r.get("course", ""), r.get("distance"))


class RateBook:
    """Smoothed win rate over a rolling window of runs."""

    def __init__(self, window=400, prior=0.08, strength=20):
        self.d = defaultdict(lambda: deque(maxlen=window))
        self.prior, self.k = prior, strength

    def rate(self, key):
        h = self.d.get(key)
        n = len(h) if h else 0
        w = sum(h) if h else 0
        return (w + self.prior * self.k) / (n + self.k)

    def add(self, key, won):
        if key:
            self.d[key].append(1 if won else 0)

    def rate_vs(self, key, base):
        """Recent rate shrunk towards `base` (e.g. the long-run rate), as log ratio."""
        h = self.d.get(key)
        n = len(h) if h else 0
        w = sum(h) if h else 0
        return math.log(((w + base * self.k) / (n + self.k)) / base)


class State:
    def __init__(self):
        self.horse = defaultdict(list)            # code -> runs
        self.winner_times = defaultdict(list)     # track_key -> times on good ground
        self.going_diff = defaultdict(list)       # going -> (time-std)/km on winners
        self.draw = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # key -> draw -> [runs, win, place]
        self.course_early = defaultdict(list)     # draw_key -> early frac of winners
        self.jockey = RateBook()
        self.trainer = RateBook()
        # recent form (top-3 rate, last few weeks) relative to each one's long-run rate
        self.jockey_place = RateBook(window=400, prior=0.24, strength=20)
        self.trainer_place = RateBook(window=400, prior=0.24, strength=20)
        self.jockey_recent = RateBook(window=40, strength=15)
        self.trainer_recent = RateBook(window=30, strength=15)
        self.combo = RateBook(window=150, prior=0.24, strength=15)  # jockey x trainer, top-3
        # trainer's top-3 rate at this venue relative to their overall rate (HV/ST specialists)
        self.trainer_venue = RateBook(window=300, strength=30)
        self.meeting = defaultdict(list)          # (date, venue) -> finished race summaries
        # (venue, surface) -> placed horses with an early position, of them in the front third,
        # placed horses with a draw, of them from the inside third, winners' early position sum,
        # winners. What a normal day looks like, for the page's description of today only.
        self.style_norm = defaultdict(lambda: [0, 0, 0, 0, 0.0, 0])
        self.class_var = defaultdict(list)        # class -> (winner time - std)/km, good ground
        self.late_top_hist = []                   # per race: avg final-section edge of top 3
        # (venue, projected pace) -> running style -> [runners, winners]: what pace really does
        self.pace_style = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        self.trials = reports.load_trials()

    # ---------------------------------------------------------------- helpers
    def std_time(self, r):
        times = self.winner_times.get(track_key(r))
        if not times or len(times) < 3:
            return None
        return median(times[-60:])

    def going_adj(self, going):
        if going in GOOD:
            return 0.0
        d = self.going_diff.get(going_key(going))
        return median(d[-80:]) if d and len(d) >= 3 else 0.0

    def speed_fig(self, r, time, weight):
        std = self.std_time(r)
        if std is None or not time or not r.get("distance"):
            return None
        km = r["distance"] / 1000
        diff = (time - std) / km - self.going_adj(r.get("going"))
        diff -= (weight - 126) * LB_SEC_PER_KM if weight else 0
        return round(-diff * 10, 2)  # 1 point = 0.1 s per km faster than standard

    def draw_bias(self, r, draw):
        stats = self.draw.get(draw_key(r))
        if not stats or not draw:
            return 0.0
        runs = sum(v[0] for v in stats.values())
        places = sum(v[2] for v in stats.values())
        if runs < 50:
            return 0.0
        base = places / runs
        s = stats.get(draw, [0, 0, 0])
        k = 30
        rate = (s[2] + base * k) / (s[0] + k)
        return math.log(rate / base)

    def course_leader_edge(self, r):
        """How far forward winners have been at this course/distance (0.5 = neutral)."""
        e = self.course_early.get(draw_key(r))
        if not e or len(e) < 8:
            return 0.0
        return 0.45 - float(np.mean(e[-80:]))  # >0: leaders favoured

    def today_bias(self, r):
        """Live bias at this meeting: (style_edge, draw_edge) from races already run."""
        done = [m for m in self.meeting.get((r["date"], r["venue"]), [])
                if m["surface"] == r.get("surface", "TURF")]
        if not done:
            return 0.0, 0.0, 0
        style = float(np.mean([m["top_early"] for m in done]))   # avg early frac of top-3
        draw = float(np.mean([m["top_draw"] for m in done]))     # avg draw frac of top-3
        n = len(done)
        shrink = n / (n + 3)
        return (0.45 - style) * shrink, (0.5 - draw) * shrink, n

    def today_read(self, r):
        """Today's finished races on this surface against a normal day, for the page.

        The model's own today_bias above is left as it is (its weights were fitted on it).
        Races whose running positions aren't published yet don't count for running style.
        """
        surface = r.get("surface", "TURF")
        done = [m for m in self.meeting.get((r["date"], r["venue"]), []) if m["surface"] == surface]
        nb = self.style_norm.get((r["venue"], surface))
        out = {"races": len(done),
               "style_races": sum(1 for m in done if m.get("n_early")),
               "n_early": sum(m.get("n_early", 0) for m in done),
               "front": sum(m.get("front", 0) for m in done),
               "n_draw": sum(m.get("n_draw", 0) for m in done),
               "inner": sum(m.get("inner", 0) for m in done),
               "norm_front": nb[1] / nb[0] if nb and nb[0] >= 300 else None,
               "norm_inner": nb[3] / nb[2] if nb and nb[2] >= 300 else None}
        return out

    def course_vs_norm(self, r):
        """How much further forward winners have been at this course/distance than at this
        venue and surface in general (>0: leaders do better here than usual). None if thin."""
        e = self.course_early.get(draw_key(r))
        nb = self.style_norm.get((r["venue"], r.get("surface", "TURF")))
        if not e or len(e) < 20 or not nb or nb[5] < 200:
            return None
        return nb[4] / nb[5] - float(np.mean(e[-80:]))

    def pace_history(self, venue, bucket):
        """Per-style win rate in past races with this projected pace at this venue
        (both venues pooled if the venue has too little history)."""
        book = self.pace_style.get((venue, bucket), {})
        if sum(v[0] for v in book.values()) < 300:
            merged = defaultdict(lambda: [0, 0])
            for (v, b), bk in self.pace_style.items():
                if b == bucket:
                    for s, (n, w) in bk.items():
                        merged[s][0] += n
                        merged[s][1] += w
            book, venue = merged, None
        rates = {s: (book[s][1] / book[s][0], book[s][0]) for s in STYLES
                 if s in book and book[s][0] >= 50}
        return rates, venue

    def today_sectionals(self, r):
        """Live read from today's sectional times on this surface.

        variant: s/km slower (+) or faster (-) than the class-adjusted standard
        late:    how much more than usual the top 3 out-finished the field (s, + = closers)
        """
        done = [m for m in self.meeting.get((r["date"], r["venue"]), [])
                if m["surface"] == r.get("surface", "TURF")]
        var = [m["variant"] for m in done if m.get("variant") is not None]
        late = [m["late_excess"] for m in done if m.get("late_excess") is not None]
        v = float(np.mean(var)) * len(var) / (len(var) + 2) if var else 0.0
        lt = float(np.mean(late)) * len(late) / (len(late) + 2) if late else 0.0
        return v, lt, len(var), len(late)

    # ---------------------------------------------------------------- features
    def runner_profile(self, code, dist, venue=None, surface=None):
        runs = self.horse.get(code, [])
        here = [x["fscore"] for x in runs
                if x["venue"] == venue and x.get("surface", "TURF") == surface][-6:]
        figs = [x["speed"] for x in runs[-6:] if x["speed"] is not None]
        early = [x["early"] for x in runs[-6:] if x["early"] is not None]
        fsc = [x["fscore"] for x in runs[-5:]]
        dsc = [x["fscore"] for x in runs if dist and abs((x["dist"] or 0) - dist) <= 200]
        wts = np.array([0.6 ** i for i in range(len(fsc))][::-1])
        return {
            "n_runs": len(runs),
            "speed_best": float(np.mean(sorted(figs)[-2:])) if figs else None,
            "speed_last": figs[-1] if figs else None,
            "early": float(np.mean(early)) if early else None,
            "form": float(np.dot(wts, fsc) / wts.sum()) if fsc else None,
            "dist_form": float(np.mean(dsc[-6:])) if dsc else None,
            "last_date": runs[-1]["date"] if runs else None,
            "mkt": _mean([x.get("mkt") for x in runs[-3:]]),
            "vs_mkt": _mean([x.get("vs_mkt") for x in runs[-3:]]),
            "closing": _mean([x.get("closing") for x in runs[-4:]]),
            "perf_best": _mean(sorted(x["perf"] for x in runs[-5:]
                                      if x.get("perf") is not None)[-2:]),
            "perf_last": _mean([x.get("perf") for x in runs[-1:]]),
            "late_s": _mean([x.get("late_s") for x in runs[-3:]]),
            "early_s": _mean([x.get("early_s") for x in runs[-3:]]),
            "last_cls": runs[-1].get("cls") if runs else None,
            "last_bw": runs[-1].get("bw") if runs else None,
            "last_est": runs[-1].get("est") if runs else None,
            "last_wide": bool(runs and runs[-1].get("wide")),
            "last_jockey": runs[-1].get("jockey") if runs else None,
            "awt_pref": float(np.mean([x["fscore"] for x in runs if x.get("surface") == "AWT"])
                              - np.mean([x["fscore"] for x in runs]))
                        if any(x.get("surface") == "AWT" for x in runs) else None,
            # last run compared with the ones before it (+ = improving)
            "form_trend": runs[-1]["fscore"] - float(np.mean([x["fscore"] for x in runs[-4:-1]]))
                          if len(runs) >= 3 else None,
            # form at this venue + surface (needs 2+ runs there)
            "venue_form": float(np.mean(here)) if len(here) >= 2 else None,
        }

    def trial_factors(self, code, race_date):
        """Barrier trials on earlier days: how many in the last 60 days, and how the latest went."""
        day = date.fromisoformat(race_date)
        count, score = 0, 0.0
        for t in reversed(self.trials.get(code, [])):
            if t["date"] >= race_date:
                continue
            age = (day - date.fromisoformat(t["date"])).days
            if age > 60:
                break
            if count == 0 and age <= 45:
                score = 0.5 - (t["pos"] - 1) / max(t["field"] - 1, 1)
            count += 1
        return {"trial_score": score, "trial_count": float(count)}

    def race_features(self, race, runners):
        """runners: list of dicts with horse_code, draw, weight, jockey_id, trainer_id."""
        dist = race.get("distance")
        profs = [self.runner_profile(x.get("horse_code"), dist, race["venue"],
                                     race.get("surface", "TURF")) for x in runners]
        field = max(len(runners), 2)
        pace, n_lead = pace_projection(profs, field)
        t_style, t_draw, t_n = self.today_bias(race)
        t_var, t_late, t_nv, t_nl = self.today_sectionals(race)
        c_edge = self.course_leader_edge(race)
        sp_vals = [p["speed_best"] for p in profs if p["speed_best"] is not None]
        sp_fill = float(np.percentile(sp_vals, 35)) if sp_vals else 0.0
        wts = [x.get("weight") or 0 for x in runners]
        mean_wt = np.mean([w for w in wts if w]) if any(wts) else 0
        this_cls = class_num(race.get("class"))
        ratings = est_ratings(race, runners)
        known_r = [r for r in ratings if r is not None]
        r_fill = float(np.median(known_r)) if known_r else 0.0
        pb_known = [p["perf_best"] for p in profs if p["perf_best"] is not None]
        pb_fill = float(np.percentile(pb_known, 35)) if pb_known else r_fill

        X, extra = [], []
        for x, p, rating in zip(runners, profs, ratings):
            rating = rating if rating is not None else r_fill
            gear = gear_flags(x.get("gear"))
            pb = p["perf_best"] if p["perf_best"] is not None else pb_fill
            pl = p["perf_last"] if p["perf_last"] is not None else pb
            early = p["early"] if p["early"] is not None else 0.5
            e_c = 0.5 - early  # >0 forward-running
            draw = x.get("draw") or 0
            draw_frac = (draw - 1) / max(field - 1, 1) if draw else 0.5
            layoff = 0.0
            if p["last_date"]:
                days = (np.datetime64(race["date"]) - np.datetime64(p["last_date"])).astype(int)
                layoff = math.log1p(max(days, 0)) - math.log1p(21)
            sb = p["speed_best"] if p["speed_best"] is not None else sp_fill
            sl = p["speed_last"] if p["speed_last"] is not None else sb
            f = {
                "speed_best": sb,
                "speed_last": sl,
                "speed_missing": 1.0 if p["speed_best"] is None else 0.0,
                "form": p["form"] if p["form"] is not None else 0.4,
                "dist_form": p["dist_form"] if p["dist_form"] is not None else 0.4,
                "early": e_c,
                "early_x_pace": -e_c * pace,
                "early_x_today": e_c * t_style * 4,
                "early_x_course": e_c * c_edge * 4,
                "draw_bias": self.draw_bias(race, draw),
                "draw_x_today": (0.5 - draw_frac) * t_draw * 4,
                "jockey": math.log(self.jockey.rate(x.get("jockey_id"))),
                "trainer": math.log(self.trainer.rate(x.get("trainer_id"))),
                "jockey_form": self.jockey_recent.rate_vs(
                    x.get("jockey_id"), self.jockey_place.rate(x.get("jockey_id"))),
                "trainer_form": self.trainer_recent.rate_vs(
                    x.get("trainer_id"), self.trainer_place.rate(x.get("trainer_id"))),
                "combo": math.log(self.combo.rate((x.get("jockey_id"), x.get("trainer_id")))),
                "trainer_venue": self.trainer_venue.rate_vs(
                    (x.get("trainer_id"), race["venue"]), self.trainer_place.rate(x.get("trainer_id"))),
                # weight-implied rating now vs. last run (+ = raised by the handicapper)
                "rating_move": (rating - p["last_est"]) / 5 if p["last_est"] is not None else 0.0,
                "form_trend": p["form_trend"] if p["form_trend"] is not None else 0.0,
                "gear_first": gear["first"],
                "gear_off": gear["off"],
                "gear_back": gear["back"],
                "run_wide": float(p["last_wide"]),
                "jockey_up": math.log(self.jockey.rate(x.get("jockey_id")))
                             - math.log(self.jockey.rate(p["last_jockey"]))
                             if p["last_jockey"] and x.get("jockey_id")
                             and x.get("jockey_id") != p["last_jockey"] else 0.0,
                "awt_pref": p["awt_pref"] if race.get("surface") == "AWT" and p["awt_pref"] is not None
                            else 0.0,
                **self.trial_factors(x.get("horse_code"), race["date"]),
                "venue_form": (p["venue_form"] - p["form"])
                              if p["venue_form"] is not None and p["form"] is not None else 0.0,
                "weight_rel": ((x.get("weight") or mean_wt) - mean_wt) / 5,
                "layoff": layoff,
                "debut": 1.0 if p["n_runs"] == 0 else 0.0,
                "mkt_rating": p["mkt"] if p["mkt"] is not None else 0.0,
                "vs_mkt": p["vs_mkt"] if p["vs_mkt"] is not None else 0.0,
                "closing": p["closing"] if p["closing"] is not None else 0.0,
                "class_drop": (this_cls - p["last_cls"])
                              if this_cls is not None and p["last_cls"] is not None else 0.0,
                "bw_change": abs(x["body_weight"] - p["last_bw"]) / 10
                             if x.get("body_weight") and p["last_bw"] else 0.0,
                "perf_best": pb,
                "perf_last": pl,
                # recent performance above today's rating = well handicapped
                "handicap_edge": (p["perf_best"] - rating) if p["perf_best"] is not None else 0.0,
                "late_speed": p["late_s"] if p["late_s"] is not None else 0.0,
                "early_speed": p["early_s"] if p["early_s"] is not None else 0.0,
            }
            X.append([f[k] for k in FEATURES])
            extra.append({"profile": p, "draw_frac": draw_frac})
        ctx = {"pace": pace, "n_lead": n_lead, "today_style": t_style, "today_draw": t_draw,
               "today_n": t_n, "course_edge": c_edge, "today_variant": t_var,
               "today_late": t_late, "today_sect_n": max(t_nv, t_nl),
               "today_read": self.today_read(race), "course_vs_norm": self.course_vs_norm(race)}
        return np.array(X, dtype=float), extra, ctx

    # ---------------------------------------------------------------- update
    def update(self, race):
        # only horses that actually started: withdrawn (WV/WX...) and void runs never ran
        runners = [x for x in race["runners"] if x.get("horse_code") and started(x)]
        finished = [x for x in runners if x.get("place")]
        field = len(runners)
        if not finished:
            return
        km = (race.get("distance") or 1000) / 1000
        winner = min(finished, key=lambda x: x["place"])
        # record how each pre-race running style fared under the projected pace
        # (profiles are still pre-race here: this race isn't in the histories yet)
        profs = [self.runner_profile(x["horse_code"], race.get("distance")) for x in runners]
        pace, _ = pace_projection(profs, max(field, 2))
        book = self.pace_style[(race["venue"], pace_bucket(pace))]
        for x, p in zip(runners, profs):
            s = style_label(p["early"])
            if s in STYLES:
                book[s][0] += 1
                book[s][1] += 1 if x is winner else 0
        variant = None
        if winner.get("time") and race.get("distance"):
            std = self.std_time(race)
            if std:
                # track speed: winner vs. standard, minus the usual gap for this class
                raw = (winner["time"] - std) / km
                cls = race.get("class") or "?"
                cv = self.class_var.get(cls)
                variant = raw - (median(cv[-150:]) if cv and len(cv) >= 5 else 0.0)
                if race.get("going") in GOOD:
                    self.class_var[cls].append(raw)
            if race.get("going") in GOOD:
                self.winner_times[track_key(race)].append(winner["time"])
            elif std:
                self.going_diff[going_key(race.get("going"))].append((winner["time"] - std) / km)
        priced = [x for x in runners if (x.get("win_odds") or 0) > 0]
        mkt = {}
        if len(priced) >= 4:
            inv = np.array([1 / x["win_odds"] for x in priced])
            q = inv / inv.sum()
            ranks = (-q).argsort().argsort() + 1
            mkt = {x["no"]: (float(qi), int(rk)) for x, qi, rk in zip(priced, q, ranks)}
        # performance rating: estimated rating + (field-median beaten lbs - own beaten lbs)
        perf = {}
        wt_time = winner.get("time")
        if wt_time and race.get("distance"):
            lb_per_len = 3000 / race["distance"]
            rs = est_ratings(race, runners)
            beaten = {}
            for x, r in zip(runners, rs):
                if x.get("place") and x.get("time") and r is not None:
                    lens = min((x["time"] - wt_time) / SEC_PER_LENGTH, 30)
                    beaten[x["no"]] = (r, max(lens, 0) * lb_per_len)
            if len(beaten) >= 4:
                med = float(np.median([b for _, b in beaten.values()]))
                perf = {no: r + med - b for no, (r, b) in beaten.items()}
        # sectionals: own first / final section vs. the field median (seconds, + = faster)
        sect_rel = {}
        with_sect = [x for x in runners if x.get("place") and x.get("sect")]
        if len(with_sect) >= 4:
            first_med = float(np.median([x["sect"][0] for x in with_sect]))
            last_med = float(np.median([x["sect"][-1] for x in with_sect]))
            sect_rel = {x["no"]: (first_med - x["sect"][0], last_med - x["sect"][-1])
                        for x in with_sect}
        # closing advantage: how much the top 3 out-finished the field, vs. a normal race
        late_top = [sect_rel[x["no"]][1] for x in finished
                    if x["place"] <= 3 and x["no"] in sect_rel]
        late_excess = None
        if len(late_top) >= 2:
            lt = float(np.mean(late_top))
            hist = self.late_top_hist
            if len(hist) >= 30:
                late_excess = lt - float(np.mean(hist[-400:]))
            hist.append(lt)
        est = dict(zip((x["no"] for x in runners), est_ratings(race, runners)))
        dk = draw_key(race)
        top_early, top_draw = [], []
        for x in runners:
            pos = x.get("place")
            early = None
            if x.get("running"):
                early = (x["running"][0] - 1) / max(field - 1, 1)
            if x.get("draw"):
                s = self.draw[dk][x["draw"]]
                s[0] += 1
                s[1] += 1 if pos == 1 else 0
                s[2] += 1 if pos and pos <= 3 else 0
            if pos and pos <= 3:
                if early is not None:
                    top_early.append(early)
                if x.get("draw"):
                    top_draw.append((x["draw"] - 1) / max(field - 1, 1))
            if pos == 1 and early is not None:
                self.course_early[dk].append(early)
            self.jockey.add(x.get("jockey_id"), pos == 1)
            self.trainer.add(x.get("trainer_id"), pos == 1)
            placed = bool(pos and pos <= 3)
            self.jockey_place.add(x.get("jockey_id"), placed)
            self.trainer_place.add(x.get("trainer_id"), placed)
            self.jockey_recent.add(x.get("jockey_id"), placed)
            self.trainer_recent.add(x.get("trainer_id"), placed)
            if x.get("jockey_id") and x.get("trainer_id"):
                self.combo.add((x["jockey_id"], x["trainer_id"]), placed)
            self.trainer_venue.add((x.get("trainer_id"), race["venue"]), placed)
            fscore = 1 - (pos - 1) / max(field - 1, 1) if pos else 0.0
            m = mkt.get(x["no"])
            self.horse[x["horse_code"]].append({
                "date": race["date"], "dist": race.get("distance"), "venue": race["venue"],
                "surface": race.get("surface", "TURF"),
                "speed": self.speed_fig(race, x.get("time"), x.get("weight")) if pos else None,
                "early": early, "fscore": fscore, "place": pos,
                # how the market rated the horse that day (log of implied prob x field size)
                "mkt": math.log(m[0] * field) if m else None,
                # finished better (+) or worse (-) than its market rank implied
                "vs_mkt": (m[1] - pos) / max(field - 1, 1) if m and pos else None,
                "closing": (x["running"][0] - pos) / max(field - 1, 1)
                           if pos and x.get("running") else None,
                "cls": class_num(race.get("class")), "bw": x.get("body_weight"),
                "jockey": x.get("jockey_id"),
                "perf": perf.get(x["no"]), "est": est.get(x["no"]),
                "wide": reports.ran_wide(x),
                "early_s": sect_rel[x["no"]][0] if x["no"] in sect_rel else None,
                "late_s": sect_rel[x["no"]][1] if x["no"] in sect_rel else None,
            })
        front = sum(1 for e in top_early if e <= 1 / 3)
        inner = sum(1 for d in top_draw if d <= 1 / 3)
        self.meeting[(race["date"], race["venue"])].append({
            "race_no": race["race_no"], "surface": race.get("surface", "TURF"),
            "top_early": float(np.mean(top_early)) if top_early else 0.45,
            "top_draw": float(np.mean(top_draw)) if top_draw else 0.5,
            "variant": variant, "late_excess": late_excess,
            "n_early": len(top_early), "front": front, "n_draw": len(top_draw), "inner": inner,
        })
        nb = self.style_norm[(race["venue"], race.get("surface", "TURF"))]
        nb[0] += len(top_early)
        nb[1] += front
        nb[2] += len(top_draw)
        nb[3] += inner
        win_early = [(x["running"][0] - 1) / max(field - 1, 1) for x in runners
                     if x.get("place") == 1 and x.get("running")]
        if win_early:
            nb[4] += win_early[0]
            nb[5] += 1


# -------------------------------------------------------------------- training

def build_dataset(races):
    """Replay history. Returns (state, samples) where each sample is one race."""
    st = State()
    samples = []
    for race in races:
        # results list horses in finishing order; by number, a tie on odds can't favour the winner
        runners = sorted((x for x in race["runners"]
                          if x.get("horse_code") and x.get("draw") and started(x)),
                         key=lambda x: x.get("no") or 99)
        win_idx = [i for i, x in enumerate(runners) if x.get("place") == 1]
        if len(runners) >= 5 and len(win_idx) == 1:
            X, _, _ = st.race_features(race, runners)
            odds = np.array([x.get("win_odds") or 0 for x in runners], dtype=float)
            # finishing order of the first three (stops at a dead heat)
            order = []
            for place in (1, 2, 3):
                idx = [i for i, x in enumerate(runners) if x.get("place") == place]
                if len(idx) != 1:
                    break
                order.append(idx[0])
            samples.append({"date": race["date"], "X": X, "y": win_idx[0], "odds": odds,
                            "order": order, "nos": [x["no"] for x in runners],
                            "dividends": race.get("dividends")})
        st.update(race)
    return st, samples


def _standardise(samples):
    allx = np.vstack([s["X"] for s in samples])
    mu, sd = allx.mean(0), allx.std(0)
    sd[sd < 1e-3] = 1.0  # constant so far (e.g. bias stats still warming up)
    return mu, sd


def _z(X, mu, sd):
    return np.clip((X - mu) / sd, -4, 4)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# Domain-knowledge starting weights (per standardised feature). Training shrinks
# towards these instead of towards zero, so thin data can't flip obvious signs.
PRIOR = {
    "speed_best": 0.35, "speed_last": 0.15, "form": 0.3, "dist_form": 0.1,
    "early_x_pace": 0.1, "early_x_today": 0.1, "early_x_course": 0.1,
    "draw_bias": 0.15, "draw_x_today": 0.1, "jockey": 0.3, "trainer": 0.2,
    "speed_missing": -0.1, "mkt_rating": 0.4, "vs_mkt": 0.1, "class_drop": 0.1,
    "perf_best": 0.2, "handicap_edge": 0.2, "late_speed": 0.1,
}


def prior_vector():
    return np.array([PRIOR.get(f, 0.0) for f in FEATURES])


STAGE_WEIGHTS = (1.0, 0.6, 0.4)


# stages=1 (winner only) tested slightly better than 3 on the walk-forward backtest
def fit(samples, l2=0.05, iters=600, lr=0.05, stages=1):
    mu, sd = _standardise(samples)
    w0 = prior_vector()
    w = w0.copy()
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    # pad races into one tensor so each step is a few vectorised ops
    n, kmax = len(samples), max(len(s["X"]) for s in samples)
    Xp = np.zeros((n, kmax, len(FEATURES)))
    mask = np.zeros((n, kmax), dtype=bool)
    for i, s in enumerate(samples):
        Xp[i, : len(s["X"])] = _z(s["X"], mu, sd)
        mask[i, : len(s["X"])] = True
    # Rank-ordered logit: 1st among all, 2nd among the rest, 3rd among the rest.
    # Later stages are down-weighted because minor placings are noisier.
    stage_masks, stage_X, stage_has = [], [], []
    cur = mask.copy()
    for k in range(min(stages, len(STAGE_WEIGHTS))):
        has = np.array([len(s.get("order") or [s["y"]]) > k for s in samples])
        pick = np.array([(s.get("order") or [s["y"]])[k] if h else 0
                         for s, h in zip(samples, has)])
        stage_masks.append(cur.copy())
        stage_X.append(Xp[np.arange(n), pick])
        stage_has.append(has)
        cur[np.arange(n)[has], pick[has]] = False
    for t in range(1, iters + 1):
        g = l2 * (w - w0)
        s_all = Xp @ w
        for k, (mk, xk, hk) in enumerate(zip(stage_masks, stage_X, stage_has)):
            z = np.where(mk, s_all, -np.inf)
            z -= z.max(1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(1, keepdims=True)
            gk = (p[..., None] * Xp).sum(1) - xk
            g += STAGE_WEIGHTS[k] * gk[hk].sum(0) / n
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        w -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return {"w": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist(), "features": FEATURES}


def predict(params, X):
    w, mu, sd = (np.array(params[k]) for k in ("w", "mu", "sd"))
    return softmax(_z(X, mu, sd) @ w)


def contributions(params, X):
    w, mu, sd = (np.array(params[k]) for k in ("w", "mu", "sd"))
    return _z(X, mu, sd) * w


def evaluate(params, samples):
    ll, hit, mkt_hit, mkt_ll, n, roi_stake, roi_ret = 0, 0, 0, 0, 0, 0, 0
    for s in samples:
        p = predict(params, s["X"])
        ll -= math.log(max(p[s["y"]], 1e-9))
        top = int(np.argmax(p))
        hit += top == s["y"]
        odds = s["odds"]
        if (odds > 0).all():
            q = (1 / odds) / (1 / odds).sum()
            mkt_ll -= math.log(max(q[s["y"]], 1e-9))
            mkt_hit += int(np.argmin(odds)) == s["y"]
            roi_stake += 1
            roi_ret += odds[top] if top == s["y"] else 0
        n += 1
    return {
        "races": n,
        "logloss": round(ll / n, 4) if n else None,
        "top_pick_win_rate": round(hit / n, 4) if n else None,
        "market_logloss": round(mkt_ll / max(roi_stake, 1), 4),
        "favourite_win_rate": round(mkt_hit / max(roi_stake, 1), 4),
        "flat_bet_roi_top_pick": round(float(roi_ret) / roi_stake - 1, 4) if roi_stake else None,
        "uniform_logloss": round(float(np.mean([math.log(len(s["X"])) for s in samples])), 4) if n else None,
    }


def market_probs(odds):
    """Implied win probabilities from win odds (takeout removed by normalising)."""
    odds = np.asarray(odds, dtype=float)
    if len(odds) == 0 or not (odds > 0).all():
        return None
    q = 1 / odds
    return q / q.sum()


def blend(p, q, a, b):
    """Combine model and market: p_blend ~ p_model^a * q_market^b."""
    z = a * np.log(np.maximum(p, 1e-9)) + b * np.log(np.maximum(q, 1e-9))
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


MIN_MODEL_GAIN = 0.002   # log loss, the same bar the self-check uses to switch settings


def fit_blend(params, samples):
    """Grid-fit blend weights on races the model was not trained on."""
    val = [(predict(params, s["X"]), market_probs(s["odds"]), s["y"]) for s in samples]
    val = [v for v in val if v[1] is not None]
    if len(val) < 50:
        return None
    best = market_only = None
    for a in np.arange(0, 1.01, 0.05):
        for b in np.arange(0.8, 1.31, 0.05):
            ll = -float(np.mean([math.log(blend(p, q, a, b)[y]) for p, q, y in val]))
            cand = {"a": round(float(a), 2), "b": round(float(b), 2), "logloss": round(ll, 4)}
            if best is None or ll < best["logloss"]:
                best = cand
            if a == 0 and (market_only is None or ll < market_only["logloss"]):
                market_only = cand
    # a tiny "gain" is noise: it switched the model's weight on and off between updates
    if best["logloss"] > market_only["logloss"] - MIN_MODEL_GAIN:
        best = market_only
    best["market_logloss"] = round(-float(np.mean([math.log(q[y]) for _, q, y in val])), 4)
    return best


def train(races, cfg=None):
    """cfg: tuned settings from the self-check ({"l2", "stages"}); defaults otherwise."""
    global FEATURES
    cfg = cfg or {}
    l2, stages = cfg.get("l2", 0.05), cfg.get("stages", 1)
    FEATURES = active_features(cfg)   # what the self-check has learned to use
    st, samples = build_dataset(races)
    # the first weeks are warm-up: horse histories and bias stats are still empty
    if samples:
        start = np.datetime64(samples[0]["date"]) + np.timedelta64(WARMUP_DAYS, "D")
        samples = [s for s in samples if np.datetime64(s["date"]) >= start]
    if len(samples) < 60:
        return st, None, {"note": "not enough history", "races": len(samples)}
    cut = int(len(samples) * 0.8)
    params = fit(samples[:cut], l2=l2, stages=stages)
    report = {"holdout": evaluate(params, samples[cut:]), "train_races": cut,
              "holdout_from": samples[cut]["date"]}
    # how much to trust the model vs. the market, judged on the unseen holdout
    blend_w = fit_blend(params, samples[cut:])
    params = fit(samples, l2=l2, stages=stages)  # final weights use everything
    params["blend"] = blend_w
    report["blend"] = blend_w
    report["weights"] = dict(zip(FEATURES, [round(x, 3) for x in params["w"]]))
    return st, params, report
