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
from statistics import median

import numpy as np

FEATURES = [
    "speed_best", "speed_last", "speed_missing", "form", "dist_form",
    "early", "early_x_pace", "early_x_today", "early_x_course",
    "draw_bias", "draw_x_today", "jockey", "trainer", "weight_rel",
    "layoff", "debut", "mkt_rating", "vs_mkt", "closing", "class_drop", "bw_change",
    "perf_best", "perf_last", "handicap_edge", "late_speed", "early_speed",
]
# Same-day sectional reads (track variant, closing advantage) are shown to the user
# but kept out of the model: as features they worsened walk-forward log loss
# (2.2251 -> 2.2303 on 452 races).
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
    """Rating per runner: card rating if published, else band top minus weight below top weight."""
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


class State:
    def __init__(self):
        self.horse = defaultdict(list)            # code -> runs
        self.winner_times = defaultdict(list)     # track_key -> times on good ground
        self.going_diff = defaultdict(list)       # going -> (time-std)/km on winners
        self.draw = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # key -> draw -> [runs, win, place]
        self.course_early = defaultdict(list)     # draw_key -> early frac of winners
        self.jockey = RateBook()
        self.trainer = RateBook()
        self.meeting = defaultdict(list)          # (date, venue) -> finished race summaries
        self.class_var = defaultdict(list)        # class -> (winner time - std)/km, good ground
        self.late_top_hist = []                   # per race: avg final-section edge of top 3

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
    def runner_profile(self, code, dist):
        runs = self.horse.get(code, [])
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
        }

    def race_features(self, race, runners):
        """runners: list of dicts with horse_code, draw, weight, jockey_id, trainer_id."""
        dist = race.get("distance")
        profs = [self.runner_profile(x.get("horse_code"), dist) for x in runners]
        field = max(len(runners), 2)
        early_known = [p["early"] for p in profs if p["early"] is not None]
        n_lead = sum(1 for e in early_known if e <= 0.25)
        exp_lead = 0.2 * field
        pace = (n_lead - exp_lead) / max(field, 1) * 4  # >0 = contested early pace
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
               "today_late": t_late, "today_sect_n": max(t_nv, t_nl)}
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
            fscore = 1 - (pos - 1) / max(field - 1, 1) if pos else 0.0
            m = mkt.get(x["no"])
            self.horse[x["horse_code"]].append({
                "date": race["date"], "dist": race.get("distance"), "venue": race["venue"],
                "speed": self.speed_fig(race, x.get("time"), x.get("weight")) if pos else None,
                "early": early, "fscore": fscore, "place": pos,
                # how the market rated the horse that day (log of implied prob x field size)
                "mkt": math.log(m[0] * field) if m else None,
                # finished better (+) or worse (-) than its market rank implied
                "vs_mkt": (m[1] - pos) / max(field - 1, 1) if m and pos else None,
                "closing": (x["running"][0] - pos) / max(field - 1, 1)
                           if pos and x.get("running") else None,
                "cls": class_num(race.get("class")), "bw": x.get("body_weight"),
                "perf": perf.get(x["no"]),
                "early_s": sect_rel[x["no"]][0] if x["no"] in sect_rel else None,
                "late_s": sect_rel[x["no"]][1] if x["no"] in sect_rel else None,
            })
        self.meeting[(race["date"], race["venue"])].append({
            "race_no": race["race_no"], "surface": race.get("surface", "TURF"),
            "top_early": float(np.mean(top_early)) if top_early else 0.45,
            "top_draw": float(np.mean(top_draw)) if top_draw else 0.5,
            "variant": variant, "late_excess": late_excess,
        })


# -------------------------------------------------------------------- training

def build_dataset(races):
    """Replay history. Returns (state, samples) where each sample is one race."""
    st = State()
    samples = []
    for race in races:
        runners = [x for x in race["runners"]
                   if x.get("horse_code") and x.get("draw") and started(x)]
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
                            "order": order})
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


def fit_blend(params, samples):
    """Grid-fit blend weights on races the model was not trained on."""
    val = [(predict(params, s["X"]), market_probs(s["odds"]), s["y"]) for s in samples]
    val = [v for v in val if v[1] is not None]
    if len(val) < 50:
        return None
    best = None
    for a in np.arange(0, 1.01, 0.05):
        for b in np.arange(0.8, 1.31, 0.05):
            ll = -float(np.mean([math.log(blend(p, q, a, b)[y]) for p, q, y in val]))
            if best is None or ll < best["logloss"]:
                best = {"a": round(float(a), 2), "b": round(float(b), 2), "logloss": round(ll, 4)}
    best["market_logloss"] = round(-float(np.mean([math.log(q[y]) for _, q, y in val])), 4)
    return best


def train(races):
    st, samples = build_dataset(races)
    # the first weeks are warm-up: horse histories and bias stats are still empty
    if samples:
        start = np.datetime64(samples[0]["date"]) + np.timedelta64(WARMUP_DAYS, "D")
        samples = [s for s in samples if np.datetime64(s["date"]) >= start]
    if len(samples) < 60:
        return st, None, {"note": "not enough history", "races": len(samples)}
    cut = int(len(samples) * 0.8)
    params = fit(samples[:cut])
    report = {"holdout": evaluate(params, samples[cut:]), "train_races": cut,
              "holdout_from": samples[cut]["date"]}
    # how much to trust the model vs. the market, judged on the unseen holdout
    blend_w = fit_blend(params, samples[cut:])
    params = fit(samples)  # final weights use everything
    params["blend"] = blend_w
    report["blend"] = blend_w
    report["weights"] = dict(zip(FEATURES, [round(x, 3) for x in params["w"]]))
    return st, params, report
