"""Audit of the model's factors: are they computed the same way live as in training, are any
mostly empty, does any know the result, and does each point the way it should?

    python tools/model_audit.py        (from the repo root; about a minute)

Nothing is written and nothing is fitted for adoption. It reads data/ and prints a report.
"""
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hkjc import model, store  # noqa: E402

FIELDS = ("horse_code", "draw", "weight", "jockey_id", "trainer_id", "rating", "gear", "body_weight")


def main():
    model.FEATURES = list(model.ALL_FEATURES)
    F = model.FEATURES
    races = store.load_all_results()
    cards = {}
    for p in sorted(store.CARDS.glob("20*_*.json")):
        c = store.read_json(p)
        for r in c.get("races", []):
            cards[(c["date"], c["venue"], r["race_no"])] = r
    print(f"races {len(races)} ({races[0]['date']} to {races[-1]['date']}); races with a stored card {len(cards)}")

    # ---------------------------------------------------------------- A. card against result
    print("\nA. The same runner on the race card and in the result (fields the model reads)")
    diff, seen, examples = Counter(), Counter(), {}
    for r in races:
        card = cards.get((r["date"], r["venue"], r["race_no"]))
        if not card:
            continue
        by = {x["no"]: x for x in card["runners"]}
        for x in r["runners"]:
            c = by.get(x.get("no"))
            if not c or not model.started(x):
                continue
            for f in FIELDS:
                a, b = c.get(f), x.get(f)
                if f == "gear":
                    a, b = a or "", b or ""
                if a in (None, "") and b in (None, ""):
                    continue
                seen[f] += 1
                if str(a) != str(b):
                    diff[f] += 1
                    examples.setdefault(f, []).append((r["date"], r["race_no"], x["no"], a, b))
    for f in FIELDS:
        ex = "; e.g. " + ", ".join(f"{d} R{n} #{h}: card {a!r} / result {b!r}" for d, n, h, a, b in examples.get(f, [])[:3]) \
            if diff[f] else ""
        print(f"   {f:12s} compared {seen[f]:5d}, different {diff[f]:5d} ({diff[f] / max(seen[f], 1):.1%}){ex}")

    # ---------------------------------------------------------------- B, C, D: replay
    st = model.State()
    rows, skew, skew_n = [], np.zeros(len(F)), 0
    skew_big = Counter()
    for race in races:
        runners = sorted((x for x in race["runners"] if x.get("horse_code") and x.get("draw") and model.started(x)),
                         key=lambda x: x.get("no") or 99)
        win = [i for i, x in enumerate(runners) if x.get("place") == 1]
        if len(runners) >= 5 and len(win) == 1:
            X, _, _ = st.race_features(race, runners)
            card = cards.get((race["date"], race["venue"], race["race_no"]))
            if card:
                ran = {x["no"] for x in runners}
                live = sorted((x for x in card["runners"] if x.get("draw") and x["no"] in ran), key=lambda x: x["no"])
                if [x["no"] for x in live] == [x["no"] for x in runners]:
                    XL, _, _ = st.race_features({**card, "date": race["date"], "venue": race["venue"]}, live)
                    d = np.abs(XL - X)
                    skew += d.sum(0)
                    skew_n += len(runners)
                    for j in np.flatnonzero((d > 1e-9).any(0)):
                        skew_big[F[j]] += int((d[:, j] > 1e-9).sum())
            field = len(runners)
            fscore = np.array([1 - (x["place"] - 1) / max(field - 1, 1) if x.get("place") else 0.0 for x in runners])
            odds = np.array([x.get("win_odds") or 0 for x in runners], dtype=float)
            rows.append((race["date"], X, win[0], fscore, odds))
        st.update(race)

    print(f"\nB. Factors computed from the card against from the result, same race and same history "
          f"({skew_n} runners with both)")
    if skew_n:
        any_diff = False
        for j, f in enumerate(F):
            if skew_big[f]:
                any_diff = True
                print(f"   {f:15s} differs for {skew_big[f]:5d} runners ({skew_big[f] / skew_n:.1%}); mean gap {skew[j] / skew_n:.4f}")
        if not any_diff:
            print("   none differs")

    start = np.datetime64(rows[0][0]) + np.timedelta64(model.WARMUP_DAYS, "D")
    rows = [r for r in rows if np.datetime64(r[0]) >= start]
    allx = np.vstack([r[1] for r in rows])
    mu, sd = allx.mean(0), allx.std(0)
    sd[sd < 1e-9] = 1.0
    won = np.concatenate([np.arange(len(r[1])) == r[2] for r in rows])
    fs = np.concatenate([r[3] for r in rows])
    # within-race standardising, so a factor is compared with the other runners of its race
    within = np.vstack([(r[1] - r[1].mean(0)) / np.where(r[1].std(0) > 1e-9, r[1].std(0), 1.0) for r in rows])
    logq = np.concatenate([np.log(np.where(r[4] > 0, 1 / np.maximum(r[4], 1e-9), np.nan)) -
                           np.nanmean(np.log(np.where(r[4] > 0, 1 / np.maximum(r[4], 1e-9), np.nan))) for r in rows])
    mode = [Counter(np.round(allx[:, j], 6)).most_common(1)[0] for j in range(len(F))]
    samples = [{"date": d, "X": X, "y": y, "odds": o, "order": [y]} for d, X, y, _, o in rows]
    params = model.fit(samples)
    w = dict(zip(F, params["w"]))

    print(f"\nC. Each factor over {len(rows)} races, {len(allx)} runners (after the warm-up)")
    print("   factor           most common value (share)   mean      sd    clipped  winners-vs-rest  with market  weight  prior")
    notes = []
    for j, f in enumerate(F):
        val, cnt = mode[j]
        share = cnt / len(allx)
        z = (allx[:, j] - mu[j]) / sd[j]
        clipped = float((np.abs(z) > 4).mean())
        gap = float(within[won, j].mean() - within[~won, j].mean())
        ok = ~np.isnan(logq)
        with_mkt = float(np.corrcoef(within[ok, j], logq[ok])[0, 1]) if within[:, j].std() > 0 else 0.0
        prior = model.PRIOR.get(f, 0.0)
        print(f"   {f:15s}  {val:9.3f} ({share:5.1%})          {mu[j]:8.3f} {sd[j]:7.3f}  {clipped:6.2%}     {gap:+7.3f}       {with_mkt:+6.2f}    {w[f]:+6.3f}  {prior:+5.2f}")
        if share > 0.85 and f not in ("speed_missing", "debut", "gear_first", "gear_off", "gear_back", "run_wide",
                                      "jockey_up", "awt_pref", "trial_score", "trial_count"):
            notes.append(f"{f}: {share:.0%} of runners share one value ({val:g})")
        if prior and gap * prior < 0 and abs(gap) > 0.05:
            notes.append(f"{f}: winners differ from the rest the opposite way to its starting weight ({gap:+.2f} vs prior {prior:+.2f})")
        if abs(w[f]) > 0.05 and gap * w[f] < 0 and abs(gap) > 0.05:
            notes.append(f"{f}: fitted weight {w[f]:+.3f} points against the plain winners-vs-rest gap {gap:+.2f}")

    print("\nD. Does any factor know the result? (link with the same race's finishing position; a factor"
          "\n   built before the race should be no better linked than the market is)")
    ok = ~np.isnan(logq)
    mkt_link = float(np.corrcoef(logq[ok], fs[ok])[0, 1])
    print(f"   market odds with finishing position: {mkt_link:+.3f}")
    for j, f in enumerate(F):
        c = float(np.corrcoef(within[:, j], fs)[0, 1]) if within[:, j].std() > 0 else 0.0
        if abs(c) > abs(mkt_link) * 0.8:
            notes.append(f"{f}: linked with the same race's result at {c:+.3f} (market {mkt_link:+.3f}) - check for a leak")
    strongest = sorted(((abs(float(np.corrcoef(within[:, j], fs)[0, 1])), f) for j, f in enumerate(F) if within[:, j].std() > 0), reverse=True)[:6]
    print("   strongest factors:", ", ".join(f"{f} {c:.3f}" for c, f in strongest))

    print("\nPoints to look at:" if notes else "\nNo point flagged by these checks.")
    for n in notes:
        print("   -", n)
    inputs_by_season(races)
    speed_by_course(races)


def season(day):
    y = int(day[:4])
    a = y if day[5:7] >= "09" else y - 1
    return f"{a}/{str(a + 1)[2:]}"


def inputs_by_season(races):
    """E. Is an input there for some seasons only? Its factor then means different things."""
    fields = ("jockey_id", "trainer_id", "rating", "gear", "comment", "incident", "sect", "running",
              "body_weight", "win_odds", "time")
    tab = {}
    for r in races:
        row = tab.setdefault(season(r["date"]), Counter())
        for x in r["runners"]:
            if model.started(x):
                row["n"] += 1
                for f in fields:
                    row[f] += bool(x.get(f))
    print("\nE. Share of runners that have each input, by season")
    print("   season   runners  " + " ".join(f"{f[:10]:>10s}" for f in fields))
    for s in sorted(tab):
        print(f"   {s}  {tab[s]['n']:7d}  " + " ".join(f"{tab[s][f] / tab[s]['n']:10.0%}" for f in fields))
    print("   (no id: a jockey or trainer who has left Hong Kong; model.who() then goes by the name."
          "\n    gear is empty for a horse wearing none. Incident reports start in 2023/24.)")


def speed_by_course(races):
    """F. Does the same horse get a different speed figure just for where it ran?"""
    from collections import defaultdict
    from datetime import date
    st = model.State()
    figs = defaultdict(list)
    for race in races:
        st.update(race)
        for x in race["runners"]:
            runs = st.horse.get(x.get("horse_code")) or []
            if runs and runs[-1]["date"] == race["date"] and runs[-1]["speed"] is not None and model.started(x):
                figs[x["horse_code"]].append((race["date"], model.track_key(race), runs[-1]["speed"],
                                              model.class_num(race.get("class"))))
    pairs = defaultdict(list)
    for runs in figs.values():
        for a, b in zip(runs, runs[1:]):
            days = (date.fromisoformat(b[0]) - date.fromisoformat(a[0])).days
            if a[1] != b[1] and days <= 45 and a[3] == b[3]:
                pairs[(a[1], b[1])].append(b[2] - a[2])
    keys = sorted({k for pair in pairs for k in pair}, key=str)
    idx = {k: i for i, k in enumerate(keys)}
    rows, y = [], []
    for (a, b), d in pairs.items():
        for v in d:
            r = np.zeros(len(keys))
            r[idx[b]], r[idx[a]] = 1, -1
            rows.append(r)
            y.append(v)
    A = np.array(rows)
    off = np.linalg.lstsq(A, np.array(y), rcond=None)[0]
    off -= off.mean()
    count = np.abs(A).sum(0)
    print(f"\nF. Speed figures of the same horse in the same class within 45 days at two different courses "
          f"or distances ({len(y)} pairs):\n   points a figure gains or loses just for where the run was (150+ pairs)")
    shown = [k for k in sorted(keys, key=lambda k: off[idx[k]]) if count[idx[k]] >= 150]
    for k in shown:
        print(f"      {str(k):28s} {off[idx[k]]:+5.1f}")
    vals = [off[idx[k]] for k in shown]
    print(f"   spread {np.std(vals):.1f} points (range {min(vals):+.1f} to {max(vals):+.1f}); the factor itself spreads about 6")


if __name__ == "__main__":
    main()
