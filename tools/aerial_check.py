"""How well the positions in data/aerial agree with the official results.

    python tools/aerial_check.py        (from the repo root)

The finish panel is the check: the horse nearest the line should be the official winner. It
uses results only to check the reading, never to judge a factor.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from hkjc import aerial, store

rank = lambda v: np.argsort(np.argsort(v)).astype(float)
races = ran = read = weak = win_ok = top3 = exact = n = 0
per_panel = [0, 0, 0, 0]
rhos = []
for path in sorted(aerial.FOLDER.glob("20*_*.json")):
    day, venue = path.stem.split("_")
    results = {r["race_no"]: r for r in store.read_json(store.result_path(day, venue), [])}
    for no, panels in aerial.load(day, venue).items():
        race = results[no]
        place = {x["no"]: x["place"] for x in race["runners"] if x.get("place")}
        k = len(aerial._ran(race))
        races += 1; ran += 4 * k
        for i, p in enumerate(panels):
            per_panel[i] += len(p); read += len(p); weak += sum(1 for r in p if r[3] < aerial.WEAK_FIT)
        rows = [(r[0], r[1]) for r in panels[3] if r[0] in place]
        if len(rows) < 4:
            continue
        xs = np.array([x for _, x in rows]); pl = np.array([place[h] for h, _ in rows], dtype=float)
        rhos.append(np.corrcoef(rank(xs), rank(pl))[0, 1])
        exact += int((rank(xs) == rank(pl)).sum()); n += len(rows)
        win_ok += place.get(rows[int(np.argmin(xs))][0]) == 1
        top3 += len({h for h, _ in sorted(rows, key=lambda r: r[1])[:3]} & {h for h, p in place.items() if p <= 3})
print(f"races {races}; positions read {read}/{ran} ({read / ran:.1%}); weak number fits {weak} ({weak / read:.2%})")
print("per panel:", [f"{v / (ran / 4):.1%}" for v in per_panel])
print(f"finish: nearest the line = official winner {win_ok}/{races}; first three {top3}/{races * 3}; "
      f"exact place {exact}/{n} ({exact / n:.1%}); mean agreement {np.mean(rhos):.3f}")
