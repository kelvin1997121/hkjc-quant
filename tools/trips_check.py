"""The two measures of hkjc/trips.py beside what was written about the same runs.

    python tools/trips_check.py        (from the repo root)

It compares measurements with the race caller's and stewards' words for the same race. It never
looks at how a horse ran afterwards, so it can be run at any time without touching the test.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hkjc import aerial, reports, store, trips  # noqa: E402

SAID_BLOCKED = ("未能望空", "受困", "受阻")
W, wide_said, B, block_said = [], [], [], []
for path in sorted(aerial.FOLDER.glob("20*_*.json")):
    day, venue = path.stem.split("_")
    races = {r["race_no"]: r for r in store.read_json(store.result_path(day, venue), [])}
    for n, panels in aerial.load(day, venue).items():
        by = {x["no"]: x for x in races[n]["runners"]}
        for h, m in trips.run_measures(races[n], panels).items():
            text = (by[h].get("comment") or "") + " " + (by[h].get("incident") or "")
            if m["wide"] is not None:
                W.append(m["wide"]); wide_said.append(reports.ran_wide(by[h]))
            if m["boxed"] is not None:
                B.append(m["boxed"]); block_said.append(any(w in text for w in SAID_BLOCKED))
W, wide_said, B, block_said = (np.array(v) for v in (W, wide_said, B, block_said))
print(f"wide: {len(W)} runs; median {np.median(W):.1f} m; written up as wide {wide_said.mean():.0%}: "
      f"their median {np.median(W[wide_said]):.1f} m against {np.median(W[~wide_said]):.1f} m for the rest")
print(f"boxed: {len(B)} runs; flagged {B.mean():.1%}; written up as blocked: {block_said[B].mean():.1%} of the flagged "
      f"against {block_said[~B].mean():.1%} of the rest")
