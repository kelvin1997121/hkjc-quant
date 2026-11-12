"""The tests must never write the repo's own data/ or docs/."""
import importlib
import pkgutil
from pathlib import Path

import hkjc
from hkjc import store


def test_every_data_path_is_redirected(sandbox):
    real = [store.ROOT / "data", store.ROOT / "docs"]
    leaks = []
    for info in pkgutil.iter_modules(hkjc.__path__):
        module = importlib.import_module(f"hkjc.{info.name}")
        for name, value in vars(module).items():
            if isinstance(value, Path) and any(r == value or r in value.parents for r in real):
                leaks.append(f"hkjc.{info.name}.{name}")
    assert not leaks, f"add these to REDIRECTS in tests/conftest.py: {leaks}"


def test_writes_land_in_the_sandbox(sandbox):
    store.save_results([{"date": "2030-01-01", "venue": "ST", "race_no": 1, "runners": []}])
    assert (sandbox / "data" / "results" / "2030-01-01_ST.json").exists()
    assert not (store.ROOT / "data" / "results" / "2030-01-01_ST.json").exists()
