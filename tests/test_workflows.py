"""The cloud workflows must commit every file the code they run writes and reads back later."""
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def git_adds(name):
    text = (WORKFLOWS / name).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip().startswith("git add")]


def test_selfcheck_commits_its_report_its_settings_and_its_learning_record():
    # data/learning.json was missing here until 12 Nov 2026 (see tests/test_selfcheck.py)
    (add,) = git_adds("selfcheck.yml")
    for path in ("docs/data/selfcheck.json", "data/model_config.json", "data/learning.json"):
        assert path in add.split()


def test_update_commits_all_of_data_and_docs():
    adds = git_adds("update.yml")
    assert adds and all(add == "git add data docs" for add in adds)


def test_trackwork_commits_its_folder():
    assert git_adds("trackwork.yml") == ["git add data/trackwork"]


def test_tests_do_not_run_on_data_commits():
    # a run on every race-day data commit would bury real failures
    text = (WORKFLOWS / "tests.yml").read_text(encoding="utf-8")
    paths = text.split("paths:", 1)[1].split("workflow_dispatch", 1)[0]
    assert '"data' not in paths and '"docs' not in paths
