"""The shell logic inside the workflows, run as it is written there.

ticker.yml decides how often to update (every 5 minutes during a meeting, every 2 while a result
is due, every 30 otherwise); update.yml decides whether to publish the site. Each test takes the
script text out of the workflow file and runs it with bash, with the clock, GitHub and git
replaced by stand-ins.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from helpers import need

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def step_script(workflow, step):
    """The `run: |` script of a named step, as bash would get it."""
    lines = (WORKFLOWS / workflow).read_text(encoding="utf-8").splitlines()
    at = next(i for i, line in enumerate(lines) if line.strip() == f"- name: {step}")
    run = next(i for i in range(at, len(lines)) if lines[i].strip() == "run: |")
    indent = len(lines[run]) - len(lines[run].lstrip()) + 2
    body = []
    for line in lines[run + 1:]:
        if line.strip() and len(line) - len(line.lstrip()) < indent:
            break
        body.append(line[indent:])
    return "\n".join(body) + "\n"


def bash(tmp_path, script, **env):
    import os
    path = tmp_path / "script.sh"
    path.write_text(script, encoding="utf-8", newline="\n")
    full = {**os.environ, "PYTHON": Path(sys.executable).as_posix(), **{k: str(v) for k, v in env.items()}}
    done = subprocess.run([need("bash"), path.as_posix()], capture_output=True, text=True, encoding="utf-8",
                          env=full, timeout=60)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


# ------------------------------------------------------------------ ticker: how often to update

CLOCK = r'''
date() {   # the clock, set by the test
  case "$1" in
    +%s) echo "${FAKE_EPOCH:-1000000}" ;;
    +%H:%M) echo "$FAKE_HM" ;;
    +%F) echo "$FAKE_DAY" ;;
    *) echo "unexpected date call: $*" >&2; exit 9 ;;
  esac
}
gh() { [ -n "$FAKE_SCHEDULE" ] && cat "$FAKE_SCHEDULE"; }    # the only gh call here reads schedule.json
curl() { return 22; }
python3() { "$PYTHON" "$@"; }
test_until=${FAKE_TEST_UNTIL:-0}
GH_REPO=owner/repo
'''

DAY_MEETING = {"date": "2026-11-14", "venue": "ST", "first_post": "12:30", "last_post": "18:10",
               "posts": {"1": "12:30", "2": "13:00", "3": "13:55", "4": "14:25", "5": "14:55", "6": "15:25",
                         "7": "16:00", "8": "16:30", "9": "17:00", "10": "17:35", "11": "18:10"}}
NIGHT_MEETING = {"date": "2026-11-18", "venue": "HV", "first_post": "18:40", "last_post": "22:50",
                 "posts": {"1": "18:40", "2": "19:10", "3": "19:40", "4": "20:10", "5": "20:40",
                           "6": "21:10", "7": "21:45", "8": "22:15", "9": "22:50"}}


@pytest.fixture(scope="module")
def ticker_functions():
    script = step_script("ticker.yml", "Trigger updates on the race-day clock")
    return script[script.index("to_min() {"):script.index("busy() {")]


def interval(tmp_path, functions, day, clock, schedule=None, resulted=(), **env):
    need("jq")
    if schedule is not None:
        path = tmp_path / "schedule.json"
        path.write_text(json.dumps({**schedule, "resulted": list(resulted)}), encoding="utf-8")
        env["FAKE_SCHEDULE"] = path.as_posix()
    return int(bash(tmp_path, CLOCK + functions + "\ninterval\n", FAKE_DAY=day, FAKE_HM=clock, **env))


def test_to_min_reads_times_with_leading_zeros(tmp_path, ticker_functions):
    out = bash(tmp_path, ticker_functions + "\nto_min 08:09; to_min 00:00; to_min 23:59\n")
    assert out.split() == ["489", "0", "1439"]          # 08 and 09 are not octal numbers here


@pytest.mark.parametrize("clock, minutes", [
    ("03:00", 0), ("07:59", 0),             # nothing overnight
    ("08:00", 30), ("15:00", 30), ("23:59", 30),
])
def test_ordinary_day(tmp_path, ticker_functions, clock, minutes):
    assert interval(tmp_path, ticker_functions, "2026-11-13", clock, DAY_MEETING) == minutes


@pytest.mark.parametrize("clock, resulted, minutes", [
    ("08:30", [], 30),                      # race day morning
    ("11:29", [], 30),                      # a minute before the window (60 minutes before the first post)
    ("11:30", [], 5),
    ("12:29", [], 5),
    ("12:33", [], 5),                       # race 1 off 3 minutes ago: its result is not due yet
    ("12:36", [], 2),                       # 6 minutes ago and no result: every 2 minutes
    ("12:36", [1], 5),                      # the result is in
    ("13:15", [], 2),                       # race 1 is 45 minutes old (still due), race 2 is 15
    ("13:16", [1], 2),                      # race 2 due
    ("13:16", [1, 2], 5),
    ("13:46", [2], 5),                      # race 1 never got a result: given up after 45 minutes
    ("18:55", list(range(1, 12)), 5),       # last post + 45
    ("18:55", list(range(1, 11)), 2),       # the last race is 45 minutes old: still asked for
    ("18:56", list(range(1, 11)), 30),      # window over
    ("23:00", list(range(1, 12)), 30),
])
def test_day_meeting(tmp_path, ticker_functions, clock, resulted, minutes):
    assert interval(tmp_path, ticker_functions, "2026-11-14", clock, DAY_MEETING, resulted) == minutes


@pytest.mark.parametrize("day, clock, resulted, minutes", [
    ("2026-11-18", "17:39", [], 30),
    ("2026-11-18", "17:40", [], 5),
    ("2026-11-18", "22:57", list(range(1, 9)), 2),      # race 9 off 7 minutes ago
    ("2026-11-18", "23:35", list(range(1, 10)), 5),     # last post + 45
    ("2026-11-18", "23:36", list(range(1, 10)), 30),
    ("2026-11-19", "00:10", list(range(1, 10)), 0),     # after midnight: yesterday's meeting, night rule
    ("2026-11-19", "09:00", list(range(1, 10)), 30),
])
def test_night_meeting(tmp_path, ticker_functions, day, clock, resulted, minutes):
    assert interval(tmp_path, ticker_functions, day, clock, NIGHT_MEETING, resulted) == minutes


@pytest.mark.parametrize("clock, minutes", [("03:00", 0), ("12:40", 30)])
def test_schedule_unreadable_falls_back_to_the_plain_clock(tmp_path, ticker_functions, clock, minutes):
    assert interval(tmp_path, ticker_functions, "2026-11-14", clock) == minutes


def test_schedule_without_post_times_is_not_a_meeting(tmp_path, ticker_functions):
    empty = {"date": "2026-11-14", "venue": "ST", "first_post": None, "last_post": None, "posts": {}}
    assert interval(tmp_path, ticker_functions, "2026-11-14", "13:00", empty) == 30


def test_test_mode_runs_the_race_day_clock(tmp_path, ticker_functions):
    assert interval(tmp_path, ticker_functions, "2026-11-13", "03:00", DAY_MEETING, FAKE_TEST_UNTIL=2000000) == 5


def test_due_results_counts_races_6_to_45_minutes_old_without_a_result(tmp_path, ticker_functions):
    need("jq")
    sched = json.dumps({**DAY_MEETING, "resulted": [3]})
    # 14:35 = 875 minutes: race 3 (13:55) has a result, race 4 (14:25) is 10 minutes old,
    # race 2 (13:00) is 95 minutes old, race 5 (14:55) has not run
    out = bash(tmp_path, CLOCK + ticker_functions + f"\ndue_results 875 '{sched}'\n")
    assert out == "1"


# ------------------------------------------------------------------ update: publish the site or not

GITHUB = r'''
gh() {
  if [ "$1" = run ] && [ "$2" = list ]; then
    [ -n "$FAKE_LAST" ] || return 1
    printf '%s\n' "$FAKE_LAST"        # commit, minutes ago, status, conclusion
  else
    echo "CALLED gh $*"
  fi
}
git() { echo "$FAKE_HEAD"; }
'''
HEAD, OLD = "a" * 40, "b" * 40


def publish(tmp_path, last, head=HEAD):
    script = GITHUB + step_script("update.yml", "Publish the site when something is new, at most every 10 minutes")
    out = bash(tmp_path, script, FAKE_LAST="\t".join(map(str, last)) if last else "", FAKE_HEAD=head)
    return "CALLED gh workflow run pages.yml" in out, out


@pytest.mark.parametrize("last, published", [
    ((HEAD, 3, "completed", "success"), False),      # this commit is already on the site
    ((HEAD, 300, "completed", "success"), False),    # however long ago
    ((HEAD, 1, "in_progress", ""), False),           # being published right now
    ((HEAD, 12, "completed", "failure"), True),      # the last publish failed: try again
    ((HEAD, 3, "completed", "failure"), False),      # ...but not more than every 10 minutes
    ((OLD, 12, "completed", "success"), True),       # new commit since the last publish
    ((OLD, 10, "completed", "success"), True),
    ((OLD, 9, "completed", "success"), False),       # new commit, published 9 minutes ago: wait
    ((OLD, 12, "in_progress", ""), True),
    (None, True),                                    # GitHub didn't answer: publish rather than go stale
])
def test_publish_decision(tmp_path, last, published):
    did, out = publish(tmp_path, last)
    assert did is published, out


def test_publish_says_why_it_did_not(tmp_path):
    assert "nothing new since the last publish" in publish(tmp_path, (HEAD, 3, "completed", "success"))[1]
    assert "published 9 minutes ago" in publish(tmp_path, (OLD, 9, "completed", "success"))[1]
