"""Reading HKJC's overhead position pictures (hkjc/aerial.py).

The pictures here are drawn by the test itself, in the colours and sizes of the real ones and
with the number shapes of aerial_digits.json, then saved as JPEG like HKJC's. Whether the
reader agrees with real pictures was checked once on a whole season (see the module's notes);
these tests pin down what it must keep doing: find circles that overlap or are cut by the
frame, read their numbers, and store a race once.
"""
import io
import itertools
from datetime import date

import numpy as np
import pytest
from PIL import Image

from helpers import make_card, make_result
from hkjc import aerial, store

GRASS, BLUE, EDGE = (72, 122, 48), (4, 113, 211), (10, 78, 135)
SHAPES = aerial.digit_shapes()


def draw(panels, half=False, quality=95):
    """panels: four lists of (horse number, x, y in the panel), drawn in that order, so a later
    circle lies on top of an earlier one. Returns the picture as JPEG bytes."""
    img = np.zeros((aerial.PANEL * 4, aerial.WIDTH, 3), dtype=np.uint8)
    img[:] = GRASS
    yy, xx = np.mgrid[0:aerial.PANEL, 0:aerial.WIDTH]
    for p, circles in enumerate(panels):
        view = img[p * aerial.PANEL:(p + 1) * aerial.PANEL]
        for no, x, y in circles:
            d = np.hypot(yy - y, xx - x)
            view[d <= 20.5] = EDGE
            view[d <= 18.2] = BLUE
            shape = SHAPES[no] > 0.5
            for dy, dx in zip(*np.nonzero(shape)):
                py, px = int(round(y)) + dy - aerial.G, int(round(x)) + dx - aerial.G
                if 0 <= py < aerial.PANEL and 0 <= px < aerial.WIDTH:
                    view[py, px] = (255, 255, 255)
    im = Image.fromarray(img)
    if half:
        im = im.resize((aerial.WIDTH // 2, aerial.PANEL * 2), Image.BICUBIC)
    out = io.BytesIO()
    im.save(out, "JPEG", quality=quality, subsampling=0)
    return out.getvalue()


def spread(numbers, x0=120, y0=150, step=62):
    """Circles standing apart, in two rows."""
    return [(no, x0 + step * (i // 2), y0 + 70 * (i % 2)) for i, no in enumerate(numbers)]


def as_dict(panel):
    return {row[0]: row for row in panel}


def close(row, x, y, by=1.6):
    return abs(row[1] - x) <= by and abs(row[2] - y) <= by


# ------------------------------------------------------------------ the picture itself

def test_every_number_from_1_to_14_is_read_where_it_stands():
    numbers = list(range(1, 15))
    panels = [spread(numbers), spread(numbers[::-1], x0=200), spread(numbers, y0=250), spread(numbers[::2] + numbers[1::2])]
    read = aerial.read_picture(aerial.decode(draw(panels)), numbers)
    for want, got in zip(panels, read):
        got = as_dict(got)
        assert sorted(got) == numbers
        for no, x, y in want:
            assert close(got[no], x, y), (no, got[no])
            assert got[no][3] >= aerial.WEAK_FIT and got[no][4] == 1.0          # read surely, seen whole


def test_circles_lying_over_one_another_are_told_apart():
    # 8 partly under 3, 1 under 13: drawn first, so the later circle hides part of the number
    panel = [(8, 300, 200), (3, 322, 190), (1, 420, 260), (13, 404, 280), (5, 600, 150), (7, 150, 330)]
    numbers = [1, 3, 5, 7, 8, 13]
    read = as_dict(aerial.read_picture(aerial.decode(draw([panel, [], [], []])), numbers)[0])
    assert sorted(read) == numbers
    for no, x, y in panel:
        assert close(read[no], x, y, by=2.5), (no, read[no])
    assert read[8][4] < 1.0 and read[1][4] < 1.0                               # the two underneath
    assert read[3][4] == 1.0 and read[13][4] == 1.0 and read[5][4] == 1.0


def test_a_circle_cut_by_the_frame_is_still_found():
    panel = [(4, 6, 200), (9, 794, 120), (2, 400, 444), (6, 400, 225)]         # left, right and bottom edges
    read = as_dict(aerial.read_picture(aerial.decode(draw([[], panel, [], []])), [2, 4, 6, 9])[1])
    assert sorted(read) == [2, 4, 6, 9]
    for no, x, y in panel:
        assert close(read[no], x, y, by=3.0), (no, read[no])


def test_a_horse_outside_the_frame_is_simply_not_there():
    numbers = [1, 2, 3, 4, 5, 6, 7, 8]
    panels = [spread(numbers[:5]), spread(numbers), spread(numbers), spread(numbers[:7])]
    read = aerial.read_picture(aerial.decode(draw(panels)), numbers)
    assert [len(p) for p in read] == [5, 8, 8, 7]
    assert sorted(as_dict(read[0])) == [1, 2, 3, 4, 5]


def test_panels_do_not_borrow_from_each_other():
    numbers = [1, 2, 3]
    panels = [[(1, 100, 430)], [(2, 100, 20)], [], [(3, 700, 225)]]            # 1 and 2 either side of a border
    read = aerial.read_picture(aerial.decode(draw(panels)), numbers)
    assert [sorted(as_dict(p)) for p in read] == [[1], [2], [], [3]]


def test_a_half_size_picture_is_enlarged_and_read():
    numbers = [1, 2, 4, 7, 10, 11, 12, 14]
    panels = [spread(numbers, step=90), spread(numbers[::-1], step=90), [], spread(numbers, step=90)]
    read = aerial.read_picture(aerial.decode(draw(panels, half=True)), numbers)
    for want, got in zip(panels, read):
        got = as_dict(got)
        assert sorted(got) == sorted(no for no, _, _ in want)
        for no, x, y in want:
            assert close(got[no], x, y, by=3.0), (no, got[no])


def test_only_the_runners_of_the_race_are_looked_for():
    # 3 and 8 look alike; with 8 not in the race the circle can only be 3
    read = aerial.read_picture(aerial.decode(draw([[(3, 300, 200), (5, 500, 200)], [], [], []])), [3, 5, 6])
    assert sorted(as_dict(read[0])) == [3, 5]


@pytest.mark.parametrize("data", [b"", b"not a picture", b"\xff\xd8 broken"])
def test_what_is_not_a_picture_is_refused(data):
    assert aerial.decode(data) is None


def test_a_picture_of_another_shape_is_refused():
    out = io.BytesIO()
    Image.new("RGB", (720, 602), GRASS).save(out, "JPEG")                      # HKJC's photo version
    assert aerial.decode(out.getvalue()) is None


def test_number_shapes_cover_every_runner_number():
    assert sorted(SHAPES) == list(range(1, 15))
    assert all(s.shape == (2 * aerial.G + 1, 2 * aerial.G + 1) and 0 <= s.min() and s.max() <= 1 for s in SHAPES.values())


def test_assign_finds_the_cheapest_pairing():
    rng = np.random.default_rng(7)
    for n, m in ((1, 1), (2, 4), (4, 4), (5, 7)):
        for _ in range(20):
            cost = rng.random((n, m)).tolist()
            got = aerial.assign(cost)
            assert len(set(got)) == n
            best = min(sum(cost[i][j] for i, j in enumerate(cols)) for cols in itertools.permutations(range(m), n))
            assert sum(cost[i][j] for i, j in enumerate(got)) == pytest.approx(best)


@pytest.mark.parametrize("day, n, tail", [
    ("2026-11-11", 1, "26_27/20261111/20261111R1_Aerial_L.jpg"),
    ("2027-02-03", 10, "26_27/20270203/20270203R10_Aerial_L.jpg"),
    ("2027-09-05", 3, "27_28/20270905/20270905R3_Aerial_L.jpg"),
])
def test_picture_address_follows_the_season(day, n, tail):
    assert aerial.picture_url(day, n).endswith("/RaceResult/" + tail)


# ------------------------------------------------------------------ keeping what was read

DAY, VENUE = "2026-11-11", "HV"
NUMBERS = list(range(1, 9))


@pytest.fixture
def meeting(sandbox, monkeypatch):
    """Three races run on 11 Nov; "today" is the day after."""
    card = make_card(DAY, VENUE, {1: "18:40", 2: "19:10", 3: "19:40"})
    store.save_results([make_result(r, NUMBERS, codes={}) for r in card["races"]])
    monkeypatch.setattr(aerial, "today_hkt", lambda: date(2026, 11, 12))
    picture = draw([spread(NUMBERS), spread(NUMBERS), spread(NUMBERS), spread(NUMBERS)])
    asked = []

    def fetch(day, race_no):
        asked.append((day, race_no))
        return picture if race_no in fetch.there else None

    fetch.there, fetch.asked = {1, 2}, asked
    return fetch


def saved():
    return store.read_json(aerial.FOLDER / f"{DAY}_{VENUE}.json")


def test_pictures_that_are_there_are_read_and_the_rest_waited_for(meeting):
    assert aerial.update(log=lambda *_: None, fetch=meeting) == (2, 1)
    doc = saved()
    assert (doc["date"], doc["venue"], sorted(doc["races"])) == (DAY, VENUE, ["1", "2"])
    race = doc["races"]["1"]
    assert race["ran"] == 8 and [len(p) for p in race["panels"]] == [8, 8, 8, 8]
    assert aerial.load(DAY, VENUE)[2] == doc["races"]["2"]["panels"]


def test_a_race_is_read_once_and_a_missing_picture_is_asked_for_again(meeting):
    aerial.update(log=lambda *_: None, fetch=meeting)
    first = saved()["races"]["1"]
    meeting.asked.clear()
    meeting.there = {1, 2, 3}
    assert aerial.update(log=lambda *_: None, fetch=meeting) == (1, 0)
    assert meeting.asked == [(DAY, 3)]
    assert saved()["races"]["1"] == first and sorted(saved()["races"]) == ["1", "2", "3"]


def test_nothing_is_asked_on_race_day_or_long_after(meeting, monkeypatch):
    monkeypatch.setattr(aerial, "today_hkt", lambda: date(2026, 11, 11))       # the pictures come a day later
    assert aerial.update(log=lambda *_: None, fetch=meeting) == (0, 0)
    monkeypatch.setattr(aerial, "today_hkt", lambda: date(2026, 11, 22))       # 11 days on: given up
    assert aerial.update(log=lambda *_: None, fetch=meeting) == (0, 0)
    assert meeting.asked == [] and saved() is None


def test_a_fault_on_one_picture_does_not_stop_the_others(meeting):
    def fetch(day, race_no):
        if race_no == 1:
            raise RuntimeError("picture of race 1: Timeout")
        return meeting(day, race_no)

    said = []
    assert aerial.update(log=said.append, fetch=fetch) == (1, 2)
    assert sorted(saved()["races"]) == ["2"] and any("Timeout" in s for s in said)


def test_horses_that_did_not_start_are_not_looked_for(sandbox, monkeypatch):
    card = make_card(DAY, VENUE, {1: "18:40"})
    store.save_results([make_result(card["races"][0], [1, 2, 3, 4, 5, 6, 7], codes={8: "WV"})])
    monkeypatch.setattr(aerial, "today_hkt", lambda: date(2026, 11, 12))
    picture = draw([spread(NUMBERS[:7])] * 4)
    aerial.update(log=lambda *_: None, fetch=lambda day, n: picture)
    race = saved()["races"]["1"]
    assert race["ran"] == 7 and all(sorted(r[0] for r in p) == NUMBERS[:7] for p in race["panels"])


def test_saved_pictures_can_be_read_from_a_folder(sandbox, tmp_path):
    card = make_card(DAY, VENUE, {1: "18:40", 2: "19:10"})
    store.save_results([make_result(r, NUMBERS) for r in card["races"]])
    folder = tmp_path / "pictures"
    folder.mkdir()
    (folder / f"{DAY}_{VENUE}_R1.jpg").write_bytes(draw([spread(NUMBERS)] * 4))
    (folder / f"{DAY}_{VENUE}_R2.jpg").write_bytes(b"not a picture")
    assert aerial.read_folder(folder, log=lambda *_: None) == 1
    assert sorted(saved()["races"]) == ["1"]
