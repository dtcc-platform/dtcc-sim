from collections import Counter
from itertools import pairwise

import numpy as np
from dtcc_core.model import Person, RoadNetwork

from dtcc_sim.people.inputs import Anchor, building_height
from dtcc_sim.people.population import (
    CAMPUS_BLOCKS,
    Destinations,
    _escorted_day,
    _schedule,
    _stay,
)
from dtcc_sim.people.routing import Router, timed_route
from dtcc_sim.people.scenario import rng
from dtcc_sim.people.transit import Call, Shapes, Timetable

X, Y = 319800, 6397900


def street(length=600, step=50):
    count = length // step + 1
    return Router(
        RoadNetwork(
            vertices=np.array([[X + i * step, Y] for i in range(count)], dtype=float),
            edges=np.array([[i, i + 1] for i in range(count - 1)]),
            attributes={"highway": ["footway"] * (count - 1)},
        )
    )


def anchor(name, metres, kind, capacity=100):
    return Anchor(name, np.array([X + metres, Y, 0.0]), kind, capacity)


def test_escorted_child_walks_with_the_adult_and_waits_at_school():
    router = street()
    home, school = anchor("home", 0, "home"), anchor("school", 300, "education")
    there = router.between(home.xyz, school.xyz)
    back = router.between(school.xyz, home.xyz)
    drop = timed_route(there, 27000, 1.3, "escort", "adult-0")
    drop_back = timed_route(back, drop[-1].end + 400, 1.3, "home", "adult-0-return")
    fetch = timed_route(there, 58000, 1.3, "escort", "adult-1")
    pickup = timed_route(back, fetch[-1].end + 300, 1.3, "home", "adult-1-return")
    paths, regular, trips = _escorted_day(
        "child", home, school, [(drop, drop_back, "walk"), (fetch, pickup, "walk")]
    )
    child = Person(id="child", paths=paths)
    child._validate_day()
    assert regular == "school"
    walks = [path for path in paths if path.state == "walk"]
    assert np.array_equal(walks[0].samples, drop[0].samples)
    assert np.array_equal(walks[-1].samples, pickup[-1].samples)
    at_school = paths[2]
    assert (at_school.state, at_school.purpose, at_school.place_id) == (
        "indoor",
        "education",
        "school",
    )
    assert at_school.start == drop[-1].end and at_school.end == pickup[0].start
    assert [purpose for _, purpose, _ in trips] == ["education", "home"]


def test_escort_needs_both_drop_off_and_pick_up():
    assert _escorted_day("child", None, None, []) is None


def test_campus_day_keeps_teaching_blocks_and_walks_between_buildings():
    router = street()
    halls = [anchor(f"hall-{i}", 100 + 100 * i, "study") for i in range(3)]
    cafe = anchor("cafe", 550, "leisure")
    sites = {
        "study": Destinations(halls),
        "lunch": Destinations([cafe]),
    }
    moves = Counter()
    for seed in range(12):
        diagnostics = Counter()
        arrive = CAMPUS_BLOCKS[0][0] - 400
        leave = CAMPUS_BLOCKS[2][1] + 300
        paths, _, trips, end = _stay(
            f"student-{seed}",
            "study",
            halls[0],
            arrive,
            leave,
            (0, 2),
            1.3,
            router,
            sites,
            rng(seed, "campus"),
            diagnostics,
        )
        Person(id="student", paths=tuple(paths))._validate_day()
        assert paths[0].start == arrive and paths[-1].end == end >= leave
        for before, after in pairwise(paths):
            assert before.end == after.start
        for path in paths:
            if path.state == "walk":
                # Moves happen in the breaks, never through a teaching block.
                assert not any(
                    start + 60 < second < stop - 60
                    for start, stop in CAMPUS_BLOCKS[:3]
                    for second in (path.start, path.end)
                )
        moves.update(purpose for _, purpose, _ in trips)
        moves.update(diagnostics)
    assert moves["lunch"] > 0 and moves["campus_room_changes"] > 0


def test_exhausted_job_capacity_overflows_instead_of_cancelling_work():
    jobs = Destinations([anchor("office", 200, "work", capacity=1)])
    stream = rng(1, "work")
    first = jobs.choose(np.array([X, Y, 0.0]), "walk", stream, allow_overflow=True)
    jobs.reserve(first)
    second = jobs.choose(np.array([X, Y, 0.0]), "walk", stream, allow_overflow=True)
    jobs.reserve(second)
    assert first.id == second.id == "office"
    assert jobs.overflow == 1


def test_trams_follow_mapped_rails_and_buses_fall_back_to_straight_lines():
    rails = [np.array([[X, Y, 0], [X + 400, Y, 0], [X + 400, Y + 300, 0]], dtype=float)]
    shapes = Shapes(rails)
    path = shapes.between("tram", (X + 5, Y + 3, 0.0), (X + 398, Y + 298, 0.0))
    assert any(np.allclose(point[:2], (X + 400, Y)) for point in path)
    straight = shapes.between("bus", (X, Y, 0.0), (X + 400, Y + 300, 0.0))
    assert len(straight) == 2


def test_vehicle_paths_share_rider_geometry():
    stops = {"a": (X, Y, 0.0), "b": (X + 400, Y + 300, 0.0)}
    timetable = Timetable(
        stops,
        {
            "trip": [
                Call(1, "a", 28800, 28820, True, False),
                Call(2, "b", 29000, 29000, False, True),
            ]
        },
        {},
        {"trip": ("tram", "7")},
    )
    timetable.shapes = Shapes(
        [np.array([[X, Y, 0], [X + 400, Y, 0], [X + 400, Y + 300, 0]], dtype=float)]
    )
    (vehicle,) = timetable.vehicles((X - 10, Y - 10, X + 10, Y + 10))
    assert (vehicle["kind"], vehicle["line"]) == ("tram", "7")
    samples = np.asarray(vehicle["samples"])
    assert np.all(np.diff(samples[:, 0]) > 0)
    assert any(np.allclose(point, (X + 400, Y)) for point in samples[:, 1:3])
    (ride,) = timetable.pairs["a", "b"][1]
    rider = timetable._path(ride, "work", "journey")
    assert np.array_equal(rider.samples, samples)
    assert timetable.vehicles((X + 2000, Y + 2000, X + 2100, Y + 2100)) == []


def test_building_heights_prefer_mapped_values():
    assert building_height({"height": "14.5 m", "building:levels": "3"}) == (
        14.5,
        "mapped height",
    )
    assert building_height({"building:levels": "4", "roof:levels": "1"}) == (
        13.5,
        "mapped levels",
    )
    assert building_height({"building": "university"}) == (9.0, "inferred floors")
    assert building_height({"building": "garage"}) == (3.0, "inferred floors")


def test_vehicle_runs_without_movement_are_skipped():
    stops = {"a": (X, Y, 0.0), "b": (X + 400, Y, 0.0)}
    timetable = Timetable(
        stops,
        {
            "still": [
                Call(1, "a", 28800, 28800, True, True),
                Call(2, "b", 28800, 28800, True, True),
            ]
        },
        {},
    )
    assert timetable.vehicles((X - 10, Y - 10, X + 500, Y + 10)) == []


def test_failed_commute_leaves_the_job_available():
    jobs = Destinations([anchor("office", 300, "work", capacity=1)])
    sites = {"work": jobs, "lunch": Destinations([])}
    survey = {
        "table_1_annual_thousand": {"work": [1, 0, 0, 0]},
        "table_5_daily_rate_and_95ci": {"shopping": [0], "leisure": [0]},
    }
    home, stops = anchor("home", 0, "home"), Destinations([])
    for router, expected in [(street(100), 1), (street(), 0)]:
        _, _, trips, _ = _schedule(
            "adult",
            30,
            "work",
            home,
            sites,
            stops,
            router,
            survey,
            42,
            Counter(),
            attend=True,
        )
        assert bool(trips) == (expected == 0)
        assert jobs.remaining[0] == expected
    assert jobs.overflow == 0


def test_vehicle_crossing_view_between_stops_is_included():
    timetable = Timetable(
        {"a": (X, Y, 0.0), "b": (X + 400, Y, 0.0)},
        {
            "bus": [
                Call(1, "a", 28800, 28800, True, False),
                Call(2, "b", 29000, 29000, False, True),
            ]
        },
        {},
    )
    (vehicle,) = timetable.vehicles((X + 180, Y - 20, X + 220, Y + 20), margin=0)
    assert vehicle["id"] == "bus"
