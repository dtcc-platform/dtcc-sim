from datetime import date
from itertools import pairwise
from zipfile import ZipFile

import numpy as np
from dtcc_core.model import RoadNetwork
from pyproj import Transformer

from dtcc_sim.people.routing import Router
from dtcc_sim.people.transit import Call, Timetable


def test_vehicle_keeps_ride_before_midnight_when_service_continues_next_day():
    stops = {"a": (0, 0, 0), "b": (700, 0, 0), "c": (1400, 0, 0)}
    timetable = Timetable(
        stops,
        {
            "last": [
                Call(1, "a", 86300, 86300, True, False),
                Call(2, "b", 86390, 86395, True, True),
                Call(3, "c", 86430, 86430, False, True),
            ]
        },
        {},
    )
    (ride,) = timetable._rides("a", "b", 86280)
    (vehicle,) = timetable.vehicles((0, -10, 800, 10))
    np.testing.assert_array_equal(
        vehicle["samples"], timetable._path(ride, "home", "trip").samples
    )


def test_gtfs_scheduled_walk_wait_ride_and_transfer(tmp_path):
    x, y = 319800, 6397900
    points = [(x + i * 800, y) for i in range(4)]
    lonlat = Transformer.from_crs(3006, 4326, always_xy=True)
    archive = tmp_path / "schedule.zip"
    with ZipFile(archive, "w") as feed:
        feed.writestr(
            "stops.txt",
            "stop_id,stop_lat,stop_lon\n"
            + "".join(
                f"{stop},{lat},{lon}\n"
                for stop, (lon, lat) in zip(
                    "abcd", [lonlat.transform(*p) for p in points]
                )
            ),
        )
        feed.writestr(
            "calendar.txt",
            "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
            "weekday,1,1,1,1,1,0,0,20191001,20191030\n"
            "removed,1,1,1,1,1,0,0,20191001,20191030\n",
        )
        feed.writestr(
            "calendar_dates.txt",
            "service_id,date,exception_type\nextra,20191015,1\nremoved,20191015,2\n",
        )
        feed.writestr(
            "trips.txt",
            "trip_id,service_id\nlocal,weekday\nextra,extra\nremoved,removed\nexpress,weekday\n",
        )
        feed.writestr(
            "stop_times.txt",
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time,pickup_type,drop_off_type\n"
            "local,a,1,08:02:00,08:02:00,0,1\n"
            "local,b,2,08:03:00,08:03:30,0,0\n"
            "local,c,3,08:05:00,08:05:00,1,0\n"
            "extra,c,1,08:09:00,08:09:00,0,1\n"
            "extra,d,2,08:13:00,08:13:00,1,0\n"
            "removed,a,1,08:01:00,08:01:00,0,1\n"
            "removed,d,2,08:10:00,08:10:00,1,0\n"
            "express,a,1,08:01:00,08:01:00,1,1\n"
            "express,d,2,08:08:00,08:08:00,1,0\n",
        )
        feed.writestr(
            "transfers.txt",
            "from_stop_id,to_stop_id,transfer_type,min_transfer_time\nc,c,2,180\n",
        )
    timetable = Timetable.read(
        archive, date(2019, 10, 15), (x - 50, y - 50, x + 2450, y + 50)
    )
    assert timetable.service_trips == 3
    assert timetable.transfers == {"c": 180}
    vertices = [(x - 30, y), *points, (x + 2440, y)]
    router = Router(
        RoadNetwork(
            vertices=np.array(vertices, dtype=float),
            edges=np.array(list(pairwise(range(len(vertices))))),
            attributes={"highway": ["footway"] * (len(vertices) - 1)},
        )
    )
    origin, destination = router.xyz[0], router.xyz[-1]
    stages = timetable.itinerary(
        origin, destination, 28800, 1.0, "work", "trip", router
    )
    assert [stage.state for stage in stages] == [
        "walk",
        "wait",
        "transit",
        "wait",
        "transit",
        "walk",
    ]
    assert [(stages[i].start, stages[i].end) for i in (2, 3, 4)] == [
        (28920, 29100),
        (29100, 29340),
        (29340, 29580),
    ]
    assert list(stages[2].samples[:, 0]) == [28920, 28980, 29010, 29100]
    assert stages[3].place_id == "gtfs:c"
    assert stages[4].edge_ids == ("gtfs:extra:2",)
    for before, after in pairwise(stages):
        assert before.end == after.start
        np.testing.assert_array_equal(before.samples[-1, 1:], after.samples[0, 1:])
    assert (
        timetable.itinerary(origin, destination, 29000, 1.0, "work", "late", router)
        is None
    )

    too_slow = Timetable(timetable.stops, {}, {})
    assert (
        too_slow.itinerary(origin, destination, 28800, 1.0, "work", "empty", router)
        is None
    )
