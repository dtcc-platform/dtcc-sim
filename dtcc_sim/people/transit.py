"""Scheduled transit itineraries from an archived national GTFS feed."""

import csv
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from heapq import heappop, heappush
from io import TextIOWrapper
from itertools import pairwise
from operator import attrgetter
from typing import NamedTuple
from zipfile import ZipFile

import numpy as np
from dtcc_core.model.object.people import TimedPath
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import LineString, box

from .inputs import _fetch
from .routing import stationary, timed_route
from .scenario import fingerprint

ARCHIVE = "gtfs-sverige-2-20191014.zip"
ARCHIVE_URL = (
    "https://data.samtrafiken.se/trafiklab/gtfs-sverige-2/2019/10/sweden-20191014.zip"
)
ARCHIVE_SHA256 = "c1d7d346a7ae0be4ddae64855e60b99ce06fa0a085c411e26228a66b79d7540b"
MAX_WAIT = 1800


def _rows(archive, table):
    if table in archive.namelist():
        with TextIOWrapper(archive.open(table), encoding="utf-8-sig") as stream:
            yield from csv.DictReader(stream)


def _active_services(archive, day):
    compact = day.strftime("%Y%m%d")
    weekday = (
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    )[day.weekday()]
    active = {
        row["service_id"]
        for row in _rows(archive, "calendar.txt")
        if row["start_date"] <= compact <= row["end_date"] and row[weekday] == "1"
    }
    for row in _rows(archive, "calendar_dates.txt"):
        if row["date"] == compact:
            if row["exception_type"] == "1":
                active.add(row["service_id"])
            elif row["exception_type"] == "2":
                active.discard(row["service_id"])
    return active


def _second(value):
    hour, minute, second = map(int, value.split(":"))
    return hour * 3600 + minute * 60 + second


class Call(NamedTuple):
    sequence: int
    stop: str
    arrival: int
    departure: int
    pickup: bool
    dropoff: bool


@dataclass(frozen=True, slots=True)
class Ride:
    trip: str
    calls: tuple[Call, ...]
    first: int
    last: int

    @property
    def departure(self):
        return self.calls[self.first].departure

    @property
    def arrival(self):
        return self.calls[self.last].arrival


def _vehicle_kind(route_type):
    code = int(route_type) if route_type.isdecimal() else 3
    if code == 0 or 900 <= code < 1000:
        return "tram"
    if code in {1, 2} or 100 <= code < 500:
        return "train"
    if code == 4 or 1000 <= code < 1300:
        return "ferry"
    return "bus"


class Shapes:
    """Vehicle geometry between stop areas: mapped tram rails, roads for buses, else straight."""

    def __init__(self, rails, roads=None, snap=60):
        keys = {}
        points = []
        self.graph = []
        for line in rails:
            previous = None
            for x, y, *_ in line:
                key = (round(x, 1), round(y, 1))
                if key not in keys:
                    keys[key] = len(points)
                    points.append((x, y, 0.0))
                    self.graph.append([])
                node = keys[key]
                if previous is not None and previous != node:
                    metres = float(
                        np.hypot(*np.subtract(points[node][:2], points[previous][:2]))
                    )
                    self.graph[node].append((previous, metres))
                    self.graph[previous].append((node, metres))
                previous = node
        self.points = np.asarray(points, dtype=float).reshape((-1, 3))
        self.tree = cKDTree(self.points[:, :2]) if len(points) else None
        self.roads = roads
        self.snap = snap
        self._cache = {}

    def between(self, kind, start, end):
        """Return a polyline from start to end, never longer than 2.5 times the straight line."""
        key = (kind, tuple(start), tuple(end))
        if key not in self._cache:
            direct = float(np.hypot(end[0] - start[0], end[1] - start[1]))
            path = None
            if kind == "tram":
                path = self._rail(start, end)
            elif kind == "bus" and self.roads is not None:
                route = self.roads.between(np.asarray(start), np.asarray(end))
                path = None if route is None else route.xyz
            if path is not None and len(path) > 1:
                path = np.vstack((start, path, end))
                length = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum()
                if length > 2.5 * direct + 50:
                    path = None
            self._cache[key] = (
                np.asarray([start, end], dtype=float) if path is None else path
            )
        return self._cache[key]

    def _rail(self, start, end):
        if self.tree is None:
            return None
        (da, a), (db, b) = (self.tree.query(point[:2]) for point in (start, end))
        if da > self.snap or db > self.snap:
            return None
        target = self.points[b, :2]
        cost = {a: 0.0}
        prior = {}
        queue = [(0.0, 0.0, a)]
        while queue:
            _, travel, node = heappop(queue)
            if node == b:
                nodes = [b]
                while node != a:
                    node = prior[node]
                    nodes.append(node)
                return self.points[nodes[::-1]]
            if travel > cost[node]:
                continue
            for neighbour, metres in self.graph[node]:
                total = travel + metres
                if total < cost.get(neighbour, float("inf")):
                    cost[neighbour] = total
                    prior[neighbour] = node
                    heappush(
                        queue,
                        (
                            total
                            + float(np.hypot(*(self.points[neighbour, :2] - target))),
                            total,
                            neighbour,
                        ),
                    )
        return None


class Timetable:
    """Indexed stop-area services, including permitted same-stop transfers."""

    def __init__(self, stops, runs, transfers, lines=None):
        self.lines = lines or {}
        self.shapes = None
        self.runs = {}
        pairs = defaultdict(list)
        self.from_stop = defaultdict(set)
        self.to_stop = defaultdict(set)
        self.service_trips = 0
        for trip, calls in runs.items():
            calls = tuple(sorted(calls))
            if any(a.arrival > a.departure for a in calls) or any(
                a.departure > b.arrival for a, b in pairwise(calls)
            ):
                continue
            if len(calls) >= 2:
                self.service_trips += 1
                self.runs[trip] = calls
            for i, board in enumerate(calls):
                if not board.pickup:
                    continue
                for j in range(i + 1, len(calls)):
                    alight = calls[j]
                    if (
                        alight.dropoff
                        and board.stop != alight.stop
                        and 0 <= board.departure < alight.arrival < 86400
                    ):
                        pairs[board.stop, alight.stop].append(Ride(trip, calls, i, j))
                        self.from_stop[board.stop].add(alight.stop)
                        self.to_stop[alight.stop].add(board.stop)
        self.stops = stops
        self.stop_ids = sorted(self.from_stop.keys() | self.to_stop.keys())
        self.tree = cKDTree(
            np.asarray([stops[stop][:2] for stop in self.stop_ids]).reshape((-1, 2))
        )
        self.pairs = {}
        for pair, rides in pairs.items():
            rides.sort(key=attrgetter("departure", "arrival", "trip"))
            self.pairs[pair] = (tuple(ride.departure for ride in rides), rides)
        self.transfers = transfers
        self._transfer_options = {}

    @classmethod
    def read(cls, filename, day, bounds):
        """Read active trips and their calls inside the buffered region."""
        with ZipFile(filename) as archive:
            services = _active_services(archive, day)
            if not services:
                raise ValueError(f"No GTFS services for {day} in {filename}")
            routes = {
                row["route_id"]: (
                    _vehicle_kind(row.get("route_type", "3")),
                    row.get("route_short_name") or row.get("route_long_name", ""),
                )
                for row in _rows(archive, "routes.txt")
            }
            lines = {}
            for row in _rows(archive, "trips.txt"):
                if row["service_id"] in services:
                    lines[row["trip_id"]] = routes.get(
                        row.get("route_id", ""), ("bus", "")
                    )
            active_trips = set(lines)
            projector = Transformer.from_crs(4326, 3006, always_xy=True)
            west, south, east, north = bounds
            candidates = [
                row
                for row in _rows(archive, "stops.txt")
                if row.get("location_type", "") in {"", "0"}
            ]
            xs, ys = projector.transform(
                [float(row["stop_lon"]) for row in candidates],
                [float(row["stop_lat"]) for row in candidates],
            )
            stops = {
                row["stop_id"]: (x, y, 0.0)
                for row, x, y in zip(candidates, xs, ys)
                if west - 600 <= x <= east + 600 and south - 600 <= y <= north + 600
            }
            runs = defaultdict(list)
            if stops:
                with TextIOWrapper(
                    archive.open("stop_times.txt"), encoding="utf-8-sig"
                ) as stream:
                    reader = csv.reader(stream)
                    fields = {name: i for i, name in enumerate(next(reader))}
                    trip, stop, arrival, departure, seq = (
                        fields[k]
                        for k in (
                            "trip_id",
                            "stop_id",
                            "arrival_time",
                            "departure_time",
                            "stop_sequence",
                        )
                    )
                    pickup, dropoff = (
                        fields.get("pickup_type"),
                        fields.get("drop_off_type"),
                    )
                    for row in reader:
                        if (
                            row[stop] in stops
                            and row[trip] in active_trips
                            and row[arrival]
                            and row[departure]
                        ):
                            runs[row[trip]].append(
                                Call(
                                    int(row[seq]),
                                    row[stop],
                                    _second(row[arrival]),
                                    _second(row[departure]),
                                    pickup is None or row[pickup] in {"", "0"},
                                    dropoff is None or row[dropoff] in {"", "0"},
                                )
                            )
            transfers = {}
            for row in _rows(archive, "transfers.txt"):
                stop = row["from_stop_id"]
                if stop != row["to_stop_id"] or stop not in stops:
                    continue
                if any(
                    row.get(key)
                    for key in (
                        "from_trip_id",
                        "to_trip_id",
                        "from_route_id",
                        "to_route_id",
                    )
                ):
                    continue
                if row["transfer_type"] == "3":
                    transfers[stop] = None
                elif row["transfer_type"] == "2":
                    transfers[stop] = int(row["min_transfer_time"])
        return cls(stops, runs, transfers, {trip: lines[trip] for trip in runs})

    def _near(self, xyz, boarding):
        eligible = self.from_stop if boarding else self.to_stop
        near = self.tree.query_ball_point(xyz[:2], r=550)
        return sorted(
            (self.stop_ids[i] for i in near if self.stop_ids[i] in eligible),
            key=lambda stop: (
                (self.stops[stop][0] - xyz[0]) ** 2
                + (self.stops[stop][1] - xyz[1]) ** 2,
                stop,
            ),
        )[:3]

    def _rides(self, start, end, ready):
        times, rides = self.pairs.get((start, end), ((), ()))
        return rides[bisect_left(times, ready) : bisect_right(times, ready + MAX_WAIT)]

    def _changes(self, start, end):
        key = (start, end)
        if key not in self._transfer_options:
            ax, ay, _ = self.stops[start]
            bx, by, _ = self.stops[end]
            eligible = self.from_stop[start] & self.to_stop[end]
            self._transfer_options[key] = sorted(
                (
                    stop
                    for stop in eligible
                    if self.transfers.get(stop, 120) is not None
                ),
                key=lambda stop: (
                    (self.stops[stop][0] - ax) ** 2
                    + (self.stops[stop][1] - ay) ** 2
                    + (self.stops[stop][0] - bx) ** 2
                    + (self.stops[stop][1] - by) ** 2,
                    stop,
                ),
            )[:6]
        return self._transfer_options[key]

    def itinerary(self, origin, destination, depart, speed, purpose, journey, router):
        """Time walk access and egress over the pedestrian network."""
        if not self.stop_ids:
            return None
        starts = {}
        for stop in self._near(origin, boarding=True):
            route = router.between(origin, self.stops[stop])
            if route is not None:
                walk = timed_route(
                    route, depart, speed, purpose, journey, signal_wait=12
                )
                if walk:
                    starts[stop] = walk
            elif np.array_equal(origin, self.stops[stop]):
                starts[stop] = []

        ends = {}
        for stop in self._near(destination, boarding=False):
            route = router.between(self.stops[stop], destination)
            if route is not None or np.array_equal(self.stops[stop], destination):
                ends[stop] = route

        best = None
        best_arrival = 86400
        for changes in (False, True):
            for start, access in starts.items():
                ready = access[-1].end if access else depart
                for end, exit_route in ends.items():
                    if start == end:
                        continue
                    option = min(
                        self._options(start, end, ready, changes),
                        key=lambda option: option[-1].arrival,
                        default=(),
                    )
                    if not option or option[-1].arrival >= best_arrival:
                        continue
                    arrival = option[-1].arrival
                    egress = []
                    if exit_route is not None:
                        egress = timed_route(
                            exit_route, arrival, speed, purpose, journey, signal_wait=12
                        )
                        if not egress:
                            continue
                        arrival = egress[-1].end
                    if arrival < best_arrival:
                        best = (access, option, egress)
                        best_arrival = arrival
            if best is not None:
                break
        if best is None:
            return None
        access, rides, egress = best
        stages = list(access)
        ready = stages[-1].end if stages else depart
        for ride in rides:
            stop = ride.calls[ride.first].stop
            if ready < ride.departure:
                stages.append(
                    stationary(
                        "wait",
                        purpose,
                        f"gtfs:{stop}",
                        self.stops[stop],
                        ready,
                        ride.departure,
                        journey,
                    )
                )
            stages.append(self._path(ride, purpose, journey))
            ready = ride.arrival
        return [*stages, *egress]

    def _options(self, start, end, ready, changes):
        if not changes:
            yield from ((ride,) for ride in self._rides(start, end, ready))
            return
        for stop in self._changes(start, end):
            minimum = self.transfers.get(stop, 120)
            for first in self._rides(start, stop, ready):
                for second in self._rides(stop, end, first.arrival + minimum):
                    if first.trip != second.trip:
                        yield first, second

    def _track(self, trip, start, end, depart, arrive):
        """Timed vertices from one stop area to the next at constant speed."""
        kind = self.lines.get(trip, ("bus", ""))[0]
        a, b = self.stops[start], self.stops[end]
        path = (
            self.shapes.between(kind, a, b)
            if self.shapes is not None
            else np.asarray([a, b], dtype=float)
        )
        steps = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
        keep = np.r_[True, steps > 0.01]
        path = path[keep]
        distance = np.r_[0.0, np.cumsum(steps[keep[1:]])]
        if len(path) < 2 or distance[-1] <= 0:
            return [(arrive, b)]
        times = depart + (arrive - depart) * distance / distance[-1]
        return [
            (float(t), tuple(map(float, xyz)))
            for t, xyz in zip(times[1:-1], path[1:-1])
            if depart < t < arrive
        ] + [(arrive, b)]

    def _samples(self, trip, calls, first, last):
        board, alight = calls[first], calls[last]
        samples = [[board.departure, *self.stops[board.stop]]]
        edges = []
        previous = board
        for call in calls[first + 1 : last + 1]:
            edge = f"gtfs:{trip}:{call.sequence}"
            if call.arrival > samples[-1][0]:
                for second, xyz in self._track(
                    trip, previous.stop, call.stop, samples[-1][0], call.arrival
                ):
                    if second > samples[-1][0]:
                        samples.append([second, *xyz])
                        edges.append(edge)
            if call != alight and call.departure > samples[-1][0]:
                samples.append([call.departure, *self.stops[call.stop]])
                edges.append(edge)
            previous = call
        if not edges:
            return samples, edges
        samples[-1] = [alight.arrival, *self.stops[alight.stop]]
        edges[-1] = f"gtfs:{trip}:{alight.sequence}"
        return samples, edges

    def _path(self, ride, purpose, journey):
        samples, edges = self._samples(ride.trip, ride.calls, ride.first, ride.last)
        alight = ride.calls[ride.last]
        return TimedPath(
            "transit", purpose, samples, f"gtfs:{alight.stop}", journey, tuple(edges)
        )

    def vehicles(self, bounds, margin=150):
        """Scheduled vehicle paths whose track passes within the margin of the bounds."""
        west, south, east, north = bounds
        view = box(west - margin, south - margin, east + margin, north + margin)
        result = []
        for trip, calls in sorted(self.runs.items()):
            last_today = bisect_left(calls, 86400, key=attrgetter("arrival")) - 1
            if calls[0].departure >= 86400 or last_today < 1:
                continue
            samples, edges = self._samples(trip, calls, 0, last_today)
            if not edges:
                continue
            if not view.intersects(LineString([(x, y) for _, x, y, _ in samples])):
                continue
            kind, line = self.lines.get(trip, ("bus", ""))
            result.append({"id": trip, "kind": kind, "line": line, "samples": samples})
        return result


def load_timetable(root, service_date, bounds):
    """Verify the immutable source before interpreting scheduled service."""
    filename = root / ARCHIVE
    _fetch(filename, ARCHIVE_URL)
    digest = fingerprint(filename)
    if digest != ARCHIVE_SHA256:
        raise ValueError(f"Unexpected GTFS archive checksum for {filename}: {digest}")
    source = {
        "name": "Samtrafiken Trafiklab GTFS Sverige 2",
        "url": ARCHIVE_URL,
        "vintage": "2019-10-14",
        "archive": filename.name,
        "sha256": digest,
        "terms": "CC0 1.0",
        "selection": {
            "service_date": service_date,
            "region_bounds": list(bounds),
            "stop_halo_m": 600,
        },
    }
    return Timetable.read(filename, date.fromisoformat(service_date), bounds), source
