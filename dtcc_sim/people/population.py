"""Regional households, activity choices and street trajectories."""

import json
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

import geopandas as gpd
import numpy as np
from dtcc_core.io.data.deso import DESO_AGE_BANDS, DESO_HOUSEHOLD_TYPES
from dtcc_core.model import Person
from dtcc_core.model.object.people import TimedPath
from openpyxl import load_workbook
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, box
from shapely.strtree import STRtree

from .diaries import ARCHIVE as DIARY_ARCHIVE
from .diaries import DEPARTURE_BIN_SECONDS, load_diaries
from .inputs import Anchor, archives, read_osm, select_region
from .routing import Router, stationary, timed_route
from .scenario import fingerprint, rng, simulation

MODES = ("walk", "cycle", "car", "transit")
PRIMARY_PARTICIPATION = 0.86
# Skolverket: 82% of children aged 1-5 in metropolitan municipalities were enrolled in preschool (autumn 2017).
PRESCHOOL_ENROLMENT = 0.82
# Assumed share of a workplace building's mapped person capacity that is jobs, by anchor kind.
JOB_SHARE = {
    "work": 1.0,
    "shopping": 0.25,
    "leisure": 0.2,
    "education": 0.15,
    "study": 0.25,
}
# Chalmers teaching blocks 08-10, 10-12, 13-15 and 15-17, ending a quarter before the hour.
CAMPUS_BLOCKS = (
    (8 * 3600, 9 * 3600 + 45 * 60),
    (10 * 3600, 11 * 3600 + 45 * 60),
    (13 * 3600 + 15 * 60, 15 * 3600),
    (15 * 3600 + 15 * 60, 17 * 3600),
)
# Assumed shares; RVU main journeys start and end at home, work or school and omit these outings.
LUNCH_OUTING = {"work": 0.35, "study": 0.6}
ROOM_CHANGE = 0.45
APPROACH = {"walk": 1.0, "cycle": 3.6, "car": 6.0, "transit": 4.0}
MINIMUM = {
    "ESUB": (1, 0),
    "ESMB": (1, 1),
    "SBUB": (2, 0),
    "SBMB": (2, 1),
    "OVRIGA": (1, 0),
}


def _integerize(values, count):
    values = np.asarray(values, dtype=float)
    if count <= 0 or not values.sum():
        return np.zeros(len(values), dtype=int)
    scaled = values * count / values.sum()
    result = np.floor(scaled).astype(int)
    result[np.argsort(-(scaled - result), kind="stable")[: count - result.sum()]] += 1
    return result


def _control(path, name):
    payload = json.loads((path / f"{name}.json").read_text())
    controls = defaultdict(dict)
    for row in payload.get("response", {}).get("data", []):
        zone, value = row["key"][:2]
        token = str(row["values"][0])
        controls[zone][value] = int(token) if token.isdigit() else None
    return controls


def _survey(path):
    with path.open("rb") as stream:
        workbook = load_workbook(stream, read_only=True, data_only=True)
        table = workbook["Tabell 1"]
        counts = {
            purpose: [int(table.cell(row, col).value) for col in (2, 4, 6, 8)]
            for purpose, row in (("work", 8), ("shopping", 10), ("leisure", 12))
        }
        totals = {
            purpose: [
                float(workbook["Tabell 5"].cell(row, col).value) for col in (12, 13)
            ]
            for purpose, row in (
                ("work", 8),
                ("shopping", 10),
                ("leisure", 12),
                ("all", 18),
            )
        }
        workbook.close()
    return {"table_1_annual_thousand": counts, "table_5_daily_rate_and_95ci": totals}


def _ages(control, count, stream):
    margin = np.array([control.get(code) or 0 for code in DESO_AGE_BANDS])
    known = min(count, int(margin.sum()))
    ages = []
    for band, n in zip(DESO_AGE_BANDS, _integerize(margin, known)):
        lower = int(band.split("-")[0] or 0)
        upper = lower + (16 if band == "80-" else 5)
        ages.extend((int(age), band) for age in stream.integers(lower, upper, int(n)))
    ages.extend((None, "unknown") for _ in range(count - known))
    return ages


def _members(zonal_ages, controls, scale, stream):
    adults = [(age, band) for age, band in zonal_ages if age is None or age >= 18]
    children = [(age, band) for age, band in zonal_ages if age is not None and age < 18]
    stream.shuffle(adults)
    stream.shuffle(children)
    observed = [(controls.get(tag) or 0) for tag in DESO_HOUSEHOLD_TYPES]
    house_count = min(
        len(zonal_ages), round((controls.get("TOTALT") or sum(observed)) * scale)
    )
    if not house_count and zonal_ages:
        house_count = max(1, min(len(adults), round(len(zonal_ages) / 2)))
    counts = _integerize(observed if sum(observed) else [1, 0, 0, 0, 0], house_count)
    groups = []
    unmatched = 0
    for tag in ("SBMB", "ESMB", "SBUB", "ESUB", "OVRIGA"):
        na, nc = MINIMUM[tag]
        for _ in range(counts[DESO_HOUSEHOLD_TYPES.index(tag)]):
            if len(adults) >= na and len(children) >= nc:
                group = [adults.pop() for _ in range(na)] + [
                    children.pop() for _ in range(nc)
                ]
                groups.append((tag, group))
            else:
                unmatched += 1
    if not groups and zonal_ages:
        return [("unclassified", list(zonal_ages))], unmatched
    child_groups = [group for tag, group in groups if tag in {"ESMB", "SBMB", "OVRIGA"}]
    if children and not child_groups:
        remaining_children = children[:]
        children.clear()
        groups.append(
            (
                "unclassified",
                [adults.pop()] + remaining_children if adults else remaining_children,
            )
        )
    else:
        while children:
            min(child_groups, key=len).append(children.pop())
    multi = [group for tag, group in groups if tag == "OVRIGA"]
    while adults:
        if not multi:
            group = [adults.pop()]
            groups.append(("unclassified", group))
            multi.append(group)
        if adults:
            min(multi, key=len).append(adults.pop())
    return groups, unmatched


class Destinations:
    """Spatial index for capacity-weighted choice of nearby activity anchors."""

    def __init__(self, anchors):
        self.anchors = anchors
        self.remaining = np.array([a.capacity for a in anchors], dtype=float)
        self.tree = cKDTree(np.asarray([a.xyz[:2] for a in anchors]).reshape((-1, 2)))
        self.by_id = {anchor.id: index for index, anchor in enumerate(anchors)}
        self.overflow = 0

    def reserve(self, anchor):
        index = self.by_id[anchor.id]
        if self.remaining[index] > 0:
            self.remaining[index] -= 1
        else:
            self.overflow += 1

    def choose(self, origin, mode, stream, *, allow_overflow=False, nearest=False):
        if not self.anchors:
            return None
        scale = {"walk": 650, "cycle": 1750, "car": 4200, "transit": 3600}[mode]
        limit = min(len(self.anchors), 60 if allow_overflow else 30)
        distances, keys = self.tree.query(origin[:2], k=limit)
        keys, distances = np.atleast_1d(keys), np.atleast_1d(distances)
        decay = np.exp(-distances / scale)
        decay[distances < 35] = 0
        if mode == "walk":
            decay[distances > 2200] = 0
        weights = np.sqrt(self.remaining[keys]) * decay
        if not weights.sum() and allow_overflow:
            weights = decay
        if not weights.sum():
            return None
        picked = int(
            keys[
                np.flatnonzero(weights)[0]
                if nearest
                else stream.choice(len(keys), p=weights / weights.sum())
            ]
        )
        return self.anchors[picked]

    def near(self, origin, radius, stream, exclude=""):
        """Pick a nearby anchor within a walking radius, favouring close and large ones."""
        if not self.anchors:
            return None
        keys = [
            key
            for key in self.tree.query_ball_point(origin[:2], r=radius)
            if self.anchors[key].id != exclude
        ]
        if not keys:
            return None
        keys = np.asarray(sorted(keys))
        distances = np.linalg.norm(
            np.asarray([self.anchors[k].xyz[:2] for k in keys]) - origin[:2], axis=1
        )
        weights = np.sqrt(
            np.maximum(1, [self.anchors[k].capacity for k in keys])
        ) * np.exp(-distances / 220)
        return self.anchors[int(stream.choice(keys, p=weights / weights.sum()))]

    def stop(self, coordinate):
        if self.anchors:
            distance, index = self.tree.query(coordinate[:2])
            if 8 < distance <= 500:
                return self.anchors[index]
        return None


def _role(age, stream):
    if age is None:
        return "unknown"
    if 1 <= age <= 5:
        return "preschool" if stream.random() < PRESCHOOL_ENROLMENT else "home"
    if 6 <= age < 18:
        return "education"
    if 18 <= age <= 28 and stream.random() < 0.32:
        return "study"
    if 18 <= age < 67 and stream.random() < 0.78:
        return "work"
    return "home"


def _walk_speed(age, stream):
    speed = float(np.clip(stream.normal(1.35, 0.14), 0.8, 1.8))
    return speed * 0.85 if age is not None and (age < 12 or age > 74) else speed


def _legs(
    origin, destination, mode, depart, speed, purpose, key, router, stops, timetable
):
    if depart >= 86400:
        return None
    if mode == "walk":
        route = router.between(origin.xyz, destination.xyz)
        return (
            (timed_route(route, depart, speed, purpose, key, signal_wait=12) or None)
            if route
            else None
        )
    if mode == "transit":
        if timetable is not None:
            return timetable.itinerary(
                origin.xyz, destination.xyz, depart, speed, purpose, key, router
            )
        board, alight = stops.stop(origin.xyz), stops.stop(destination.xyz)
        if board is None or alight is None or board.id == alight.id:
            return None
        access = router.between(origin.xyz, board.xyz)
        egress = router.between(alight.xyz, destination.xyz)
        if access is None or egress is None:
            return None
        stages = timed_route(access, depart, speed, purpose, key, signal_wait=12)
        if not stages or stages[-1].end + 240 >= 86400:
            return None
        wait = stationary(
            "wait",
            purpose,
            board.id,
            board.xyz,
            stages[-1].end,
            stages[-1].end + 240,
            key,
        )
        end = wait.end + float(np.linalg.norm(alight.xyz - board.xyz)) / 7.5
        if not wait.end < end < 86400:
            return None
        exit_walk = timed_route(egress, end, speed, purpose, key, signal_wait=12)
        if not exit_walk:
            return None
        ride = TimedPath(
            "transit",
            purpose,
            [[wait.end, *board.xyz], [end, *alight.xyz]],
            alight.id,
            key,
        )
        return [
            *stages,
            wait,
            ride,
            *exit_walk,
        ]
    duration = (
        float(np.linalg.norm(destination.xyz - origin.xyz))
        / {"car": 8.0, "cycle": 4.1}[mode]
    )
    if duration <= 0 or depart + duration > 86400:
        return None
    return [
        TimedPath(
            mode,
            purpose,
            [[depart, *origin.xyz], [depart + duration, *destination.xyz]],
            journey_id=key,
        )
    ]


def _campus_plan(stream):
    """Draw a contiguous run of teaching blocks for one campus day."""
    first = int(stream.choice(3, p=(0.55, 0.25, 0.2)))
    count = int(stream.choice(np.arange(1, 5), p=(0.2, 0.4, 0.25, 0.15)))
    return first, min(len(CAMPUS_BLOCKS) - 1, first + count - 1)


def _stay(
    key, purpose, place, arrive, leave, plan, speed, router, sites, stream, diagnostics
):
    """Time at a destination, split by walking moves between campus buildings and lunch outings."""
    paths, trips = [], []
    state = {"place": place, "purpose": purpose, "since": arrive}

    def move(target, depart, label, until):
        if target is None or depart <= state["since"] + 60:
            return False
        route = router.between(state["place"].xyz, target.xyz)
        legs = (
            timed_route(
                route, depart, speed, label, f"{key}-m{len(trips)}", signal_wait=12
            )
            if route
            else None
        )
        if not legs or legs[-1].end >= until:
            return False
        paths.append(
            stationary(
                "indoor",
                state["purpose"],
                state["place"].id,
                state["place"].xyz,
                state["since"],
                depart,
            )
        )
        paths.extend(legs)
        trips.append((f"{key}-m{len(trips)}", label, "walk"))
        state.update(place=target, purpose=label, since=legs[-1].end)
        return True

    def lunch(depart, back_by, after):
        restaurant = sites["lunch"].near(state["place"].xyz, 650, stream)
        home_base = state["place"]
        if not move(restaurant, depart, "lunch", back_by - 120):
            return
        diagnostics["lunch_outings"] += 1
        stay = float(stream.uniform(22, 42)) * 60
        target = after or home_base
        if not move(
            target, min(state["since"] + stay, back_by - 600), purpose, back_by
        ):
            move(home_base, state["since"] + 300, purpose, leave)

    if plan is not None:
        blocks = CAMPUS_BLOCKS[plan[0] : plan[1] + 1]
        for (_, end), (start, _) in pairwise(blocks):
            if start - end > 3600 and stream.random() < LUNCH_OUTING["study"]:
                after = (
                    sites["study"].near(state["place"].xyz, 500, stream)
                    if stream.random() < 0.5
                    else None
                )
                lunch(end + float(stream.uniform(0, 300)), start, after)
            elif start - end <= 3600 and stream.random() < ROOM_CHANGE:
                if move(
                    sites["study"].near(
                        state["place"].xyz, 450, stream, state["place"].id
                    ),
                    end + float(stream.uniform(0, 150)),
                    purpose,
                    start + 600,
                ):
                    diagnostics["campus_room_changes"] += 1
    elif (
        purpose in LUNCH_OUTING
        and arrive < 11.4 * 3600
        and leave > 13.2 * 3600
        and stream.random() < LUNCH_OUTING[purpose]
    ):
        depart = float(np.clip(stream.normal(11.9, 0.3), 11.3, 12.7)) * 3600
        lunch(depart, depart + 75 * 60, None)
    leave = max(leave, state["since"] + 300)
    paths.append(
        stationary(
            "indoor",
            state["purpose"],
            state["place"].id,
            state["place"].xyz,
            state["since"],
            leave,
        )
    )
    return paths, state["place"], trips, leave


def _schedule(
    person_id,
    age,
    role,
    home,
    destination,
    stops,
    router,
    survey,
    seed,
    diagnostics,
    escort=None,
    baseline=False,
    timetable=None,
    attend=None,
):
    stream = rng(seed, person_id, "day")
    speed = _walk_speed(age, stream)
    activities = []
    if escort is not None:
        activities.append(("escort", 7.7, 0.12, None))
        activities.append(
            ("escort", float(np.clip(stream.normal(16.3, 0.45), 15.0, 17.6)), 0.1, None)
        )
    if attend is None:
        attend = stream.random() < PRIMARY_PARTICIPATION
    if role in {"education", "study", "work"} and attend:
        plan = _campus_plan(stream) if role == "study" else None
        hour = float(np.clip(stream.normal(7.9, 0.55), 6.2, 9.7))
        activities.append((role, hour, 6.2 if role == "education" else 7.0, plan))
    if age is None or age >= 6:
        for purpose, hour, length in (("shopping", 16.8, 0.45), ("leisure", 18.4, 1.0)):
            rate = survey["table_5_daily_rate_and_95ci"][purpose][0]
            if stream.random() < rate:
                activities.append(
                    (
                        purpose,
                        float(np.clip(stream.normal(hour, 1.75), 10, 21)),
                        length,
                        None,
                    )
                )
    activities.sort(key=lambda item: item[1])
    paths = []
    trips = []
    escorts = []
    regular = ""
    at_home_from = 0.0
    for purpose, hour, dwell, plan in activities:
        mode_counts = np.asarray(
            survey["table_1_annual_thousand"].get(
                purpose, survey["table_1_annual_thousand"]["work"]
            ),
            dtype=float,
        )
        mode = str(stream.choice(MODES, p=mode_counts / mode_counts.sum()))
        if purpose in {"education", "escort"} and stream.random() < 0.65:
            mode = "walk"
        if purpose == "escort":
            place = escort
        else:
            sites = destination[purpose]
            place = sites.choose(
                home.xyz,
                mode,
                stream,
                allow_overflow=purpose == "work",
                nearest=baseline,
            )
            if place is None and mode in {"walk", "cycle"}:
                diagnostics["destination_mode_fallback"] += 1
                mode = "transit"
                place = sites.choose(
                    home.xyz,
                    mode,
                    stream,
                    allow_overflow=purpose == "work",
                    nearest=baseline,
                )
        if place is None:
            diagnostics["unavailable_destinations"] += 1
            continue
        if plan is not None:
            # Leave home in time for the first teaching block at the chosen campus building.
            approach = float(np.linalg.norm(place.xyz - home.xyz)) / (
                APPROACH[mode] * speed
            )
            hour = (
                CAMPUS_BLOCKS[plan[0]][0]
                - approach
                - 600
                - (480 if mode == "transit" else 0)
            ) / 3600
        depart = max(at_home_from + 60, hour * 3600)
        key = f"{person_id}-{len(trips)}"
        scheduled_transit = timetable is not None and mode == "transit"
        if scheduled_transit:
            diagnostics["scheduled_transit_outings_requested"] += 1
        outbound = _legs(
            home, place, mode, depart, speed, purpose, key, router, stops, timetable
        )
        if outbound is None and mode == "transit":
            diagnostics[
                "scheduled_transit_outbound_to_car"
                if scheduled_transit
                else "transit_proxy_to_car"
            ] += 1
            mode = "car"
            outbound = _legs(
                home, place, mode, depart, speed, purpose, key, router, stops, timetable
            )
        if outbound is None:
            diagnostics["unreachable_journeys"] += 1
            continue
        leave = outbound[-1].end + dwell * 3600
        if plan is not None:
            leave = max(
                outbound[-1].end + 900,
                CAMPUS_BLOCKS[plan[1]][1] + float(stream.uniform(0, 1500)),
            )
        if leave >= 86400:
            diagnostics["infeasible_journeys"] += 1
            if scheduled_transit:
                diagnostics["scheduled_transit_outings_dropped_on_return"] += 1
            continue
        stay_diagnostics = Counter()
        stay, last, moves, leave = (
            _stay(
                key,
                purpose,
                place,
                outbound[-1].end,
                leave,
                plan,
                speed,
                router,
                destination,
                stream,
                stay_diagnostics,
            )
            if purpose in LUNCH_OUTING
            else (
                [
                    stationary(
                        "indoor", purpose, place.id, place.xyz, outbound[-1].end, leave
                    )
                ],
                place,
                [],
                leave,
            )
        )
        inbound = _legs(
            last,
            home,
            mode,
            leave,
            speed,
            "home",
            f"{key}-return",
            router,
            stops,
            timetable,
        )
        if inbound is None or inbound[-1].end >= 86400:
            diagnostics["infeasible_journeys"] += 1
            if scheduled_transit:
                diagnostics["scheduled_transit_outings_dropped_on_return"] += 1
            continue
        diagnostics.update(stay_diagnostics)
        if scheduled_transit and mode == "transit":
            diagnostics["scheduled_transit_outings_completed"] += 1
            diagnostics["scheduled_transit_journeys_with_transfer"] += sum(
                sum(stage.state == "transit" for stage in journey) > 1
                for journey in (outbound, inbound)
            )
        if depart > hour * 3600:
            diagnostics["schedule_repairs"] += 1
        paths.append(
            stationary("indoor", "home", home.id, home.xyz, at_home_from, depart)
        )
        paths.extend(outbound)
        paths.extend(stay)
        paths.extend(inbound)
        at_home_from = inbound[-1].end
        if purpose == role:
            regular = place.id
        if purpose == "work":
            destination["work"].reserve(place)
        if purpose == "escort":
            escorts.append((outbound, inbound, mode))
        trips.append((key, purpose, mode))
        trips.extend(moves)
        trips.append((f"{key}-return", "home", mode))
    paths.append(stationary("indoor", "home", home.id, home.xyz, at_home_from, 86400))
    return tuple(paths), regular, trips, escorts


def _escorted_day(person_id, home, school, escorts):
    """Share the escorting adult's drop-off and pick-up journeys, so the two walk together."""
    if len(escorts) < 2:
        return None
    (drop, _, mode), (_, pickup, back) = escorts[0], escorts[-1]
    if pickup[0].start <= drop[-1].end:
        return None

    def copy(stages, purpose, key):
        return [
            TimedPath(
                stage.state, purpose, stage.samples, stage.place_id, key, stage.edge_ids
            )
            for stage in stages
        ]

    key = f"{person_id}-0"
    outbound = copy(drop, "education", key)
    inbound = copy(pickup, "home", f"{key}-return")
    paths = (
        stationary("indoor", "home", home.id, home.xyz, 0, outbound[0].start),
        *outbound,
        stationary(
            "indoor",
            "education",
            school.id,
            school.xyz,
            outbound[-1].end,
            inbound[0].start,
        ),
        *inbound,
        stationary("indoor", "home", home.id, home.xyz, inbound[-1].end, 86400),
    )
    return paths, school.id, [(key, "education", mode), (f"{key}-return", "home", back)]


def _linked_day(
    person_id,
    age,
    home,
    sites,
    stops,
    router,
    timetable,
    role,
    stream,
    journeys,
    diagnostics,
    baseline,
):
    speed = _walk_speed(age, stream)
    paths, trips = [], []
    workplaces = {}
    origin = home
    activity = "home"
    arrived = 0.0
    regular = ""
    for index, journey in enumerate(journeys):
        depart = max(
            arrived + 60,
            journey.departure_bin * DEPARTURE_BIN_SECONDS
            + float(stream.uniform(0, DEPARTURE_BIN_SECONDS)),
        )
        purpose, mode = journey.purpose, journey.mode
        anchor_kind = "education" if purpose == "escort" else purpose
        destination = (
            home
            if purpose == "home"
            else sites[anchor_kind].choose(
                origin.xyz,
                mode,
                stream,
                allow_overflow=purpose == "work",
                nearest=baseline,
            )
        )
        key = f"{person_id}-{index}"
        legs = (
            _legs(
                origin,
                destination,
                mode,
                depart,
                speed,
                purpose,
                key,
                router,
                stops,
                timetable,
            )
            if destination is not None
            else None
        )
        if not legs or legs[-1].end >= 86400:
            diagnostics["diary_profiles_unroutable"] += 1
            return (stationary("indoor", "home", home.id, home.xyz, 0, 86400),), "", []
        paths.append(
            stationary("indoor", activity, origin.id, origin.xyz, arrived, depart)
        )
        paths.extend(legs)
        trips.append((key, purpose, mode))
        if purpose == role and not regular:
            regular = destination.id
        if purpose == "work":
            workplaces[destination.id] = destination
        origin, activity, arrived = destination, purpose, legs[-1].end
    paths.append(stationary("indoor", "home", home.id, home.xyz, arrived, 86400))
    for place in workplaces.values():
        sites["work"].reserve(place)
    diagnostics["diary_profiles_routed" if journeys else "diary_profiles_at_home"] += 1
    return tuple(paths), regular, trips


@lru_cache(maxsize=2)
def _inputs(path, regional, year, _revision):
    regions = gpd.read_file(path / "deso.gpkg").to_crs(3006).sort_values("desokod")
    grid = gpd.read_file(path.parent / f"grid-{year}.gpkg", bbox=regional)
    network, homes, activities, rails = read_osm(sorted(path.glob("*.osm")))
    population = _control(path, "population")
    household = _control(path, "households")
    survey = _survey(path.parent / f"rvu-{year}.xlsx")
    return (
        regions,
        grid,
        network,
        homes,
        activities,
        rails,
        population,
        household,
        survey,
    )


@lru_cache(maxsize=2)
def _regional_day(
    path,
    regional,
    year,
    seed,
    closed_edge,
    baseline,
    transit_service_date,
    survey_profile,
    revision,
):
    (
        regions,
        grid,
        network,
        homes,
        activities,
        rails,
        population,
        controls,
        survey,
    ) = _inputs(path, regional, year, revision)
    extent = box(*regional)
    router = Router(network, closed_edge=closed_edge, distance_only=baseline)
    sites = {
        kind: Destinations([a for a in activities if a.kind == kind])
        for kind in ("education", "study", "shopping", "leisure")
    }
    sites["work"] = Destinations(
        [
            replace(a, capacity=max(1, round(a.capacity * JOB_SHARE[a.kind])))
            for a in activities
            if a.kind in JOB_SHARE
        ]
    )
    sites["lunch"] = Destinations(
        [a for a in activities if a.kind in {"leisure", "shopping"}]
    )
    stops = Destinations([a for a in activities if a.kind == "stop"])
    extra_sources = []
    timetable = None
    if transit_service_date:
        from .transit import Shapes, load_timetable

        timetable, transit_source = load_timetable(
            path.parent, transit_service_date, regional
        )
        timetable.shapes = Shapes(rails, Router(network, mode="drive"))
        extra_sources.append(transit_source)
    diaries = None
    if survey_profile == "local_diaries":
        diaries, diary_source = load_diaries(path.parent)
        extra_sources.append(diary_source)
    grid_counts = {
        (
            round(row.geometry.bounds[0] / 1000),
            round(row.geometry.bounds[1] / 1000),
        ): int(row.beftotalt)
        for row in grid.itertuples()
    }
    points = [Point(*anchor.xyz[:2]) for anchor in homes]
    tree = STRtree(points)
    remaining = np.asarray([anchor.capacity for anchor in homes], dtype=int)
    cell_capacity = Counter()
    cell_occupancy = Counter()
    diagnostics = Counter()
    if diaries is not None:
        diagnostics.update(
            {
                "diary_profiles_routed": 0,
                "diary_profiles_at_home": 0,
                "diary_profiles_unroutable": 0,
                "diary_age_uncovered_used_published_rules": 0,
            }
        )
    if timetable is not None:
        diagnostics.update(
            {
                "scheduled_transit_stop_areas": len(timetable.stop_ids),
                "scheduled_transit_service_trips": timetable.service_trips,
                "scheduled_transit_outings_requested": 0,
                "scheduled_transit_outings_completed": 0,
                "scheduled_transit_outbound_to_car": 0,
                "scheduled_transit_outings_dropped_on_return": 0,
                "scheduled_transit_journeys_with_transfer": 0,
            }
        )
    for anchor in homes:
        cell_capacity[_cell(anchor.xyz)] += anchor.capacity
    all_people, zonal = [], {}
    household_ids = {}
    for zone in regions.itertuples():
        code = zone.desokod
        geom = zone.geometry.intersection(extent)
        if geom.is_empty or geom.area <= 0:
            continue
        total = population[code].get("totalt")
        if total is None:
            diagnostics["missing_population_zones"] += 1
            continue
        fraction = geom.area / zone.geometry.area
        count = round(total * fraction)
        if count == 0:
            diagnostics["zero_population_zones"] += 1
            continue
        stream = rng(seed, code, "households")
        ages = _ages(population[code], count, stream)
        groups, unmatched = _members(ages, controls[code], fraction, stream)
        diagnostics["unmatched_household_controls"] += unmatched
        indexes = sorted(tree.query(geom, predicate="covers").tolist())
        centroid = geom.representative_point()
        unknown = Anchor(
            f"deso-{code}-unallocated",
            np.array([centroid.x, centroid.y, 0]),
            "home",
            count,
        )
        placed_types = Counter()
        for h, (category, members) in enumerate(groups):
            options = [i for i in indexes if remaining[i] >= len(members)]
            if options:
                weights = []
                for i in options:
                    cell = _cell(homes[i].xyz)
                    # SCB grid is a soft constraint where it overlaps DeSO capacity.
                    density = (
                        ((grid_counts.get(cell, 0) + 25) / (cell_capacity[cell] + 25))
                        if not baseline
                        else 1
                    )
                    pressure = max(
                        0.12,
                        1 - cell_occupancy[cell] / max(1, grid_counts.get(cell, 0)),
                    )
                    weights.append(
                        remaining[i] * density * (1 if baseline else pressure)
                    )
                choice = int(
                    stream.choice(options, p=np.asarray(weights) / sum(weights))
                )
                home = homes[choice]
                remaining[choice] -= len(members)
            else:
                home = unknown
                diagnostics["persons_without_building_capacity"] += len(members)
            cell_occupancy[_cell(home.xyz)] += len(members)
            household_id = f"{code}-h{h:04}"
            placed_types[category] += 1
            school = None
            if any(age is not None and age <= 9 for age, _ in members):
                school = sites["education"].choose(
                    home.xyz, "walk", rng(seed, household_id, "escort")
                )
            ids = [f"{household_id}-p{i}" for i in range(len(members))]
            roles = [
                _role(age, rng(seed, person_id, "role"))
                for person_id, (age, _) in zip(ids, members)
            ]
            attends = [
                rng(seed, person_id, "attend").random() < PRIMARY_PARTICIPATION
                for person_id in ids
            ]
            diary_choices = {}
            if diaries is not None:
                for i, ((age, _), person_id) in enumerate(zip(members, ids)):
                    diary_stream = rng(seed, person_id, "diary")
                    diary = diaries.choose(age, diary_stream)
                    if diary is None:
                        diagnostics["diary_age_uncovered_used_published_rules"] += 1
                    else:
                        diary_choices[i] = (diary_stream, diary)
            escorted = {
                i
                for i, ((age, _), role) in enumerate(zip(members, roles))
                if school is not None
                and attends[i]
                and i not in diary_choices
                and (role == "preschool" or (role == "education" and age <= 9))
            }
            escort_index = next(
                (
                    i
                    for i, (age, _) in enumerate(members)
                    if escorted
                    and i not in diary_choices
                    and age is not None
                    and age >= 18
                ),
                None,
            )
            days = {}
            escorts = []
            order = sorted(range(len(members)), key=lambda i: i != escort_index)
            for i in order:
                age, band = members[i]
                person_id, role = ids[i], roles[i]
                if i in diary_choices:
                    diary_stream, diary = diary_choices[i]
                    days[i] = _linked_day(
                        person_id,
                        age,
                        home,
                        sites,
                        stops,
                        router,
                        timetable,
                        role,
                        diary_stream,
                        diary,
                        diagnostics,
                        baseline,
                    )
                    continue
                if i in escorted:
                    day = _escorted_day(person_id, home, school, escorts)
                    if day is not None:
                        diagnostics["escorted_children"] += 1
                        days[i] = day
                        continue
                    diagnostics["escort_unavailable"] += 1
                if role == "preschool":
                    days[i] = (
                        (stationary("indoor", "home", home.id, home.xyz, 0, 86400),),
                        "",
                        [],
                    )
                else:
                    paths, regular, trips, escorts_made = _schedule(
                        person_id,
                        age,
                        role,
                        home,
                        sites,
                        stops,
                        router,
                        survey,
                        seed,
                        diagnostics,
                        school if i == escort_index else None,
                        baseline,
                        timetable,
                        attends[i],
                    )
                    if i == escort_index:
                        escorts = escorts_made
                    days[i] = (paths, regular, trips)
            member_ids = ids
            for i, (age, band) in enumerate(members):
                paths, regular, trips = days[i]
                all_people.append(
                    Person(
                        id=ids[i],
                        household_id=household_id,
                        age_band=band,
                        role=roles[i],
                        home_id=home.id,
                        regular_destination_id=regular,
                        appearance_key=int(
                            rng(seed, ids[i], "appearance").integers(0, 2**32)
                        ),
                        paths=paths,
                        attributes={
                            "deso": code,
                            "synthetic_age": age,
                            "household_type": category,
                            "building_type": home.building_type or "unallocated",
                            "mapped_home_access": home.mapped_access,
                            "journeys": [list(trip) for trip in trips],
                        },
                    )
                )
            household_ids[household_id] = member_ids
        zonal[code] = {
            "source_persons": total,
            "regional_fraction": fraction,
            "target_persons": count,
            "actual_persons": len(ages),
            "source_households": controls[code].get("TOTALT"),
            "actual_households": len(groups),
            "source_age": dict(population[code]),
            "source_household_types": dict(controls[code]),
            "actual_age": dict(Counter(band for _, band in ages)),
            "actual_household_types": dict(placed_types),
        }
    diagnostics["work_capacity_overflow"] = sites["work"].overflow
    diagnostics["regional_people"] = len(all_people)
    diagnostics["regional_households"] = len(household_ids)
    diagnostics["walk_network_edges"] = len(router.edges)
    diagnostics["persons_at_home_all_day"] = sum(
        len(person.paths) == 1 for person in all_people
    )
    diagnostics["mapped_transit_stop_anchors"] = len(stops.anchors)
    diagnostics["grid_mae_people"] = (
        float(
            np.mean(
                [
                    abs(grid_counts[cell] - cell_occupancy[cell])
                    for cell in grid_counts
                    if extent.covers(
                        box(
                            cell[0] * 1000,
                            cell[1] * 1000,
                            (cell[0] + 1) * 1000,
                            (cell[1] + 1) * 1000,
                        )
                    )
                ]
            )
        )
        if grid_counts and extent.area >= 4e6
        else None
    )
    return (
        all_people,
        household_ids,
        zonal,
        diagnostics,
        survey,
        homes,
        activities,
        extra_sources,
        timetable,
    )


def _cell(xyz):
    return int(xyz[0] // 1000), int(xyz[1] // 1000)


def _in_view(person, view):
    for path in person.paths:
        if path.state in {"indoor", "outdoor", "wait"}:
            if view.covers(Point(path.samples[0, 1:3])):
                return True
        else:
            coords = path.samples[:, 1:3]
            if view.intersects(LineString(coords)):
                return True
    return False


def build_population(bounds, args):
    """Synthesize a fixed regional day and deliver the requested spatial view."""
    path, record = select_region(bounds, args.reference_year, args.region)
    sources = archives(path, args.reference_year, record)
    revision = tuple(source["sha256"] for source in sources)
    if args.survey_profile == "local_diaries":
        diary = path.parent / DIARY_ARCHIVE
        revision += (fingerprint(diary) if diary.exists() else "",)
    if args.transit_service_date:
        from .transit import ARCHIVE, ARCHIVE_SHA256

        schedule = path.parent / ARCHIVE
        revision += (fingerprint(schedule) if schedule.exists() else ARCHIVE_SHA256,)
    regional = tuple(record["bounds"])
    data = _regional_day(
        Path(path),
        regional,
        args.reference_year,
        args.seed,
        args.closed_walk_edge,
        args.baseline,
        args.transit_service_date,
        args.survey_profile,
        revision,
    )
    (
        people,
        households,
        zonal,
        diagnostics,
        survey,
        homes,
        activities,
        extra_sources,
        timetable,
    ) = data
    view = box(*bounds.tuple)
    selected = [person for person in people if _in_view(person, view)]
    residents = sum(
        person.home_location is not None
        and view.covers(Point(person.home_location[:2]))
        for person in selected
    )
    relevant_households = {person.household_id for person in selected}
    variant = {
        "daylight": {
            "date": args.transit_service_date or "2019-10-15",
            "basis": "the scheduled transit service date"
            if args.transit_service_date
            else "an illustrative mid-October weekday; the scenario has no calendar date",
        }
    }
    buildings = []
    seen = set()
    for anchor in [*homes, *activities]:
        if (
            anchor.footprint is not None
            and anchor.id not in seen
            and view.intersects(anchor.footprint)
        ):
            seen.add(anchor.id)
            buildings.append(
                {
                    "id": anchor.id,
                    "type": anchor.building_type,
                    "kind": anchor.kind,
                    "height": anchor.height,
                    "height_source": anchor.height_source,
                    "ring": [list(xy) for xy in anchor.footprint.exterior.coords],
                }
            )
    if timetable is not None:
        variant["vehicles"] = timetable.vehicles(bounds.tuple)
    if args.transit_service_date:
        variant["transit_service_date"] = args.transit_service_date
    if args.survey_profile == "local_diaries":
        variant["survey_profile"] = args.survey_profile
    return simulation(
        bounds,
        args,
        [*sources, *deepcopy(extra_sources)],
        {
            "name": "Regional People diagnostics",
            **diagnostics,
            "view_people": len(selected),
            "view_residents": residents,
            "coverage": "covered"
            if zonal
            else "no populated DeSO controls in the regional archive",
        },
        deepcopy(selected),
        region=record["name"],
        regional_bounds=list(regional),
        zonal_controls=deepcopy(zonal),
        survey=deepcopy(survey),
        households={key: list(households[key]) for key in sorted(relevant_households)},
        buildings=buildings,
        showcase_start=7 * 3600 + 55 * 60,
        **variant,
    )
