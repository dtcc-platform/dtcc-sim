"""Evaluate independently generated 2019 resident days against GAPSIM."""

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import file_digest
from itertools import repeat
from multiprocessing import get_context
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
from shapely import from_wkt
from shapely.geometry import Point

from dtcc_sim import datasets

SITES = {
    "111": (
        "johanneberg",
        "e1925514150aa4d6a1082d4740029e6ed39760e5eaa73f07e8841eaebd2b8e63",
    ),
    "523": (
        "askim",
        "8e5c37ddb877587112cf253cde0125350d46cc0bf1670151b0c48e29bdde18d4",
    ),
    "606": (
        "hammarkullen",
        "45b7b64862b4439cd9a4c53f2be298db4d8553666381c18ad9b2810b01cffad4",
    ),
}
SEEDS = (17, 42, 83)
CLOCK = re.compile(
    r"Start Time: (\d\d:\d\d:\d\d), End Time: (\d\d:\d\d:\d\d), Purpose: ([^,]+), Mode: ([^\n]+)"
)
MODE = {
    "Walking": "walk",
    "Bicycle/E-bike": "cycle",
    "Car": "car",
    "Train/Tram": "transit",
    "Bus": "transit",
    "Boat": "transit",
}
PURPOSE = {
    "Home": "home",
    "Work": "work",
    "Education": "education",
    "Leisure": "leisure",
    "Grocery": "shopping",
    "Shopping": "shopping",
    "Pickup/Dropoff child": "escort",
}
HOME_TYPE = {
    "Flerfamiljshus": "multi",
    "Småhus": "single",
    "apartments": "multi",
    "dormitory": "multi",
    "house": "single",
    "detached": "single",
    "terrace": "single",
    "semidetached_house": "single",
}


def _minute(value):
    h, m, s = map(int, value.split(":"))
    return 60 * h + m + s / 60


def _age(value):
    if value is None:
        return "unknown"
    return str(next(limit for limit in (6, 18, 30, 45, 65, 85, 120) if value < limit))


def _cell(point):
    return f"{int(point.x // 1000)}:{int(point.y // 1000)}"


def _summary():
    return {
        "people": 0,
        "diary_eligible": 0,
        "diary_observed": 0,
        "diary_missing": 0,
        "zero_days": 0,
        "trip_total": 0,
        "age": Counter(),
        "age_household_size": Counter(),
        "grid": Counter(),
        "home_types": Counter(),
        "occupancy": Counter(),
        "households": set(),
        "household_sizes": Counter(),
        "chains": Counter(),
        "modes": Counter(),
        "purposes": Counter(),
        "departure": [],
        "sample_walk_min": [],
        "routed_walk_min": [],
        "routed_walk_metres": [],
        "walk_route_missing": 0,
        "zone_flows": Counter(),
        "access_walk_stages": 0,
    }


def _add_resident(result, age, hh, size, home_id, building_type, point):
    result["people"] += 1
    result["age"][_age(age)] += 1
    result["age_household_size"][f"{_age(age)}:{size}"] += 1
    result["households"].add(hh)
    result["household_sizes"][hh] = size
    if point is not None:
        result["grid"][_cell(point)] += 1
    if home_id:
        result["occupancy"][home_id] += 1
    result["home_types"][HOME_TYPE.get(building_type, "unknown")] += 1


def _add_trip(result, mode, purpose, departure):
    result["trip_total"] += 1
    result["modes"][mode] += 1
    result["purposes"][purpose] += 1
    result["departure"].append(departure)


def _od_group(origin, destination, area):
    if origin is None or destination is None:
        return "unknown"
    return f"{'within' if area.covers(origin) else 'outside'}→{'within' if area.covers(destination) else 'outside'}"


def observed(path, area, expected):
    with path.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    if digest != expected:
        raise ValueError(f"Unexpected GAPSIM file SHA-256 for {path}")
    result = _summary()
    result["sha256"] = digest
    result["homes_outside_official_boundary"] = 0
    result["under_18_with_diary"] = 0
    result["unmatched_main_trip_rows"] = 0
    lookup = {}
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        for row in db.execute("""
            SELECT p.uuid, p.age, p.uuid_household, p.origin, p.activity_sequence,
                   hh.count_members, h.uuid_building, b.type_building
            FROM person p LEFT JOIN household hh ON hh.uuid=p.uuid_household
            LEFT JOIN house h ON hh.uuid_house=h.uuid
            LEFT JOIN building b ON b.uuid=h.uuid_building
        """):
            home = (
                from_wkt(row["origin"], on_invalid="ignore") if row["origin"] else None
            )
            if home is not None and not area.covers(home):
                result["homes_outside_official_boundary"] += 1
                continue
            age = row["age"]
            _add_resident(
                result,
                age,
                row["uuid_household"],
                row["count_members"],
                row["uuid_building"],
                row["type_building"],
                home,
            )
            diary = CLOCK.findall(row["activity_sequence"] or "")
            if age is not None and age < 18 and diary:
                result["under_18_with_diary"] += 1
            if age is None or not 18 <= age <= 84:
                continue
            result["diary_eligible"] += 1
            if not diary:
                result["diary_missing"] += 1
                continue
            result["diary_observed"] += 1
            travels = [
                index for index, entry in enumerate(diary) if entry[2] == "Transit"
            ]
            result["zero_days"] += not travels
            sequence = [
                PURPOSE.get(entry[2], "other")
                for entry in diary
                if entry[2] != "Transit"
            ]
            result["chains"]["→".join(sequence)] += 1
            for index in travels:
                start, end, _, source_mode = diary[index]
                mode = MODE.get(source_mode, "other")
                purpose = (
                    PURPOSE.get(diary[index + 1][2], "other")
                    if index + 1 < len(diary)
                    else "other"
                )
                _add_trip(result, mode, purpose, _minute(start))
                token = f"Start Time: {start}, End Time: {end}, Purpose: Transit, Mode: {source_mode}"
                lookup[row["uuid"], token] = mode
                if mode == "walk":
                    result["sample_walk_min"].append(
                        (_minute(end) - _minute(start)) % 1440
                    )
        matched = set()
        for row in db.execute(
            "SELECT uuid_person, transit_activity, mode, origin, destination, route, calculated_duration FROM od_matrix"
        ):
            key = (row["uuid_person"], row["transit_activity"])
            mode = lookup.get(key)
            if mode is None or key in matched or MODE.get(row["mode"]) != mode:
                continue
            matched.add(key)
            origin = (
                from_wkt(row["origin"], on_invalid="ignore") if row["origin"] else None
            )
            dest = (
                from_wkt(row["destination"], on_invalid="ignore")
                if row["destination"]
                else None
            )
            result["zone_flows"][_od_group(origin, dest, area)] += 1
            if mode != "walk":
                continue
            route = (
                from_wkt(row["route"], on_invalid="ignore") if row["route"] else None
            )
            if route is None or route.length == 0 or row["calculated_duration"] is None:
                result["walk_route_missing"] += 1
                continue
            result["routed_walk_metres"].append(route.length)
            result["routed_walk_min"].append(row["calculated_duration"])
        result["unmatched_main_trip_rows"] = len(lookup) - len(matched)
    return result


def generated(model, area):
    result = _summary()
    for person in model.people:
        location = person.home_location
        if location is None:
            continue
        home = Point(*location[:2])
        if not area.covers(home):
            continue
        age = person.attributes["synthetic_age"]
        _add_resident(
            result,
            age,
            person.household_id,
            len(model.household_members(person.household_id)),
            person.home_id,
            person.attributes["building_type"],
            home,
        )
        if age is None or not 18 <= age <= 84:
            continue
        result["diary_eligible"] += 1
        result["diary_observed"] += 1
        journeys = person.attributes["journeys"]
        result["zero_days"] += not journeys
        result["chains"][
            "→".join(p.purpose for p in person.paths if p.state == "indoor")
        ] += 1
        stages = defaultdict(list)
        for path in person.paths:
            if path.journey_id:
                stages[path.journey_id].append(path)
        for key, purpose, mode in journeys:
            path = stages[key]
            _add_trip(result, mode, purpose, path[0].start / 60)
            result["zone_flows"][
                _od_group(
                    Point(*path[0].samples[0, 1:3]),
                    Point(*path[-1].samples[-1, 1:3]),
                    area,
                )
            ] += 1
            if mode == "walk":
                minutes = (path[-1].end - path[0].start) / 60
                result["sample_walk_min"].append(minutes)
                result["routed_walk_min"].append(minutes)
                result["routed_walk_metres"].append(
                    sum(stage.length for stage in path if stage.state == "walk")
                )
            elif mode == "transit":
                result["access_walk_stages"] += sum(
                    stage.state == "walk" for stage in path
                )
    return result


def _shares(counts):
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()} if total else {}


def _distance(a, b):
    return float(wasserstein_distance(a, b)) if len(a) and len(b) else None


def compare(sim, gap):
    result = {}
    for key in (
        "people",
        "diary_eligible",
        "diary_observed",
        "diary_missing",
        "zero_days",
        "trip_total",
        "walk_route_missing",
        "access_walk_stages",
    ):
        result[key] = {
            "dtcc": sim[key],
            "gapsim": gap[key],
            "difference": sim[key] - gap[key],
        }
    result["access_walk_stages"]["gapsim"] = None
    result["access_walk_stages"]["difference"] = None
    result["households"] = {
        "dtcc": len(sim["households"]),
        "gapsim": len(gap["households"]),
        "difference": len(sim["households"]) - len(gap["households"]),
    }
    for key, numerator, factor in (
        ("trips_per_diary", "trip_total", 1),
        ("zero_trip_percentage", "zero_days", 100),
    ):
        a = factor * sim[numerator] / max(1, sim["diary_observed"])
        b = factor * gap[numerator] / max(1, gap["diary_observed"])
        result[key] = {"dtcc": a, "gapsim": b, "difference": a - b}
    for key in (
        "age",
        "age_household_size",
        "grid",
        "home_types",
        "chains",
        "modes",
        "purposes",
        "zone_flows",
    ):
        a, b = _shares(sim[key]), _shares(gap[key])
        keys = sorted(a.keys() | b.keys())
        delta = {k: 100 * (a.get(k, 0) - b.get(k, 0)) for k in keys}
        result[key] = {
            "dtcc_counts": dict(sim[key]),
            "gapsim_counts": dict(gap[key]),
            "total_variation": sum(map(abs, delta.values())) / 200,
            "percentage_point_difference": delta,
        }
    cells = sim["grid"].keys() | gap["grid"].keys()
    result["grid_mae_people"] = (
        sum(abs(sim["grid"][k] - gap["grid"][k]) for k in cells) / len(cells)
        if cells
        else None
    )
    for key in (
        "departure",
        "sample_walk_min",
        "routed_walk_min",
        "routed_walk_metres",
        "occupancy",
    ):
        a, b = (
            (list(sim[key].values()), list(gap[key].values()))
            if key == "occupancy"
            else (sim[key], gap[key])
        )
        result[key] = {
            "dtcc_n": len(a),
            "gapsim_n": len(b),
            "dtcc_mean": float(np.mean(a)) if len(a) else None,
            "gapsim_mean": float(np.mean(b)) if len(b) else None,
            "wasserstein": _distance(a, b),
        }
    result["gapsim_under_18_with_diary"] = gap["under_18_with_diary"]
    result["gapsim_homes_outside_official_boundary"] = gap[
        "homes_outside_official_boundary"
    ]
    result["gapsim_unmatched_main_trip_rows"] = gap["unmatched_main_trip_rows"]
    return result


def _run_realization(task, name, area, gap, transit_service_date):
    seed, baseline = task
    model = datasets.people(
        bounds=area.bounds,
        reference_year=2019,
        seed=seed,
        region=name,
        baseline=baseline,
        transit_service_date=transit_service_date,
    )
    sim = generated(model, area)
    score = compare(sim, gap)
    edges = np.arange(0, 1441, 60)
    score.update(
        seed=seed,
        baseline=baseline,
        primary_fit=model.attributes["zonal_controls"],
        diagnostics=model.attributes["diagnostics"],
        survey=model.attributes["survey"],
        departure_hourly_counts=np.histogram(sim["departure"], bins=edges)[0].tolist(),
        source_hashes=[
            source["sha256"] for source in model.dataset_context.provenance.sources
        ],
    )
    print(
        name,
        seed,
        "baseline" if baseline else "scenario",
        score["people"],
        score["trips_per_diary"],
        flush=True,
    )
    return score


def run(site, evidence, output, workers=1, transit_service_date=None):
    name, expected = SITES[site]
    output.mkdir(parents=True, exist_ok=True)
    boundaries = evidence / "gothenburg-primary-areas.geojson"
    with boundaries.open("rb") as stream:
        boundary_sha256 = file_digest(stream, "sha256").hexdigest()
    areas = gpd.read_file(boundaries).to_crs(3006)
    area = areas.loc[areas["PRIMÄROMRÅ"] == site].geometry.iloc[0]
    gap = observed(evidence / f"{site}.db", area, expected)
    tasks = [*((seed, False) for seed in SEEDS), (42, True)]
    if workers > 1:
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
            results = list(
                pool.map(
                    _run_realization,
                    tasks,
                    repeat(name),
                    repeat(area),
                    repeat(gap),
                    repeat(transit_service_date),
                )
            )
    else:
        results = [
            _run_realization(task, name, area, gap, transit_service_date)
            for task in tasks
        ]
    edges = np.arange(0, 1441, 60)
    departures = [
        (
            "Nearest baseline" if run["baseline"] else f"DTCC seed {run['seed']}",
            np.asarray(run["departure_hourly_counts"]),
        )
        for run in results
    ]
    report = {
        "evaluation_version": 1,
        "model_version": datasets.people.version,
        "transit_service_date": transit_service_date,
        "reference_version": "10.5281/zenodo.13882078",
        "reference_file_sha256": expected,
        "primary_area_file_sha256": boundary_sha256,
        "area": name,
        "primary_area": site,
        "site_bounds_epsg3006": list(area.bounds),
        "runs": results,
        "gapsim_departure_hourly_counts": np.histogram(gap["departure"], bins=edges)[
            0
        ].tolist(),
        "field_observations": "See people methods and observation report for coverage and access.",
    }
    for key in ("trips_per_diary", "zero_trip_percentage"):
        values = [r[key]["dtcc"] for r in results if not r["baseline"]]
        report[f"{key}_seed_range"] = [min(values), max(values)]
    (output / f"{site}-comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    departures.append(("GAPSIM", np.histogram(gap["departure"], bins=edges)[0]))
    for label, counts in departures:
        ax.stairs(counts * 100 / max(1, counts.sum()), edges / 60, label=label)
    ax.set(
        xlabel="Local hour",
        ylabel="Share of adult trips per hour (%)",
        title=f"{name.title()} · 2019 adult residents",
    )
    ax.legend()
    fig.savefig(output / f"{site}-departures.png", dpi=160)
    plt.close(fig)
    print(
        name,
        [
            (r["seed"], r["baseline"], r["people"], r["trips_per_diary"])
            for r in results
        ],
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--site", choices=SITES, action="append")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--transit-service-date", choices=["2019-10-15"])
    args = parser.parse_args()
    for site in args.site or SITES:
        run(
            site,
            args.evidence,
            args.output,
            args.workers,
            args.transit_service_date,
        )


if __name__ == "__main__":
    main()
