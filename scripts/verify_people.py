"""Reproduce and audit People packages using only archived primary data."""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import date
from io import TextIOWrapper
from pathlib import Path
from resource import RUSAGE_SELF, getrusage
from time import perf_counter
from zipfile import ZipFile

from dtcc_core.datasets import load_model_package
from pyproj import Transformer

from dtcc_sim import datasets

SITES = {
    "johanneberg": [319720, 6397660, 320220, 6398160],
    "malmo": [373769, 6162013, 374269, 6162513],
}


def _no_evaluation(event, args):
    if event == "socket.connect":
        raise RuntimeError("The package verification run uses archived inputs only")
    if event == "open" and isinstance(args[0], str):
        value = args[0].lower()
        if any(token in value for token in ("gapsim", "/evaluation/", "zenodo")):
            raise RuntimeError(
                "Evaluation files are unavailable to this generation run"
            )


def _write(model, path):
    package = model.export(path, canonical=True, format="geojson")
    restored = load_model_package(package.path)
    assert len(model.people) == len(restored.people)
    assert (
        model.dataset_context.provenance.sources
        == restored.dataset_context.provenance.sources
    )
    return restored


def _check_timetable(model, inputs, service_date):
    source = next(
        source
        for source in model.dataset_context.provenance.sources
        if source["name"] == "Samtrafiken Trafiklab GTFS Sverige 2"
    )
    archive = inputs / source["archive"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == source["sha256"]
    assert source["selection"]["service_date"] == service_date
    trips = Counter()
    expected = set()
    sample = None
    for person in model.people:
        rides = Counter()
        for path in person.paths:
            if path.state != "transit":
                continue
            _, trip, seq = path.edge_ids[-1].split(":")
            expected.add(
                (trip, seq, path.place_id.removeprefix("gtfs:"), int(path.end))
            )
            assert path.start.is_integer() and path.end.is_integer()
            board = tuple(path.samples[0, 1:3])
            trips[(trip, int(path.start), board)] += 1
            rides[path.journey_id] += 1
        if any(count > 1 for count in rides.values()) and sample is None:
            journey = next(key for key, count in rides.items() if count > 1)
            sample = {
                "person_id": person.id,
                "journey_id": journey,
                "stages": [
                    {
                        "state": p.state,
                        "start": p.start,
                        "end": p.end,
                        "place_id": p.place_id,
                        "path_vertices": len(p.samples),
                        "start_xyz": p.samples[0, 1:].tolist(),
                        "end_xyz": p.samples[-1, 1:].tolist(),
                        "first_edge": p.edge_ids[0] if p.edge_ids else None,
                    }
                    for p in person.paths
                    if p.journey_id == journey
                ],
            }
    arrivals = set()
    departures = set()
    trip_ids = {trip for trip, _, _ in trips}

    def clock(raw):
        hours, minutes, seconds = map(int, raw.split(":"))
        return hours * 3600 + minutes * 60 + seconds

    with ZipFile(archive) as feed:

        def rows(name):
            with TextIOWrapper(feed.open(name), encoding="utf-8-sig") as stream:
                yield from csv.DictReader(stream)

        day = date.fromisoformat(service_date)
        compact_date = day.strftime("%Y%m%d")
        weekday = day.strftime("%A").lower()
        active = {
            row["service_id"]
            for row in rows("calendar.txt")
            if row["start_date"] <= compact_date <= row["end_date"]
            and row[weekday] == "1"
        }
        for row in rows("calendar_dates.txt"):
            if row["date"] == compact_date:
                if row["exception_type"] == "1":
                    active.add(row["service_id"])
                elif row["exception_type"] == "2":
                    active.discard(row["service_id"])
        assert trip_ids <= {
            row["trip_id"] for row in rows("trips.txt") if row["service_id"] in active
        }
        projector = Transformer.from_crs(4326, 3006, always_xy=True)
        stop_coordinates = {
            row["stop_id"]: projector.transform(
                float(row["stop_lon"]), float(row["stop_lat"])
            )
            for row in rows("stops.txt")
        }
        _read_stop_times(feed, trip_ids, stop_coordinates, clock, departures, arrivals)
    assert trips and trips.keys() <= departures
    assert expected <= arrivals
    return {
        "service_date": service_date,
        "source_sha256": source["sha256"],
        "rides_verified_against_stop_times": sum(trips.values()),
        "distinct_service_trips": len(trip_ids),
        "sample_transfer": sample,
    }


def _read_stop_times(feed, trip_ids, coordinates, clock, departures, arrivals):
    with TextIOWrapper(feed.open("stop_times.txt"), encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        fields = next(reader)
        trip, arr, dep, stop, seq, pickup, dropoff = map(
            fields.index,
            (
                "trip_id",
                "arrival_time",
                "departure_time",
                "stop_id",
                "stop_sequence",
                "pickup_type",
                "drop_off_type",
            ),
        )
        for row in reader:
            if row[trip] not in trip_ids:
                continue
            if row[dep] and row[pickup] in {"", "0"}:
                departures.add((row[trip], clock(row[dep]), coordinates[row[stop]]))
            if row[arr] and row[dropoff] in {"", "0"}:
                arrivals.add((row[trip], row[seq], row[stop], clock(row[arr])))


def run(name, output, inputs, service_date=None):
    started = perf_counter()
    options = {"transit_service_date": service_date} if service_date else {}
    model = datasets.people(bounds=SITES[name], region=name, seed=42, **options)
    path = output / f"{name}.dtccpkg"
    restored = _write(model, path)
    report = {
        "site": name,
        "model_version": restored.dataset_context.identity.version,
        "seconds_including_package_roundtrip": perf_counter() - started,
        "package_bytes": path.stat().st_size,
        "package_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "peak_rss_native_ru_maxrss": getrusage(RUSAGE_SELF).ru_maxrss,
        **restored.attributes["diagnostics"],
        "visible_walking_or_waiting_at_0800": sum(
            p.state in {"walk", "wait"}
            for _, p in restored.positions_at(28800, outdoor_only=True)
        ),
    }
    if name == "johanneberg":
        bounds = [319845, 6397785, 320095, 6398035]
        overlap = datasets.people(bounds=bounds, region=name, seed=42, **options)
        references = {p.id: p for p in restored.people}
        assert all(p.paths == references[p.id].paths for p in overlap.people)
        section = (319970, 6397810), (319970, 6397890)
        events = restored.passages(*section, interval=(28800, 29100))
        assert events == overlap.passages(*section, interval=(28800, 29100))
        report.update(
            overlap_people=len(overlap.people),
            screenline_passages_0800_0805=len(events),
            screenline_events=[
                {"person_id": e.person_id, "second": e.second, "direction": e.direction}
                for e in events
            ],
        )
        for bounds in (
            [322225, 6398200, 322245, 6398220],
            [318200, 6399800, 318220, 6399820],
            [322290, 6398150, 322300, 6398160],
        ):
            empty = datasets.people(bounds=bounds, region=name, seed=42, **options)
            if not empty.people:
                _write(empty, output / "empty-covered.dtccpkg")
                report.update(
                    empty_bounds=bounds,
                    empty_coverage=empty.attributes["diagnostics"]["coverage"],
                )
                break
    if service_date:
        report["timetable_verification"] = _check_timetable(
            restored, inputs, service_date
        )
    (output / f"{name}-verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--site", choices=SITES, required=True)
    parser.add_argument("--transit-service-date", choices=["2019-10-15"])
    args = parser.parse_args()
    os.environ["DTCC_PEOPLE_INPUTS"] = str(args.inputs.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    sys.addaudithook(_no_evaluation)
    run(args.site, args.output, args.inputs, args.transit_service_date)


if __name__ == "__main__":
    main()
