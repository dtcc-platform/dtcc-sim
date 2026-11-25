"""Synthesis of reproducible regional simulation days."""

import os
from hashlib import blake2s, sha256
from pathlib import Path

import numpy as np
from dtcc_core import datasets, io
from dtcc_core.model import PeopleSimulation, Person, Transform

from .routing import Router, stationary, timed_route


def rng(seed, *keys):
    """Independent deterministic stream for one scenario entity and stage."""
    key = ":".join(map(str, (seed, *keys))).encode()
    return np.random.default_rng(
        int.from_bytes(blake2s(key, digest_size=16).digest(), "little")
    )


def fingerprint(path):
    """Return the immutable SHA-256 of one archived upstream artifact."""
    with Path(path).open("rb") as stream:
        from hashlib import file_digest

        return file_digest(stream, sha256).hexdigest()


JOHANNEBERG_SCREENLINE = {
    "name": "Johanneberg E319970",
    "start": [319970, 6397810],
    "end": [319970, 6397890],
    "bin_seconds": 300,
}


def _demonstration_screenline(bounds):
    if all(
        bounds.xmin <= x < bounds.xmax and bounds.ymin <= y < bounds.ymax
        for x, y in (JOHANNEBERG_SCREENLINE["start"], JOHANNEBERG_SCREENLINE["end"])
    ):
        return JOHANNEBERG_SCREENLINE
    return None


def simulation(bounds, args, sources, diagnostics, people, **facts):
    """Create a native result in SWEREF99 TM with a Swedish wall clock."""
    result = PeopleSimulation(
        id=f"people-{args.reference_year}-{args.day_type}-{args.seed}",
        transform=Transform(srs="EPSG:3006"),
        attributes={
            "view_bounds": list(bounds.tuple),
            "reference_year": args.reference_year,
            "day_type": args.day_type,
            "time_zone": "Europe/Stockholm",
            "clock": "half-open intervals in seconds after local midnight",
            "vertical_reference": "metres above a flat local surface; terrain elevations unavailable",
            "screenline": _demonstration_screenline(bounds),
            "seed": args.seed,
            "profile": args.profile,
            "showcase_start": 8 * 3600,
            "source_provenance": sources,
            "diagnostics": diagnostics,
            **facts,
        },
    )
    result.add_children(people)
    return result


def build(bounds, args):
    """Resolve scenario inputs independently of any evaluation dataset."""
    if args.profile == "population":
        from .population import build_population

        return build_population(bounds, args)

    from .inputs import OSM, preview_map_files, read_osm

    path = os.getenv("DTCC_PEOPLE_PREVIEW_ROADS")
    files = (
        [Path(path)]
        if path
        else preview_map_files(bounds, args.reference_year, args.region)
    )
    if path and Path(path).suffix == ".dtcc":
        network = io.load_model(path)
    elif path:
        network = io.roadnetwork.load(path, load_geometry=False, round_coordinates=None)
    elif files:
        network, _, _, _ = read_osm(files)
    else:
        network = datasets.roads(bounds=bounds.tuple, network="walk")
    source = {
        "name": "OpenStreetMap walking network",
        "license": "ODbL 1.0",
        "url": "https://www.openstreetmap.org/copyright",
        "vintage": "snapshot at retrieval or cache date",
    }
    if path:
        source["sha256"] = fingerprint(files[0])
    elif files:
        source["archives"] = [
            {"archive": file.name, "sha256": fingerprint(file)} for file in files
        ]
        source["url"] = OSM
    else:
        source["sha256"] = sha256(
            network.to_proto().SerializeToString(deterministic=True)
        ).hexdigest()
    return walking_preview(bounds, args, network, source)


def walking_preview(bounds, args, network, source):
    """Route a small playback sample on mapped geography without a demand claim."""
    router = Router(network)
    vertices = router.xyz[router.anchors]
    inside = router.anchors[
        (vertices[:, 0] > bounds.xmin + 15)
        & (vertices[:, 0] < bounds.xmax - 15)
        & (vertices[:, 1] > bounds.ymin + 15)
        & (vertices[:, 1] < bounds.ymax - 15)
    ]
    people = []
    if len(inside) > 1:
        for index in range(48):
            stream = rng(args.seed, "preview", index)
            a, b = router.xyz[stream.choice(inside, 2, replace=False)]
            if not 65 < np.linalg.norm(b - a) < 400:
                continue
            route = router.between(a, b)
            if route is None:
                continue
            departure = 28800 + index * 1.5
            key = f"walker-{index:02d}"
            walks = timed_route(
                route,
                departure,
                float(stream.uniform(1.1, 1.6)),
                "leisure",
                key,
                signal_wait=8,
            )
            people.append(
                Person(
                    id=key,
                    appearance_key=int(stream.integers(0, 2**32)),
                    paths=(
                        stationary("outdoor", "leisure", "", a, 0, departure),
                        *walks,
                        stationary("indoor", "leisure", "", b, walks[-1].end, 86400),
                    ),
                )
            )
    diagnostics = {
        "name": "Walking preview",
        "sampled_people": len(people),
        "demand_scope": "route playback only; endpoints selected from walkable nodes",
    }
    result = simulation(bounds, args, [source], diagnostics, people)
    result.add_child(network)
    return result
