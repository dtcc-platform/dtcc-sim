"""Primary source archives and geographic feature extraction for people scenarios."""

import json
import os
import re
from dataclasses import dataclass
from itertools import pairwise, product
from pathlib import Path
from xml.etree import ElementTree

import geopandas as gpd
import numpy as np
import requests
from dtcc_core.io.data.deso import download_deso_geodataframe, fetch_deso_control_table
from dtcc_core.model import RoadNetwork, Transform
from platformdirs import user_cache_dir
from pyproj import Transformer
from shapely.geometry import Polygon, box

from .scenario import fingerprint

OSM = "https://api.openstreetmap.org/api/0.6/map"
GRID = "https://geodata.scb.se/geoserver/stat/wfs?service=WFS&version=1.1.0&request=GetFeature&typeName=stat:befolkning_1km_2019&outputFormat=geopackage"
RVU = "https://www.trafa.se/globalassets/statistik/resvanor/2019/resvanor-i-sverige-2019.xlsx"
SCB_GEO = "https://geodata.scb.se/geoserver/stat/wfs"


@dataclass
class Anchor:
    """A destination with a mapped or inferred access point."""

    id: str
    xyz: np.ndarray
    kind: str
    capacity: int
    building_type: str = ""
    footprint: Polygon | None = None
    mapped_access: bool = False
    height: float = 0.0
    height_source: str = ""


LEVEL_HEIGHT = 3.0


LOW_BUILDINGS = {
    "garage": 1,
    "garages": 1,
    "shed": 1,
    "hut": 1,
    "kiosk": 1,
    "roof": 1,
    "carport": 1,
    "service": 1,
    "house": 2,
    "detached": 2,
    "semidetached_house": 2,
    "terrace": 2,
}


def building_height(tags):
    """Return a presentation height in metres and whether it was mapped or inferred."""
    height = tags.get("height", "").removesuffix("m").strip()
    try:
        if float(height) > 0:
            return float(height), "mapped height"
    except ValueError:
        pass
    levels = tags.get("building:levels", "")
    roof = tags.get("roof:levels", "")
    if levels.isdecimal() and int(levels) > 0:
        extra = int(roof) if roof.isdecimal() else 0
        return LEVEL_HEIGHT * (int(levels) + 0.5 * extra), "mapped levels"
    return LEVEL_HEIGHT * LOW_BUILDINGS.get(
        tags.get("building", ""), 3
    ), "inferred floors"


def _tags(element):
    return {tag.attrib["k"]: tag.attrib["v"] for tag in element.iterfind("tag")}


def read_osm(paths):
    """Build a Core road network, building/amenity anchors and tram rails from tiled OSM XML."""
    nodes, ways = {}, {}
    for path in paths:
        root = ElementTree.parse(path).getroot()
        for node in root.iterfind("node"):
            attr = node.attrib
            nodes[int(attr["id"])] = (
                float(attr["lon"]),
                float(attr["lat"]),
                _tags(node),
            )
        for way in root.iterfind("way"):
            tags = _tags(way)
            if any(
                key in tags
                for key in (
                    "highway",
                    "building",
                    "amenity",
                    "shop",
                    "leisure",
                    "public_transport",
                    "railway",
                )
            ):
                ways[int(way.attrib["id"])] = (
                    tuple(int(nd.attrib["ref"]) for nd in way.iterfind("nd")),
                    tags,
                )

    node_ids = sorted(nodes)
    xy = np.array([nodes[n][:2] for n in node_ids], dtype=float).reshape((-1, 2))
    tr = Transformer.from_crs(4326, 3006, always_xy=True)
    x, y = tr.transform(xy[:, 0], xy[:, 1])
    xyz = np.column_stack((x, y, np.zeros(len(x))))
    index = {node: i for i, node in enumerate(node_ids)}
    used = set()
    edges = []
    attr = {
        name: []
        for name in (
            "osm_way_id",
            "segment_index",
            "highway",
            "sidewalk",
            "foot",
            "bicycle",
            "access",
            "vehicle",
            "motor_vehicle",
            "motorcar",
            "motorroad",
            "oneway",
            "start_node_highway",
            "end_node_highway",
            "bridge",
            "tunnel",
            "name",
        )
    }
    homes, activities, rails = [], [], []
    for way_id, (refs, tags) in sorted(ways.items()):
        if not all(ref in nodes for ref in refs):
            continue
        if tags.get("railway") in {"tram", "light_rail"}:
            rails.append(xyz[[index[ref] for ref in refs]])
        if "highway" in tags:
            reverse = tags.get("oneway") == "-1"
            for segment, (a, b) in enumerate(pairwise(refs)):
                if a == b:
                    continue
                if reverse:
                    a, b = b, a
                edges.append((index[a], index[b]))
                used.update((index[a], index[b]))
                record = {
                    **tags,
                    "osm_way_id": way_id,
                    "segment_index": segment,
                    "oneway": reverse
                    or tags.get("oneway") in {"yes", "1", "true"}
                    or tags.get("junction") == "roundabout",
                    "start_node_highway": nodes[a][2].get("highway"),
                    "end_node_highway": nodes[b][2].get("highway"),
                }
                for name, values in attr.items():
                    values.append(record.get(name))

        if len(refs) >= 4 and refs[0] == refs[-1]:
            polygon = Polygon(xyz[[index[ref] for ref in refs], :2])
            if not polygon.is_valid or polygon.area < 8:
                continue
            entrances = [
                ref
                for ref in refs
                if nodes[ref][2].get("entrance") in {"yes", "main", "staircase"}
            ]
            if entrances:
                anchor = xyz[index[entrances[0]]]
            else:
                anchor = np.array([*polygon.representative_point().coords[0], 0.0])
            building = tags.get("building", "")
            inferred_floors = (
                3
                if building in {"apartments", "residential", "yes", "dormitory"}
                else 1
            )
            levels = tags.get("building:levels", "")
            floors = max(1, int(levels)) if levels.isdecimal() else inferred_floors
            height, height_source = building_height(tags)
            kind = _kind(tags)
            key = f"osm-way-{way_id}"
            if (
                building
                in {
                    "apartments",
                    "residential",
                    "house",
                    "detached",
                    "terrace",
                    "semidetached_house",
                    "dormitory",
                    "yes",
                }
                and not kind
            ):
                homes.append(
                    Anchor(
                        key,
                        anchor,
                        "home",
                        max(1, int(polygon.area * floors / 35)),
                        building,
                        polygon,
                        bool(entrances),
                        height,
                        height_source,
                    )
                )
            if kind and kind != "stop":
                activities.append(
                    Anchor(
                        key,
                        anchor,
                        kind,
                        max(1, int(polygon.area * floors / 24)),
                        building,
                        polygon,
                        bool(entrances),
                        height,
                        height_source,
                    )
                )

    for node, (_, _, tags) in sorted(nodes.items()):
        kind = _kind(tags)
        if kind:
            activities.append(
                Anchor(
                    f"osm-node-{node}", xyz[index[node]], kind, 20, mapped_access=True
                )
            )

    kept = sorted(used)
    lookup = {old: new for new, old in enumerate(kept)}
    edge_array = np.array(
        [(lookup[a], lookup[b]) for a, b in edges], dtype=np.int64
    ).reshape((-1, 2))
    vertices = xyz[kept]
    lengths = np.linalg.norm(
        vertices[edge_array[:, 0]] - vertices[edge_array[:, 1]], axis=1
    )
    return (
        RoadNetwork(
            id="osm-regional-roads",
            transform=Transform(srs="EPSG:3006"),
            vertices=vertices,
            edges=edge_array,
            length=lengths,
            attributes=attr,
        ),
        homes,
        activities,
        rails,
    )


def _kind(tags):
    amenity, building = tags.get("amenity"), tags.get("building")
    if (
        tags.get("public_transport") in {"platform", "stop_position", "station"}
        or tags.get("highway") == "bus_stop"
        or tags.get("railway") == "tram_stop"
    ):
        return "stop"
    if amenity in {"school", "kindergarten", "childcare"} or building in {
        "school",
        "kindergarten",
    }:
        return "education"
    if amenity in {"university", "college"} or building in {"university", "college"}:
        return "study"
    if tags.get("shop") or amenity in {"marketplace", "pharmacy", "bank"}:
        return "shopping"
    if tags.get("leisure") or amenity in {
        "restaurant",
        "cafe",
        "cinema",
        "theatre",
        "library",
    }:
        return "leisure"
    if (
        tags.get("office")
        or building in {"commercial", "office", "retail", "industrial"}
        or amenity in {"hospital", "clinic", "townhall"}
    ):
        return "work"
    return None


def _fetch(path, url, params=None):
    if path.exists():
        return
    response = requests.get(
        url,
        params=params,
        timeout=(10, 90),
        headers={"User-Agent": "DTCC People public data retrieval"},
    )
    response.raise_for_status()
    _write(path, response.content)


def _write(path, data):
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(root, region, bounds, year=2019):
    """Archive SCB controls, kilometre grid, travel tables and OSM map tiles."""
    root = Path(root)
    path = root / region
    path.mkdir(parents=True, exist_ok=True)
    _fetch(root / f"rvu-{year}.xlsx", RVU)
    _fetch(root / f"grid-{year}.gpkg", GRID)
    if not (path / "deso.gpkg").exists():
        download_deso_geodataframe(bounds=bounds, year=2018).to_file(path / "deso.gpkg")
    zones = gpd.read_file(path / "deso.gpkg")
    codes = sorted(set(zones.desokod))
    for topic in ("population", "households"):
        if not (path / f"{topic}.json").exists():
            data = fetch_deso_control_table(codes, year, topic) if codes else {}
            _write(path / f"{topic}.json", json.dumps(data).encode())
    xmin, ymin, xmax, ymax = bounds
    cells = max(1, int(np.ceil(max(xmax - xmin, ymax - ymin) / 900)))
    tr = Transformer.from_crs(3006, 4326, always_xy=True)
    if len(zones):
        for i, j in product(range(cells), repeat=2):
            dx, dy = (xmax - xmin) / cells, (ymax - ymin) / cells
            ext = tr.transform_bounds(
                xmin + i * dx, ymin + j * dy, xmin + (i + 1) * dx, ymin + (j + 1) * dy
            )
            _fetch(
                path / f"map-{i}-{j}.osm",
                OSM,
                {"bbox": ",".join(f"{value:.7f}" for value in ext)},
            )
    _write(
        path / "region.json",
        json.dumps({"name": region, "bounds": list(bounds), "year": year}).encode(),
    )
    return path


def _archive_root(year):
    return Path(
        os.getenv("DTCC_PEOPLE_INPUTS", Path(user_cache_dir("dtcc-people")) / str(year))
    )


def _cached_region(root, bounds, year, region):
    for manifest in sorted(root.glob("*/region.json")):
        record = json.loads(manifest.read_text())
        if (
            record["year"] == year
            and (not region or record["name"] == region)
            and box(*record["bounds"]).covers(box(*bounds.tuple))
        ):
            return manifest.parent, record
    return None


def _osm_source_extent(filename):
    with filename.open("rb") as stream:
        head = stream.read(4096).decode("utf-8", errors="replace")
    extent = re.search(r"<bounds[^>]*/>", head)
    return ElementTree.fromstring(extent.group()).attrib if extent else {}


def preview_map_files(bounds, year, region):
    """Find archived OSM tiles near the preview without fetching population inputs."""
    cached = _cached_region(_archive_root(year), bounds, year, region)
    if cached is None:
        return []
    folder, _ = cached
    transform = Transformer.from_crs(3006, 4326, always_xy=True)
    expanded_bounds = box(*bounds.tuple).buffer(100).bounds
    view = box(*transform.transform_bounds(*expanded_bounds))
    files = []
    for path in sorted(folder.glob("*.osm")):
        extent = _osm_source_extent(path)
        if not extent or view.intersects(
            box(
                *(
                    float(extent[key])
                    for key in ("minlon", "minlat", "maxlon", "maxlat")
                )
            )
        ):
            files.append(path)
    return files


def select_region(bounds, year, region):
    """Resolve a pinned region or retrieve a fixed 5 km partition with a halo."""
    root = _archive_root(year)
    cached = _cached_region(root, bounds, year, region)
    if cached is not None:
        return cached
    if region:
        raise ValueError(
            f"No cached {year} region {region!r} covers the requested bounds in {root}"
        )
    x = 5000 * int(bounds.center[0] // 5000)
    y = 5000 * int(bounds.center[1] // 5000)
    regional = (x - 1000, y - 1000, x + 6000, y + 6000)
    if not box(*regional).covers(box(*bounds.tuple)):
        raise ValueError(
            "Request a smaller view or prepare a named region covering these bounds"
        )
    name = f"sweref99tm-{x}-{y}"
    return prepare(root, name, regional, year), {
        "name": name,
        "bounds": regional,
        "year": year,
    }


def archives(path, year, record):
    """Describe only the primary files actually used by the population model."""
    parent = path.parent
    definitions = [
        (path / "deso.gpkg", "SCB DeSO 2018", SCB_GEO, "2018_v3"),
        (parent / f"grid-{year}.gpkg", "SCB population kilometre grid", GRID, year),
        (parent / f"rvu-{year}.xlsx", "Trafikanalys RVU, Statistik 2020:17", RVU, year),
    ]
    for topic in ("population", "households"):
        item = path / f"{topic}.json"
        source = json.loads(item.read_text())
        if source:
            definitions.append((item, f"SCB DeSO {topic}", source["url"], year))
    definitions.extend(
        (p, "OpenStreetMap map extract", OSM, "current at retrieval")
        for p in sorted(path.glob("*.osm"))
    )
    result = []
    for filename, name, url, vintage in definitions:
        source = {
            "name": name,
            "url": url,
            "vintage": vintage,
            "sha256": fingerprint(filename),
            "archive": filename.name,
            "region": record["name"],
            "terms": "ODbL 1.0"
            if url == OSM
            else "Public official statistics; source attribution",
        }
        if filename.suffix == ".json":
            source["selection"] = json.loads(filename.read_text())["request"]
        elif filename.suffix == ".osm":
            extent = _osm_source_extent(filename)
            if extent:
                source["source_extent"] = extent
        result.append(source)
    return result
