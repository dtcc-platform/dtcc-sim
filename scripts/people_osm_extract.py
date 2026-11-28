"""Extract a regional People map from an authorized local OSM PBF archive."""

import argparse
from pathlib import Path
from xml.etree import ElementTree as ET

from prepare_people import REGIONS
from pyproj import Transformer

KEYS = (
    "highway",
    "building",
    "amenity",
    "shop",
    "office",
    "leisure",
    "public_transport",
    "entrance",
    "railway",
)


def extract(pbf, out, bounds):
    """Write relevant tagged nodes and ways with original OSM topology."""
    import osmium

    min_x, min_y, max_x, max_y = Transformer.from_crs(
        3006, 4326, always_xy=True
    ).transform_bounds(*bounds)
    nodes = {}
    ways = []

    def inside(lon, lat):
        return min_x <= lon <= max_x and min_y <= lat <= max_y

    processor = (
        osmium.FileProcessor(pbf)
        .with_locations()
        .with_filter(osmium.filter.KeyFilter(*KEYS))
    )
    header = processor.header
    for item in processor:
        if item.is_node():
            if inside(item.lon, item.lat):
                nodes[item.id] = (item.lon, item.lat, dict(item.tags))
        elif item.is_way() and any(
            n.location.valid() and inside(n.lon, n.lat) for n in item.nodes
        ):
            if any(not n.location.valid() for n in item.nodes):
                continue
            refs = []
            for node in item.nodes:
                refs.append(node.ref)
                nodes.setdefault(node.ref, (node.lon, node.lat, {}))
            ways.append((item.id, refs, dict(item.tags)))
    root = ET.Element("osm", {"version": "0.6", "generator": "DTCC regional extract"})
    ET.SubElement(
        root,
        "bounds",
        minlon=str(min_x),
        minlat=str(min_y),
        maxlon=str(max_x),
        maxlat=str(max_y),
    )
    ET.SubElement(root, "meta", osm_base=header.get("osmosis_replication_timestamp"))
    for ref, (lon, lat, tags) in sorted(nodes.items()):
        node = ET.SubElement(root, "node", id=str(ref), lon=str(lon), lat=str(lat))
        for key, value in tags.items():
            ET.SubElement(node, "tag", k=key, v=value)
    for ref, refs, tags in ways:
        way = ET.SubElement(root, "way", id=str(ref))
        for node_ref in refs:
            ET.SubElement(way, "nd", ref=str(node_ref))
        for key, value in tags.items():
            ET.SubElement(way, "tag", k=key, v=value)
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)
    print(f"{out}: {len(nodes)} nodes and {len(ways)} ways")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pbf", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("region", choices=REGIONS)
    args = parser.parse_args()
    extract(args.pbf, args.output, REGIONS[args.region])


if __name__ == "__main__":
    main()
