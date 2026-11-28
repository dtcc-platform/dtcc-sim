"""Archive public inputs for a fixed Swedish People region."""

import argparse
from pathlib import Path

from dtcc_sim.people.inputs import prepare

REGIONS = {
    "johanneberg": (317500, 6395500, 322500, 6400500),
    "malmo": (371519, 6159763, 376519, 6164763),
    "askim": (314000, 6389500, 320000, 6395500),
    "hammarkullen": (321000, 6405500, 327000, 6411500),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("region")
    parser.add_argument("--bounds", nargs=4, type=float)
    args = parser.parse_args()
    bounds = args.bounds or REGIONS.get(args.region)
    if bounds is None:
        parser.error("Supply --bounds in EPSG:3006 for this region")
    if Path(args.region).name != args.region or args.region in {".", ".."}:
        parser.error("Use a plain folder name for the region")
    print(prepare(args.output, args.region, bounds))


if __name__ == "__main__":
    main()
