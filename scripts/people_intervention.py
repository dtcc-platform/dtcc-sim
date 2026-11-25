"""Compare a link closure on identical synthetic demand across seeds."""

import argparse
import json
from pathlib import Path

from dtcc_core.datasets import load_model_package

from dtcc_sim import datasets
from dtcc_sim.people.inputs import read_osm
from dtcc_sim.people.intervention import close_walk_link

JOHANNEBERG_BOUNDS = (319720, 6397660, 320220, 6398160)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("edge")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--seed", type=int, action="append")
    args = parser.parse_args()

    import os

    os.environ["DTCC_PEOPLE_INPUTS"] = str(args.inputs.resolve())
    network, *_ = read_osm(sorted((args.inputs / "johanneberg").glob("*.osm")))
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in args.seed or (17, 42, 83):
        if seed == 42 and args.baseline:
            baseline = load_model_package(args.baseline)
            if (
                baseline.attributes.get("seed") != seed
                or baseline.attributes.get("region") != "johanneberg"
            ):
                raise ValueError(
                    "The baseline package must be the Johanneberg scenario with seed 42"
                )
        else:
            baseline = datasets.people(
                bounds=JOHANNEBERG_BOUNDS, region="johanneberg", seed=seed
            )
        scenario, report = close_walk_link(baseline, network, args.edge)
        if seed == 42:
            scenario.export(
                args.output / "johanneberg-closure.dtccpkg",
                canonical=True,
                format="geojson",
            )
        report["seed"] = seed
        runs.append(report)
        print(seed, report["statistics"], flush=True)
        (args.output / "closure-metrics.json").write_text(
            json.dumps(runs, indent=2, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    main()
