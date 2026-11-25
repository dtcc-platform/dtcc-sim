"""Score a locally supplied, independently counted pedestrian screenline."""

import argparse
import csv
import json
from collections import defaultdict
from datetime import date
from itertools import pairwise
from math import isfinite
from pathlib import Path
from statistics import mean

from dtcc_core.datasets import load_model_package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("counts", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--holdout-start", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--screenline",
        type=float,
        nargs=4,
        required=True,
        metavar=("X0", "Y0", "X1", "Y1"),
    )
    parser.add_argument("--edge", action="append")
    args = parser.parse_args()

    model = load_model_package(args.package)
    start, end = args.screenline[:2], args.screenline[2:]
    west, south, east, north = model.attributes["view_bounds"]
    if (
        not all(map(isfinite, args.screenline))
        or start == end
        or not all(west <= x <= east and south <= y <= north for x, y in (start, end))
    ):
        raise ValueError(
            "Supply a finite screenline contained in the package view bounds"
        )
    coverage = model.attributes["diagnostics"].get("coverage")
    if coverage != "covered":
        raise ValueError("Field evaluation requires a covered population scenario")
    rows = defaultdict(dict)
    with args.counts.open(newline="") as stream:
        for line in csv.DictReader(stream):
            day = date.fromisoformat(line["date"])
            interval = (
                int(line["start_second"]),
                int(line["end_second"]),
                line["direction"],
            )
            count = int(line["passages"])
            if day.year != model.attributes["reference_year"] or day.weekday() >= 5:
                raise ValueError(
                    f"Observation date {day} is outside the weekday scenario cohort"
                )
            if (
                interval in rows[day]
                or not 0 <= interval[0] < interval[1] <= 86400
                or count < 0
            ):
                raise ValueError(
                    f"Duplicate or invalid screenline interval {day} {interval}"
                )
            if interval[2] not in {"both", "+1", "-1"}:
                raise ValueError(f"Unrecognized screenline direction {interval[2]}")
            rows[day][interval] = count

    training = [day for day in sorted(rows) if day < args.holdout_start]
    test = [day for day in sorted(rows) if day >= args.holdout_start]
    if not training or not test:
        raise ValueError(
            "Supply both earlier training days and complete held-out dates"
        )
    bins = sorted(rows[training[0]])
    if any(set(rows[day]) != set(bins) for day in rows):
        raise ValueError(
            "All included days must have the same complete observation bins"
        )
    directions = sorted({key[2] for key in bins})
    for direction in directions:
        windows = [(first, last) for first, last, label in bins if label == direction]
        if any(last > next_first for (_, last), (next_first, _) in pairwise(windows)):
            raise ValueError(f"Observation windows overlap for direction {direction}")

    baseline = {key: mean(rows[day][key] for day in training) for key in bins}
    predicted = {}
    events = model.passages(start, end, edge_ids=args.edge)
    for lower, upper, direction in bins:
        predicted[lower, upper, direction] = sum(
            lower <= event.second < upper
            and (direction == "both" or event.direction == int(direction))
            for event in events
        )
    details = []
    for day in test:
        for first, last, direction in bins:
            key = first, last, direction
            details.append(
                {
                    "date": day.isoformat(),
                    "start_second": first,
                    "end_second": last,
                    "direction": direction,
                    "observed": rows[day][key],
                    "historical_mean": baseline[key],
                    "simulated": predicted[key],
                }
            )
    summary = {}
    for direction in directions:
        subset = [row for row in details if row["direction"] == direction]
        peak_errors = []
        for day in test:
            daily = [row for row in subset if row["date"] == day.isoformat()]
            peak_errors.append(
                (
                    max(daily, key=lambda row: row["simulated"])["start_second"]
                    - max(daily, key=lambda row: row["observed"])["start_second"]
                )
                / 60
            )
        summary[direction] = {
            "n_intervals": len(subset),
            "observed_passages": sum(row["observed"] for row in subset),
            "simulated_passages": sum(row["simulated"] for row in subset),
            "mae": mean(abs(row["simulated"] - row["observed"]) for row in subset),
            "signed_bias_per_interval": mean(
                row["simulated"] - row["observed"] for row in subset
            ),
            "historical_mean_mae": mean(
                abs(row["historical_mean"] - row["observed"]) for row in subset
            ),
            "peak_time_mae_minutes": mean(map(abs, peak_errors)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "training_dates": list(map(str, training)),
                "held_out_dates": list(map(str, test)),
                "screenline": args.screenline,
                "units": "pedestrian passages per observation interval",
                "metrics": summary,
                "rows": details,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
