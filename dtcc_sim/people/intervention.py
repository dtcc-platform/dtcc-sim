"""Fixed-demand experiments on pre-existing simulated walking journeys."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace

import numpy as np

from .routing import Router, timed_route


def close_walk_link(model, network, edge):
    """Reroute affected walk stages while keeping people, purposes and endpoints.

    Later stages shift by the travel-time difference until the next indoor
    episode can absorb it. Every affected stage must have a walking detour.
    """
    if model.attributes.get("transit_service_date"):
        raise ValueError("Fixed-demand link closures cannot retime scheduled transit")
    router = Router(network, closed_edge=edge)

    result = deepcopy(model)
    result.attributes["closed_walk_edge"] = edge
    if result.dataset_context is not None:
        result.dataset_context.presentation.annotations.append(
            {
                "relationship": "scenario_intervention",
                "closed_walk_edge": edge,
                "demand": "Fixed people, purposes and stage endpoints",
            }
        )
    stats = Counter()
    redistribution = Counter()
    changes = []
    for person in result.people:
        paths = []
        offset = 0.0
        for path in person.paths:
            if edge not in path.edge_ids:
                if not offset:
                    paths.append(path)
                    continue
                samples = path.samples.copy()
                samples[:, 0] += offset
                if path.state == "indoor" and path.end > path.start + offset:
                    samples[-1, 0] = path.end
                    offset = 0
                paths.append(replace(path, samples=samples))
                continue
            begin, end = path.samples[0, 1:], path.samples[-1, 1:]
            route = router.between(begin, end)
            stats["affected_walk_stages"] += 1
            if route is None:
                raise ValueError(
                    f"No walk detour for {person.id} after closing link {edge!r}"
                )
            detour = timed_route(
                route,
                path.start + offset,
                path.length / (path.end - path.start),
                path.purpose,
                path.journey_id,
                signal_wait=12,
            )
            if not detour:
                raise ValueError(
                    "Closing this link pushes a trip outside the simulated day"
                )
            offset = detour[-1].end - path.end
            paths.extend(detour)
            changes.append(
                float(detour[-1].end - detour[0].start - path.end + path.start)
            )
            redistribution.subtract(path.edge_ids)
            for stage in detour:
                redistribution.update(stage.edge_ids)
            stats["rerouted_walk_stages"] += 1
        person.paths = tuple(paths)
    stats["rerouted_people"] = sum(
        a.paths != b.paths for a, b in zip(model.people, result.people)
    )
    stats["net_walk_and_signal_seconds"] = sum(changes)
    stats["mean_walk_and_signal_seconds_per_detour"] = (
        float(np.mean(changes)) if changes else None
    )
    redistribution.pop("", None)
    return result, {
        "closed_walk_edge": edge,
        "statistics": dict(stats),
        "segment_traversal_delta": dict(
            sorted((k, v) for k, v in redistribution.items() if v)
        ),
    }
