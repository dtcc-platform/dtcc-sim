import numpy as np
import pytest
from dtcc_core.model import RoadNetwork, Transform

from dtcc_sim.people.routing import Router, timed_route


def fixture_network():
    return RoadNetwork(
        id="roads",
        transform=Transform(srs="EPSG:3006"),
        vertices=np.array(
            [[0, 35], [85, 35], [85, 130], [180, 130], [170, 45], [230, 45]],
            dtype=float,
        ),
        edges=np.array([[0, 1], [1, 2], [2, 3], [4, 5]]),
        attributes={
            "highway": ["footway"] * 4,
            "end_node_highway": ["traffic_signals", None, None, None],
        },
    )


def test_walk_edges_waits_and_unreachable():
    router = Router(fixture_network())
    assert router.between(router.xyz[0], router.xyz[4]) is None
    route = router.between(router.xyz[0], router.xyz[3])
    stages = timed_route(route, 100, 1, "work", "journey", signal_wait=12)
    assert [stage.state for stage in stages] == ["walk", "wait", "walk"]
    assert stages[0].end == 185
    assert stages[1].end == 197
    assert stages[2].length == 190


def test_router_reports_a_link_closure_it_cannot_apply():
    with pytest.raises(ValueError, match="Unknown walk link"):
        Router(fixture_network(), closed_edge="missing")
