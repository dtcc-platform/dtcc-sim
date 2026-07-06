from types import SimpleNamespace

import numpy as np
import pytest

import dtcc_core.datasets as datasets
from dtcc_core.model import Field, RoadNetwork
from dtcc_sim.traffic import TrafficAssignmentParameters, TrafficAssignmentSimulator
import dtcc_sim.datasets  # noqa: F401


pytestmark = pytest.mark.simulation


def _roads(oneway=False):
    roads = RoadNetwork()
    roads.vertices = np.array(
        [
            [0.0, 0.0],
            [100.0, 0.0],
            [200.0, 0.0],
        ],
        dtype=float,
    )
    roads.edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    roads.length = np.array([100.0, 100.0], dtype=float)
    roads.attributes = {
        "highway": ["residential", "residential"],
        "lanes": ["2", "2"],
        "maxspeed_kmh": [50.0, 50.0],
        "oneway": [oneway, oneway],
    }
    return roads


def _zones():
    fields = {
        "population_total": Field(
            name="population_total",
            values=np.array([[100.0], [100.0]]),
            dim=1,
        ),
        "employed_residents_total": Field(
            name="employed_residents_total",
            values=np.array([[50.0], [50.0]]),
            dim=1,
        ),
    }

    def to_arrays(include_attributes=True):
        return {
            "centroids": np.array([[0.0, 0.0], [200.0, 0.0]], dtype=float),
            "codes": np.array(["A", "B"], dtype=object),
            "fields": {name: field.values for name, field in fields.items()},
        }

    return SimpleNamespace(fields=fields, to_arrays=to_arrays)


def test_traffic_assignment_adds_edge_attributes():
    params = TrafficAssignmentParameters(
        trips_per_employed=1.0,
        peak_hour_factor=1.0,
        max_iterations=5,
    )
    result = TrafficAssignmentSimulator(
        roads=_roads(),
        zones=_zones(),
        params=params,
    ).simulate()

    assert isinstance(result, RoadNetwork)
    assert result.attributes["flow"] == [100.0, 100.0]
    assert result.attributes["flow_forward"] == [50.0, 50.0]
    assert result.attributes["flow_reverse"] == [50.0, 50.0]
    assert result.attributes["flow_assigned"] == [100.0, 100.0]
    assert result.attributes["flow_background"] == [0.0, 0.0]
    np.testing.assert_allclose(result.attributes["capacity"], [1980.0, 1980.0])
    np.testing.assert_allclose(
        result.attributes["capacity_forward"],
        [990.0, 990.0],
    )
    np.testing.assert_allclose(
        result.attributes["capacity_reverse"],
        [990.0, 990.0],
    )
    np.testing.assert_allclose(result.attributes["free_flow_time"], [7.2, 7.2])
    assert min(result.attributes["travel_time"]) >= 7.2
    assert len(result.attributes["travel_time"]) == 2
    assert len(result.attributes["volume_capacity_ratio"]) == 2
    assert max(result.attributes["speed"]) <= 50.0


def test_traffic_assignment_can_add_background_flow():
    params = TrafficAssignmentParameters(
        trips_per_employed=0.0,
        background_flow_fraction=0.1,
        max_iterations=3,
    )
    result = TrafficAssignmentSimulator(
        roads=_roads(),
        zones=_zones(),
        params=params,
    ).simulate()

    assert min(result.attributes["flow_background"]) > 0.0
    assert result.attributes["flow_assigned"] == [0.0, 0.0]
    assert result.attributes["flow"] == result.attributes["flow_background"]
    assert max(result.attributes["volume_capacity_ratio"]) == 0.1


def test_traffic_assignment_respects_oneway_edges():
    params = TrafficAssignmentParameters(
        trips_per_employed=1.0,
        peak_hour_factor=1.0,
        max_iterations=3,
    )
    sim = TrafficAssignmentSimulator(
        roads=_roads(oneway=True),
        zones=_zones(),
        params=params,
    )
    result = sim.simulate()

    assert result.attributes["flow_forward"] == [50.0, 50.0]
    assert result.attributes["flow_reverse"] == [0.0, 0.0]
    assert sim.diagnostics["total_demand"] == 50.0
    assert sim.diagnostics["assigned_demand"] == 50.0
    assert sim.diagnostics["unassigned_demand"] == 0.0
    assert sim.diagnostics["directed_arc_count"] == 2
    assert sim.diagnostics["bidirectional"] is True


def test_traffic_assignment_requires_population_and_employment_fields():
    zones = SimpleNamespace(
        fields={},
        to_arrays=lambda include_attributes=True: {
            "centroids": np.array([[0.0, 0.0], [200.0, 0.0]], dtype=float),
            "codes": np.array(["A", "B"], dtype=object),
            "fields": {},
        },
    )

    with pytest.raises(ValueError, match="population"):
        TrafficAssignmentSimulator(
            roads=_roads(),
            zones=zones,
            params=TrafficAssignmentParameters(),
        ).simulate()

    population_field = Field(
        name="population_total",
        values=np.array([[100.0], [100.0]]),
        dim=1,
    )
    population_only = SimpleNamespace(
        fields={"population_total": population_field},
        to_arrays=lambda include_attributes=True: {
            "centroids": np.array([[0.0, 0.0], [200.0, 0.0]], dtype=float),
            "codes": np.array(["A", "B"], dtype=object),
            "fields": {"population_total": population_field.values},
        },
    )

    with pytest.raises(ValueError, match="employment"):
        TrafficAssignmentSimulator(
            roads=_roads(),
            zones=population_only,
            params=TrafficAssignmentParameters(),
        ).simulate()


def test_traffic_simulation_dataset_returns_roadnetwork(monkeypatch):
    expected_roads = _roads()
    expected_zones = _zones()

    monkeypatch.setattr(datasets, "roads", lambda bounds: expected_roads)
    monkeypatch.setattr(
        datasets,
        "deso",
        lambda bounds, statistics, statistics_year=None: expected_zones,
    )

    result = datasets.traffic_simulation(
        bounds=(0.0, 0.0, 200.0, 10.0),
        trips_per_employed=1.0,
        peak_hour_factor=1.0,
        background_flow_fraction=0.1,
        max_iterations=3,
    )

    assert isinstance(result, RoadNetwork)
    assert min(result.attributes["flow_background"]) > 0.0
    assert min(result.attributes["flow"]) > 100.0
    assert result.attributes["simulation_diagnostics"]["total_demand"] == 100.0
    assert (
        result.attributes["simulation_diagnostics"]["background_flow_fraction"] == 0.1
    )


def test_traffic_simulation_dataset_protobuf_format(monkeypatch):
    expected_roads = _roads()
    expected_zones = _zones()

    monkeypatch.setattr(datasets, "roads", lambda bounds: expected_roads)
    monkeypatch.setattr(
        datasets,
        "deso",
        lambda bounds, statistics, statistics_year=None: expected_zones,
    )

    payload = datasets.traffic_simulation(
        bounds=(0.0, 0.0, 200.0, 10.0),
        trips_per_employed=1.0,
        peak_hour_factor=1.0,
        max_iterations=3,
        format="pb",
    )
    restored = RoadNetwork()
    restored.from_proto(payload)

    assert isinstance(payload, bytes)
    assert restored.attributes["flow"] == [100.0, 100.0]
