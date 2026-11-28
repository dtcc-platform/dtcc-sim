import os

import numpy as np
import pytest
from dtcc_core.datasets import get_dataset, load_model_package
from dtcc_core.io import save_model
from dtcc_core.model import (
    Bounds,
    PeopleSimulation,
    Person,
    RoadNetwork,
    Transform,
)
from dtcc_core.model.object.people import TimedPath

from dtcc_sim import datasets
from dtcc_sim.people import population
from dtcc_sim.people.intervention import close_walk_link
from dtcc_sim.people.population import _ages, _members
from dtcc_sim.people.routing import Router, timed_route
from dtcc_sim.people.scenario import rng, simulation, walking_preview


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


def test_fixed_demand_closure_rejects_scheduled_results():
    result = PeopleSimulation(attributes={"transit_service_date": "2019-10-15"})
    with pytest.raises(ValueError, match="scheduled transit"):
        close_walk_link(result, fixture_network(), "0")


def test_fixed_demand_closure_rejects_unreachable_detour():
    network = RoadNetwork(
        vertices=np.array([[0, 0], [90, 0], [0, 50], [90, 50]], dtype=float),
        edges=np.array([[0, 1], [0, 2], [1, 3]]),
        attributes={"highway": ["footway"] * 3},
    )
    result = PeopleSimulation()
    path = TimedPath("walk", "work", [[5, 0, 0, 0], [65, 90, 0, 0]], edge_ids=("0",))
    result.add_child(Person(id="person", paths=(path,)))
    with pytest.raises(ValueError, match="No walk detour.*person"):
        close_walk_link(result, network, "0")


def test_household_partition_preserves_people():
    stream = rng(42, "zone", "age")
    people = _ages({"25-29": 4, "5-9": 2}, 7, stream)
    households, unmatched = _members(
        people, {"TOTALT": 3, "ESMB": 1, "SBMB": 1, "OVRIGA": 1}, 1, stream
    )
    assert unmatched == 0
    assert len(households) == 3
    assert sum(len(group) for _, group in households) == 7
    assert sum(age is None for _, group in households for age, _ in group) == 1


def test_registered_preview_round_trip_without_network(tmp_path, monkeypatch):
    network = fixture_network()
    path = tmp_path / "fixture.dtcc"
    save_model(network, path)
    monkeypatch.setenv("DTCC_PEOPLE_PREVIEW_ROADS", str(path))
    args = {"bounds": [-20, 0, 250, 180], "profile": "walking_preview", "seed": 17}
    assert get_dataset("people") is datasets.people
    model = datasets.people(**args)
    assert model.people
    native = model.export(tmp_path / "native.dtccpkg", canonical=True)
    assert [artifact.role for artifact in native.artifacts] == ["canonical_model"]
    package = model.export(
        tmp_path / "people.dtccpkg", canonical=True, format="geojson"
    )
    assert [artifact.role for artifact in package.artifacts] == [
        "canonical_model",
        "derived",
    ]
    reloaded = load_model_package(package.path)
    assert reloaded.people[0].paths == model.people[0].paths
    (roads,) = reloaded.get_children(RoadNetwork)
    np.testing.assert_array_equal(roads.edges, network.edges)
    assert (
        reloaded.dataset_context.provenance.sources
        == model.dataset_context.provenance.sources
    )
    assert reloaded.attributes["profile"] == "walking_preview"


@pytest.mark.skipif(
    not os.getenv("DTCC_PEOPLE_RUN_REAL_TESTS"),
    reason="requires pinned SCB and OSM inputs",
)
def test_overlapping_views_keep_paths_and_screenlines():
    outer = datasets.people(
        bounds=[319720, 6397660, 320220, 6398160], region="johanneberg"
    )
    inner_bounds = [319845, 6397785, 320095, 6398035]
    inner = datasets.people(bounds=inner_bounds, region="johanneberg")
    by_id = {person.id: person for person in outer.people}
    assert inner.people
    for person in inner.people:
        assert person.paths == by_id[person.id].paths
    screenline = (319900, 6397785), (319900, 6398035)
    assert outer.passages(*screenline) == inner.passages(*screenline)


def test_empty_preview():
    model = walking_preview(
        Bounds(500, 500, 600, 600),
        datasets.people.validate(
            {"bounds": [500, 500, 600, 600], "profile": "walking_preview"}
        ),
        fixture_network(),
        {"name": "fixture"},
    )
    assert model.people == []


def test_population_result_does_not_share_cached_state_and_keys_runs_by_input_hash(
    tmp_path, monkeypatch
):
    bounds = [0, 0, 20, 20]
    record = {"name": "johanneberg", "bounds": bounds}
    path = TimedPath("indoor", "home", [[0, 5, 5, 0], [86400, 5, 5, 0]])
    person = Person(id="person", household_id="home", home_id="home", paths=(path,))
    households, zonal, survey = (
        {"home": ["person"]},
        {"zone": {"count": 1}},
        {"rates": [1]},
    )
    revisions = []
    input_file = tmp_path / "control.json"
    input_file.write_text("first")
    monkeypatch.setattr(population, "select_region", lambda *args: (tmp_path, record))
    monkeypatch.setattr(
        population,
        "archives",
        lambda *args: [{"name": "Control", "sha256": input_file.read_text()}],
    )

    def regional_day(*args):
        revisions.append(args[-1])
        return [person], households, zonal, {}, survey, [], [], [], None

    monkeypatch.setattr(population, "_regional_day", regional_day)
    first = datasets.people(bounds=bounds)
    first.people[0].home_id = "edited"
    first.attributes["households"]["home"].clear()
    first.attributes["survey"]["rates"].clear()
    first.attributes["zonal_controls"]["zone"]["count"] = 0
    first.presentation.annotations[-1]["evidence"]["gapsim"]["area"].clear()

    input_file.write_text("second")
    next_run = datasets.people(bounds=bounds)
    assert next_run.people[0].home_id == "home"
    assert next_run.people[0].paths[0].samples.flags.writeable is False
    assert next_run.household_members("home") == ("person",)
    assert next_run.attributes["survey"]["rates"] == [1]
    assert next_run.attributes["zonal_controls"]["zone"]["count"] == 1
    assert next_run.presentation.annotations[-1]["evidence"]["gapsim"]["area"] == [
        "Johanneberg",
        "Askim",
        "Hammarkullen",
    ]
    assert revisions == [("first",), ("second",)]
    assert next_run.provenance.sources[0]["sha256"] == "second"


@pytest.mark.parametrize(
    "options", [{"baseline": True}, {"closed_walk_edge": "fixture"}]
)
def test_walking_preview_rejects_ignored_population_options(options):
    with pytest.raises(ValueError, match="population profile"):
        datasets.people.validate(
            {"bounds": [0, 0, 20, 20], "profile": "walking_preview", **options}
        )


@pytest.mark.parametrize(
    ("region", "options", "evaluated"),
    [
        ("johanneberg", {}, True),
        ("askim", {"seed": 17}, True),
        ("hammarkullen", {"seed": 83}, True),
        ("askim", {"baseline": True, "seed": 42}, True),
        ("johanneberg", {"baseline": True, "seed": 17}, False),
        ("hammarkullen", {"seed": 82}, False),
        ("johanneberg", {"closed_walk_edge": "fixture"}, False),
        ("malmo", {}, False),
    ],
)
@pytest.mark.parametrize("service_date", [None, "2019-10-15"])
def test_current_evaluation_reference_is_scoped_to_tested_scenario(
    region, options, evaluated, service_date
):
    model = PeopleSimulation(
        attributes={"region": region, "source_provenance": [], "diagnostics": {}}
    )
    args = datasets.people.validate(
        {
            "bounds": [319720, 6397660, 320220, 6398160],
            "transit_service_date": service_date,
            **options,
        }
    )
    context = datasets.people._context_for_result(model, args)
    references = [
        a for a in context.presentation.annotations if a["relationship"] != "methods"
    ]
    (reference,) = references
    assert reference["relationship"] == (
        "model_evaluation" if evaluated else "related_model_evaluation"
    )
    assert reference["model_version"] == datasets.people.version == "0.2.1"
    assert reference["version"] == "2019-v1"
    section = "#transit-timetable-sensitivity" if service_date else ""
    assert reference["url"].endswith(f"/docs/people-validation.md{section}")
    assert reference["applies_to"]["transit_service_date"] == service_date
    assert reference["applies_to"]["regions"] == ["johanneberg", "askim", "hammarkullen"]
    observations = reference["evidence"]["independent_pedestrian_observations"]
    assert observations["status"] == "unscored"
    limitation = "The GAPSIM comparison did not evaluate this scenario."
    assert (limitation in context.presentation.limitations) == (not evaluated)


def test_link_closure_qualifies_the_evaluation_in_the_exported_package(tmp_path):
    bounds = Bounds(0, 0, 230, 180)
    args = datasets.people.validate({"bounds": list(bounds.tuple)})
    model = simulation(bounds, args, [], {}, [], region="johanneberg")
    model.dataset_context = datasets.people._context_for_result(model, args)
    closed, _ = close_walk_link(model, fixture_network(), "0")
    restored = load_model_package(
        closed.export(tmp_path / "closure.dtccpkg", canonical=True).path
    )
    assert model.presentation.annotations[-1]["relationship"] == "model_evaluation"
    assert [entry["relationship"] for entry in restored.presentation.annotations] == [
        "methods",
        "related_model_evaluation",
        "scenario_intervention",
    ]
