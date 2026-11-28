import json

import geopandas as gpd
import numpy as np
import pytest
from dtcc_core.datasets import load_model_package
from dtcc_core.model import RoadNetwork
from shapely.geometry import box

from dtcc_sim import datasets
from dtcc_sim.people import population
from dtcc_sim.people.diaries import ARCHIVE, Diary
from dtcc_sim.people.inputs import Anchor


def test_local_diaries_preserve_linked_stops_and_home_only_days(tmp_path, monkeypatch):
    x, y = 319800, 6397900
    bounds = [x - 100, y - 100, x + 500, y + 100]
    area = box(*bounds)
    homes = [Anchor("home", np.array([x, y, 0.0]), "home", 2)]
    activities = [
        Anchor("office", np.array([x + 100, y, 0.0]), "work", 1),
        Anchor("shop", np.array([x + 300, y, 0.0]), "shopping", 1),
    ]
    network = RoadNetwork(
        vertices=np.array([[x, y], [x + 100, y], [x + 300, y]], dtype=float),
        edges=np.array([[0, 1], [1, 2]]),
        attributes={"highway": ["footway"] * 2},
    )
    regional = (
        gpd.GeoDataFrame({"desokod": ["zone"]}, geometry=[area], crs=3006),
        gpd.GeoDataFrame({"beftotalt": [2]}, geometry=[area], crs=3006),
        network,
        homes,
        activities,
        [],
        {"zone": {"totalt": 2, "25-29": 1, "75-79": 1}},
        {"zone": {"TOTALT": 2, "ESUB": 2}},
        {},
    )
    region = tmp_path / "fixture"
    region.mkdir()
    monkeypatch.setattr(population, "_inputs", lambda *_: regional)
    monkeypatch.setattr(
        population,
        "select_region",
        lambda *_: (region, {"bounds": bounds, "name": "fixture"}),
    )
    monkeypatch.setattr(population, "archives", lambda *_: [])
    source = {
        "reference_year": 2019,
        "day_type": "weekday",
        "source_name": "Fabricated diary test inputs",
        "source_url": "https://example.test/survey",
        "cohorts": [
            {
                "age_min": 25,
                "age_max": 29,
                "diaries": [
                    {
                        "weight": 2.5,
                        "journeys": [
                            {"purpose": "work", "mode": "walk", "departure_bin": 16},
                            {
                                "purpose": "shopping",
                                "mode": "walk",
                                "departure_bin": 32,
                            },
                            {"purpose": "home", "mode": "walk", "departure_bin": 33},
                        ],
                    }
                ],
            },
            {
                "age_min": 75,
                "age_max": 79,
                "diaries": [{"weight": 1.25, "journeys": []}],
            },
        ],
    }
    (tmp_path / ARCHIVE).write_text(json.dumps(source))
    model = datasets.people(
        bounds=bounds, region="fixture", survey_profile="local_diaries"
    )
    restored = load_model_package(
        model.export(tmp_path / "people.dtccpkg", canonical=True).path
    )
    assert len(restored.people) == 2
    active = next(person for person in restored.people if person.age_band == "25-29")
    assert [stage.state for stage in active.paths] == ["indoor", "walk"] * 3 + [
        "indoor"
    ]
    assert [stage.place_id for stage in active.paths[::2]] == [
        "home",
        "office",
        "shop",
        "home",
    ]
    assert list(active.paths[3].samples[0, 1:]) == [x + 100, y, 0]
    assert restored.attributes["diagnostics"]["diary_profiles_routed"] == 1
    assert restored.attributes["diagnostics"]["diary_profiles_at_home"] == 1
    assert any(len(person.paths) == 1 for person in restored.people)
    provenance = restored.dataset_context.provenance.sources[0]
    assert provenance["archive"] == ARCHIVE and len(provenance["sha256"]) == 64
    assert "cohorts" not in provenance
    assert not any(
        entry["relationship"] == "model_evaluation"
        for entry in restored.dataset_context.presentation.annotations
    )


def test_diary_rejects_incomplete_chains():
    with pytest.raises(ValueError, match="finish at home"):
        Diary.model_validate(
            {
                "weight": 1,
                "journeys": [{"purpose": "work", "mode": "walk", "departure_bin": 16}],
            }
        )


def test_local_diary_takes_precedence_over_synthetic_school_runs(tmp_path, monkeypatch):
    x, y = 319800, 6397900
    bounds = [x - 100, y - 100, x + 500, y + 100]
    area = box(*bounds)
    network = RoadNetwork(
        vertices=np.array([[x, y], [x + 100, y]], dtype=float),
        edges=np.array([[0, 1]]),
        attributes={"highway": ["footway"]},
    )
    regional = (
        gpd.GeoDataFrame({"desokod": ["zone"]}, geometry=[area], crs=3006),
        gpd.GeoDataFrame({"beftotalt": [2]}, geometry=[area], crs=3006),
        network,
        [Anchor("home", np.array([x, y, 0.0]), "home", 2)],
        [Anchor("school", np.array([x + 100, y, 0.0]), "education", 20)],
        [],
        {"zone": {"totalt": 2, "25-29": 1, "5-9": 1}},
        {"zone": {"TOTALT": 1, "ESMB": 1}},
        {
            "table_1_annual_thousand": {"work": [1, 0, 0, 0]},
            "table_5_daily_rate_and_95ci": {"shopping": [0], "leisure": [0]},
        },
    )
    monkeypatch.setattr(population, "_inputs", lambda *_: regional)
    monkeypatch.setattr(
        population,
        "select_region",
        lambda *_: (tmp_path / "fixture", {"bounds": bounds, "name": "fixture"}),
    )
    monkeypatch.setattr(population, "archives", lambda *_: [])
    monkeypatch.setattr(
        population, "_role", lambda age, _: "education" if age < 18 else "home"
    )
    monkeypatch.setattr(population, "PRIMARY_PARTICIPATION", 1)
    (tmp_path / ARCHIVE).write_text(
        json.dumps(
            {
                "reference_year": 2019,
                "day_type": "weekday",
                "source_name": "Fabricated child diary",
                "source_url": "https://example.test/survey",
                "cohorts": [
                    {
                        "age_min": 6,
                        "age_max": 9,
                        "diaries": [{"weight": 1, "journeys": []}],
                    }
                ],
            }
        )
    )
    model = datasets.people(
        bounds=bounds, region="fixture", survey_profile="local_diaries"
    )
    child = next(person for person in model.people if person.age_band == "5-9")
    assert child.attributes["synthetic_age"] >= 6
    assert not child.attributes["journeys"]
    assert model.attributes["diagnostics"]["diary_profiles_at_home"] == 1
    assert all(not person.attributes["journeys"] for person in model.people)
