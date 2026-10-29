"""Geometry identity, cache reuse and shared solver mesh controls."""

from dataclasses import replace
import json

import numpy as np
import pytest

from dtcc_core import BoxMesh
from dtcc_sim import meshing
from dtcc_sim.meshing import (
    CityMeshSettings,
    CityMeshParameters,
    CityMeshCache,
    prepare_city_mesh,
)


@pytest.fixture
def generated(monkeypatch):
    calls = []

    def build(arguments, path):
        calls.append(arguments)
        mesh = BoxMesh(2, 2, 2)
        mesh.vertices += [319895.0, 6398909.0, 12.5]
        mesh.save(path)
        return {"vertices": int(mesh.num_vertices), "tetrahedra": int(mesh.num_cells)}

    monkeypatch.setattr(meshing, "_generate_mesh", build)
    monkeypatch.setattr(meshing, "_mesher_versions", lambda: {"test": 1})
    return calls


def test_scalar_flow_and_helmholtz_requests_reuse_identical_mesh(tmp_path, generated):
    bounds = (319895.0, 6398909.0, 320095.0, 6399109.0)
    settings = CityMeshSettings(
        mesh_max_mesh_size=12.5,
        mesh_domain_height=80.0,
        mesh_lod="lod2",
        mesh_cache_dir=tmp_path / "cache",
    )
    native, first = prepare_city_mesh(bounds, settings)
    assert not first["reused"]
    parameters = CityMeshParameters(
        bounds=bounds,
        h=12.5,
        domain_height=80.0,
        min_building_detail=0.5,
        options={"lod": "lod2", "flat_ground": False, "repair_boundary_corners": True},
    )
    path, record, reused = CityMeshCache(settings.mesh_cache_dir).resolve(parameters)
    assert reused and str(path) == first["mesh_path"]
    again, second = prepare_city_mesh(bounds, settings)
    assert second["reused"]
    assert len(generated) == 1
    assert record["request"] == first["request"] == second["request"]
    assert record["files"] == first["files"] == second["files"]
    np.testing.assert_array_equal(native.vertices, again.vertices)
    np.testing.assert_array_equal(native.cells, again.cells)
    assert native.transform.srs == again.transform.srs == "EPSG:3006"


@pytest.mark.parametrize(
    "option",
    [
        {"lod": "lod1"},
        {"lod2_method": "ransac"},
        {"flat_ground": True},
        {"repair_boundary_corners": False},
        {"raster_cell_size": 4.0},
    ],
)
def test_geometry_change_selects_another_cache_entry(tmp_path, generated, option):
    cache = CityMeshCache(tmp_path)
    parameters = CityMeshParameters(
        bounds=(0.0, 0.0, 20.0, 20.0), h=5.0, options={"lod": "lod2"}
    )
    first, _, _ = cache.resolve(parameters)
    changed = replace(parameters, options={**parameters.options, **option})
    second, record, reused = cache.resolve(changed)
    assert first != second and not reused
    assert len(generated) == 2
    for key, value in option.items():
        assert record["request"]["arguments"][key] == value


def test_cache_integrity_is_shared_across_consumers(tmp_path, generated):
    settings = CityMeshSettings(mesh_cache_dir=tmp_path)
    _, record = prepare_city_mesh((0.0, 0.0, 20.0, 20.0), settings)
    path = meshing.Path(record["mesh_path"])
    path.with_suffix(".h5").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        prepare_city_mesh((0.0, 0.0, 20.0, 20.0), settings)
    assert len(generated) == 1


def test_uncached_generation_preserves_returned_native_model(monkeypatch):
    from dtcc_core import datasets

    native = BoxMesh(1, 1, 1)
    seen = {}

    def generate(**arguments):
        seen.update(arguments)
        return native

    monkeypatch.setattr(datasets, "city_volume_mesh", generate)
    mesh, record = prepare_city_mesh(
        (0.0, 0.0, 20.0, 20.0), CityMeshSettings(mesh_lod="lod2")
    )
    assert mesh is native
    assert seen["lod"] == "lod2" and seen["repair_boundary_corners"] is True
    assert seen["flat_ground"] is False
    assert record["reused"] is False


def test_scalar_dataset_schemas_expose_common_geometry_settings():
    from dtcc_sim.datasets import (
        UrbanHeatSimulationArgs,
        AirQualityFieldArgs,
        UrbanWindSimulationArgs,
    )

    for model in (
        UrbanHeatSimulationArgs,
        AirQualityFieldArgs,
        UrbanWindSimulationArgs,
    ):
        args = model(
            bounds=(0.0, 0.0, 20.0, 20.0), mesh_lod="lod2", mesh_cache_dir="shared"
        )
        assert args.mesh_lod == "lod2" and args.mesh_repair_boundary_corners is True
        restored = model.model_validate_json(args.model_dump_json())
        assert restored.mesh_cache_dir == args.mesh_cache_dir
        assert set(CityMeshSettings.model_fields) <= set(model.model_fields)


@pytest.mark.parametrize("directory", ["", " "])
def test_empty_cache_directory_is_rejected(directory):
    with pytest.raises(ValueError, match="nonempty"):
        CityMeshSettings(mesh_cache_dir=directory)
