"""Mesh selection, provenance, and DTCC/FEniCS export checks."""

from dataclasses import replace
import json

import numpy as np
import pytest

from dtcc_sim.helmholtz import (
    CityMeshParameters,
    CityMeshCache,
    HelmholtzParameters,
    HelmholtzWorkflow,
)
from dtcc_sim.helmholtz import meshing, workflow


@pytest.fixture
def mesh_parameters():
    return CityMeshParameters(bounds=(0, 0, 20, 20), h=5)


@pytest.fixture
def generator(monkeypatch):
    calls = []

    def generate(arguments, path):
        calls.append(arguments)
        path.write_text('<Xdmf><DataItem Format="HDF">mesh.h5:/Mesh</DataItem></Xdmf>')
        path.with_suffix(".h5").write_bytes(b"mesh data")
        return {"vertices": 4, "tetrahedra": 1}

    monkeypatch.setattr(meshing, "_generate_mesh", generate)
    monkeypatch.setattr(meshing, "_mesher_versions", lambda: {"test": 1})
    return calls


def test_mesh_generated_once_then_reused(tmp_path, mesh_parameters, generator):
    cache = CityMeshCache(tmp_path / "cache")
    path, record, reused = cache.resolve(mesh_parameters)
    assert not reused and record["h"] == 5
    same, saved, reused = cache.resolve(mesh_parameters)
    assert reused and same == path and saved == record
    assert len(generator) == 1
    assert generator[0]["max_mesh_size"] == 5
    assert generator[0]["boundary_face_markers"] is True
    assert "raster_radius" in record["request"]["arguments"]


@pytest.mark.parametrize("changed", ["h", "bounds", "height", "options", "runtime"])
def test_full_mesh_request_selects_new_entry(
    tmp_path, mesh_parameters, generator, monkeypatch, changed
):
    cache = CityMeshCache(tmp_path / "cache")
    original, _, _ = cache.resolve(mesh_parameters)
    changes = {
        "h": {"h": 3},
        "bounds": {"bounds": (0, 0, 30, 30)},
        "height": {"domain_height": 100},
        "options": {"options": {"raster_radius": 4}},
    }
    if changed == "runtime":
        monkeypatch.setattr(meshing, "_mesher_versions", lambda: {"test": 2})
    else:
        mesh_parameters = replace(mesh_parameters, **changes[changed])
    new, _, reused = cache.resolve(mesh_parameters)
    assert new != original and not reused and len(generator) == 2


def test_corrupt_cache_rejected(tmp_path, mesh_parameters, generator):
    cache = CityMeshCache(tmp_path / "cache")
    path, _, _ = cache.resolve(mesh_parameters)
    path.with_suffix(".h5").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        cache.resolve(mesh_parameters)
    assert len(generator) == 1


def test_failed_generation_not_published(tmp_path, mesh_parameters, monkeypatch):
    def fail(arguments, path):
        path.write_text("partial")
        raise RuntimeError("mesher failed")

    monkeypatch.setattr(meshing, "_generate_mesh", fail)
    cache = CityMeshCache(tmp_path / "cache")
    with pytest.raises(RuntimeError, match="mesher failed"):
        cache.resolve(mesh_parameters)
    assert list(cache.directory.iterdir()) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"h": 0},
        {"bounds": (0, 0, 0, 1)},
        {"options": {"max_mesh_size": 3}},
        {"options": {"unknown_option": 1}},
    ],
)
def test_invalid_mesh_request_rejected(mesh_parameters, kwargs):
    with pytest.raises(ValueError):
        replace(mesh_parameters, **kwargs).resolved_arguments()


def test_plan_does_not_generate_or_create_directories(
    tmp_path, mesh_parameters, generator
):
    instance = HelmholtzWorkflow(
        output_dir=tmp_path / "case",
        mesh_cache_dir=tmp_path / "cache",
        mesh_parameters=mesh_parameters,
    )
    assert instance.describe()["mesh"]["h"] == 5
    assert (
        not generator
        and not (tmp_path / "cache").exists()
        and not (tmp_path / "case").exists()
    )


def test_mesh_only_does_not_solve(tmp_path, mesh_parameters, generator, monkeypatch):
    checked = []
    monkeypatch.setattr(
        workflow, "_check_boundary_tags", lambda path, params: checked.append(path)
    )
    instance = HelmholtzWorkflow(
        output_dir=tmp_path / "case",
        mesh_cache_dir=tmp_path / "cache",
        mesh_parameters=mesh_parameters,
    )
    path = instance.prepare_mesh()
    assert checked == [path] and len(generator) == 1
    assert instance.mesh_record["h"] == 5
    assert not instance.output_dir.exists()


def test_dtcc_export_preserves_readable_boundary_tags(tmp_path, monkeypatch):
    pytest.importorskip("dolfinx")
    from dtcc_core import datasets
    from dtcc_core.model import VolumeMesh

    mesh = VolumeMesh(
        vertices=np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [0.0, 0, 1]]),
        cells=np.array([[0, 1, 2, 3]], dtype=np.int64),
    )
    mesh.boundary_markers = {(0, 1, 2): -1, (0, 1, 3): -2, (0, 2, 3): -3, (1, 2, 3): -4}
    monkeypatch.setattr(datasets, "city_volume_mesh", lambda **kwargs: mesh)
    path = tmp_path / "mesh.xdmf"
    assert meshing._generate_mesh({}, path) == {"vertices": 4, "tetrahedra": 1}
    params = HelmholtzParameters(impedance_facet_tags=(-2, -3, -4))
    report = workflow._check_boundary_tags(path, params)
    assert report["impedance"]["present"] == [-4, -3, -2]
    report = workflow._check_boundary_tags(
        path, replace(params, impedance_facet_tags=(-6, -2))
    )
    assert report["impedance"] == {
        "requested": [-6, -2], "present": [-2], "absent": [-6]
    }
    with pytest.warns(UserWarning, match="No requested impedance"):
        workflow._check_boundary_tags(path, replace(params, impedance_facet_tags=(-6,)))


def test_launcher_paths_are_relative_to_config_file(tmp_path, monkeypatch):
    from dtcc_sim.helmholtz.launch import load_workflow, main

    config = tmp_path / "run.json"
    config.write_text(
        json.dumps(
            {
                "mesh_path": "mesh.xdmf",
                "mesh_h": 5,
                "output_dir": "case",
                "parameters": {"workers": 16},
            }
        )
    )
    loaded = load_workflow(config)
    assert loaded.mesh_path == tmp_path / "mesh.xdmf"
    monkeypatch.setattr(
        HelmholtzWorkflow,
        "prepare_mesh",
        lambda self: pytest.fail("plan must not build"),
    )
    assert main(["--config", str(config)]) == 0
    assert not loaded.output_dir.exists()
