"""Stationary wind dataset and service boundary contracts."""

import pytest
from dtcc_sim.datasets import UrbanWindSimulationArgs, UrbanWindSimulationDataset


@pytest.mark.parametrize("equations", ["navier_stokes", "stokes"])
def test_dataset_schema_and_serialized_defaults(equations):
    args = UrbanWindSimulationArgs(bounds=(0, 0, 1, 1), equations=equations)
    restored = UrbanWindSimulationArgs.model_validate(args.model_dump())
    assert restored.equations == equations
    schema = UrbanWindSimulationArgs.model_json_schema()["properties"]
    assert {"nonlinear_rtol", "nonlinear_max_it", "residual_rtol", "mesh_refinement_levels"} <= schema.keys()
    assert not {"dt", "max_steps", "simulation_mode", "petsc_pressure"} & schema.keys()


@pytest.mark.parametrize("key,value", [("dt", .1), ("max_steps", 5), ("simulation_mode", "statistical_steady"), ("inlet_ramp_steps", 10)])
def test_dataset_rejects_removed_options(key, value):
    with pytest.raises(ValueError, match="Stationary flow configuration"):
        UrbanWindSimulationArgs(bounds=(0, 0, 1, 1), **{key: value})


def test_dataset_propagates_numerical_failure(monkeypatch, tmp_path):
    pytest.importorskip("dolfinx")
    from dtcc_sim import NavierStokesSolver
    def fail(self):
        raise RuntimeError("Stationary flow solve failed: nonlinear iteration limit")
    monkeypatch.setattr(NavierStokesSolver, "simulate", fail)
    args = UrbanWindSimulationArgs(bounds=(0, 0, 1, 1), format="dtcc", output_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="nonlinear iteration limit"):
        UrbanWindSimulationDataset().build(args)
    import json
    assert [path.name for path in tmp_path.iterdir()] == ["timings.json"]
    assert json.loads((tmp_path / "timings.json").read_text())["status"] == "failed"


def test_service_advertises_stationary_schema_and_reports_failure():
    pytest.importorskip("fastapi")
    from service.routes import list_datasets, _task_state_payload
    from types import SimpleNamespace
    result = list_datasets()
    schema = result["datasets"]["urban_wind_simulation"]["args_schema"]["properties"]
    assert "nonlinear_max_it" in schema and "dt" not in schema
    payload = _task_state_payload(SimpleNamespace(state="FAILURE", result=RuntimeError("Stationary flow solve failed")))
    assert payload["status"] == "failed"
    assert "result_file" not in payload


@pytest.mark.fenics
@pytest.mark.parametrize("equations", ["navier_stokes", "stokes"])
def test_native_wind_dataset_serialization(equations, monkeypatch, tmp_path):
    pytest.importorskip("dolfinx")
    import numpy as np
    from petsc4py import PETSc
    if not np.issubdtype(PETSc.ScalarType, np.floating):
        pytest.skip("Wind execution requires real PETSc")
    from dtcc_core import BoxMesh, datasets, io
    def city_mesh(**kwargs):
        assert kwargs["repair_boundary_corners"] is True
        return BoxMesh(3, 3, 3)
    monkeypatch.setattr(datasets, "city_volume_mesh", city_mesh)
    args = UrbanWindSimulationArgs(bounds=(0, 0, 1, 1), equations=equations, nu=.1,
                                   wind_speed=.1, side_top_boundary="slip", inlet_profile="power_law",
                                   power_law_alpha=1., u_ref_height=1., format="dtcc")
    payload = UrbanWindSimulationDataset().build(args)
    target = tmp_path / "wind.dtcc"
    target.write_bytes(payload)
    result = io.load_model(target)
    assert {field.name for field in result.fields} == {"velocity", "pressure", "speed"}
    assert all(np.isfinite(field.values).all() for field in result.fields)
