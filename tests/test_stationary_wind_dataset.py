"""Stationary wind dataset contracts."""

import pytest
from dtcc_sim.datasets import UrbanWindSimulationArgs, UrbanWindSimulationDataset


@pytest.mark.parametrize("equations", ["navier_stokes", "stokes"])
def test_dataset_schema_and_serialized_defaults(equations):
    args = UrbanWindSimulationArgs(bounds=(0, 0, 1, 1), equations=equations)
    restored = UrbanWindSimulationArgs.model_validate(args.model_dump())
    assert restored.equations == equations
    assert restored.turbulence_model == ("abl" if equations == "stokes" else "mixing_length")
    assert (restored.wall_model, restored.inlet_profile, restored.side_top_boundary) == ("log_law", "log_law", "slip")
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
                                   turbulence_model="constant", wall_model="noslip",
                                   wind_speed=.1, side_top_boundary="slip", inlet_profile="power_law",
                                   power_law_alpha=1., u_ref_height=1., format="dtcc")
    payload = UrbanWindSimulationDataset().build(args)
    target = tmp_path / "wind.dtcc"
    target.write_bytes(payload)
    result = io.load_model(target)
    assert {field.name for field in result.fields} == {"velocity", "pressure", "speed"}
    assert all(np.isfinite(field.values).all() for field in result.fields)


def test_dataset_atmospheric_defaults_reach_solver_parameters():
    from dtcc_sim.navierstokes.parameters import NavierStokesParameters
    from dtcc_sim.stokes.parameters import StokesParameters
    for equations, parameter_type, model in [("navier_stokes", NavierStokesParameters, "mixing_length"),
                                             ("stokes", StokesParameters, "abl")]:
        args = UrbanWindSimulationArgs(bounds=(0, 0, 1, 1), equations=equations)
        params = parameter_type.model_validate({
            name: getattr(args, name) for name in parameter_type.model_fields if getattr(args, name) is not None
        })
        assert (params.turbulence_model, params.wall_model, params.inlet_profile) == (model, "log_law", "log_law")
    with pytest.raises(ValueError, match="turbulence_model='constant'"):
        UrbanWindSimulationArgs(bounds=(0, 0, 1, 1), inlet_profile="uniform")
    constant = UrbanWindSimulationArgs(bounds=(0, 0, 1, 1), turbulence_model="constant", nu_t=2.)
    assert constant.nu_t == 2.
