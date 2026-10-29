"""Native DTCC mesh inputs and equation-specific solver contracts."""

import numpy as np
import pytest

pytest.importorskip("dolfinx")
from petsc4py import PETSc

from dtcc_core import BoxMesh
from dtcc_core.model import VolumeMesh
import dtcc_sim
from dtcc_sim import (
    HeatSolver, HeatParameters, NavierStokesSolver, NavierStokesParameters,
    StokesSolver, StokesParameters,
)

pytestmark = pytest.mark.fenics


def _require_real_petsc():
    if not np.issubdtype(PETSc.ScalarType, np.floating):
        pytest.skip("Heat and flow equations require real PETSc")


@pytest.mark.parametrize("equation,prefix", [
    ("helmholtz", "Helmholtz"), ("navierstokes", "NavierStokes"),
    ("stokes", "Stokes"), ("heat", "Heat"),
])
def test_equation_package_exposes_canonical_solver_and_parameters(equation, prefix):
    import importlib

    package = importlib.import_module(f"dtcc_sim.{equation}")
    solver = getattr(package, f"{prefix}Solver")
    parameters = getattr(package, f"{prefix}Parameters")
    assert solver is getattr(dtcc_sim, f"{prefix}Solver")
    assert parameters is getattr(dtcc_sim, f"{prefix}Parameters")
    assert solver.__module__ == f"dtcc_sim.{equation}.solver"
    assert parameters.__module__ == f"dtcc_sim.{equation}.parameters"


def test_heat_accepts_dtcc_mesh_and_returns_dtcc_fields(tmp_path):
    _require_real_petsc()
    parameters = HeatParameters(
        sigma=1, T_ambient=10,
        wall_bc_type="dirichlet", wall_value=10,
        roof_bc_type="dirichlet", roof_value=10,
        ground_bc_type="dirichlet", ground_value=10,
        open_bc_type="dirichlet", open_value=10,
    )
    solver = HeatSolver(mesh=BoxMesh(2, 2, 2), params=parameters, output_dir=tmp_path)
    result = solver.simulate()
    assert isinstance(result, VolumeMesh)
    field = next(f for f in result.fields if f.name == "temperature")
    np.testing.assert_allclose(field.values, 10, atol=1e-7)
    assert (tmp_path / "solution.xdmf").is_file()


@pytest.mark.parametrize("solver_type,parameter_type", [
    (StokesSolver, StokesParameters), (NavierStokesSolver, NavierStokesParameters),
])
def test_flow_accepts_dtcc_mesh(solver_type, parameter_type, tmp_path):
    _require_real_petsc()
    parameters = parameter_type(
        wind_speed=.1, wind_dir_deg=270, wall_model="noslip",
        side_top_boundary="slip", nu=.1,
        inlet_profile="power_law", power_law_alpha=1., u_ref_height=1.,
    )
    solver = solver_type(mesh=BoxMesh(3, 3, 3), params=parameters, output_dir=tmp_path)
    result = solver.simulate()
    assert isinstance(result, VolumeMesh)
    assert {field.name for field in result.fields} == {"velocity", "pressure", "speed"}
    assert all(np.isfinite(field.values).all() for field in result.fields)
    assert solver.diagnostics["equations"] == parameters.equations
    assert (tmp_path / "solution_velocity.xdmf").is_file()


def test_equation_names_replace_application_solver_names():
    assert not hasattr(dtcc_sim, "UrbanHeatSimulator")
    assert not hasattr(dtcc_sim, "UrbanWindSimulator")
    assert not hasattr(dtcc_sim, "HelmholtzSimulator")
    with pytest.raises(TypeError, match="StokesParameters"):
        StokesSolver(params=NavierStokesParameters())
    with pytest.raises(TypeError, match="NavierStokesParameters"):
        NavierStokesSolver(params=StokesParameters())


def test_dtcc_heat_mesh_requires_boundary_tags():
    mesh = BoxMesh(1, 1, 1)
    mesh.boundary_faces = np.empty((0, 3), dtype=np.int64)
    mesh.boundary_markers = np.empty(0, dtype=np.int64)
    with pytest.raises(ValueError, match="tagged boundary"):
        HeatSolver(mesh=mesh).simulate()


def test_flow_equation_cannot_be_changed_behind_solver_name():
    with pytest.raises(ValueError):
        StokesParameters().equations = "navier_stokes"
    invalid = StokesParameters().model_copy(update={"equations": "navier_stokes"})
    with pytest.raises(ValueError, match="equations"):
        StokesSolver(params=invalid)


@pytest.mark.parametrize("solver_type", [StokesSolver, NavierStokesSolver])
def test_flow_rejects_complex_runtime_before_mesh_or_output(solver_type, monkeypatch, tmp_path):
    monkeypatch.setattr(PETSc, "ScalarType", np.complex128)
    monkeypatch.setattr(
        solver_type, "_load_if_needed",
        lambda self: pytest.fail("must fail before loading or generating a mesh"),
    )
    output = tmp_path / "results"
    with pytest.raises(RuntimeError, match="conda activate fenicsx-env"):
        solver_type(bounds=(0, 0, 10, 10), output_dir=output)
    assert not output.exists()


@pytest.mark.parametrize("solver_type", [StokesSolver, NavierStokesSolver])
def test_flow_rechecks_runtime_before_simulating(solver_type, monkeypatch, tmp_path):
    monkeypatch.setattr(PETSc, "ScalarType", np.float64)
    solver = solver_type(output_dir=tmp_path / "results")
    monkeypatch.setattr(PETSc, "ScalarType", np.complex128)
    monkeypatch.setattr(
        solver, "_load_if_needed",
        lambda: pytest.fail("must fail before loading or generating a mesh"),
    )
    with pytest.raises(RuntimeError, match="real PETSc"):
        solver.simulate()
    assert not solver.output_dir.exists()
