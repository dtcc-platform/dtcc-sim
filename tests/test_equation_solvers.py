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
        wind_speed=1, wind_dir_deg=270, inlet_profile="uniform", wall_model="noslip",
        side_top_boundary="slip", dt=0.05, max_steps=3, min_steps=1,
    )
    solver = solver_type(mesh=BoxMesh(2, 2, 2), params=parameters, output_dir=tmp_path)
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
