"""Native flow results from distributed solves are gathered on rank zero."""

import numpy as np
import pytest

from dtcc_sim.output import RankZeroResult


def test_rank_zero_result_ignores_output_calls():
    result = RankZeroResult()
    assert result.info() is None
    assert result.plot(field="speed") is None
    assert result.save("ignored.dtcc") is None
    with pytest.raises(AttributeError, match="MPI rank 0"):
        result.fields
    assert getattr(result, "simulation_diagnostics", None) is None


@pytest.mark.fenics
@pytest.mark.simulation
def test_generated_mesh_result_is_gathered_on_rank_zero(monkeypatch):
    pytest.importorskip("dolfinx")
    from mpi4py import MPI
    from petsc4py import PETSc

    if not np.issubdtype(PETSc.ScalarType, np.floating):
        pytest.skip("flow solvers require real PETSc")
    from dtcc_core import BoxMesh
    from dtcc_core.model import VolumeMesh
    from dtcc_sim import NavierStokesParameters, NavierStokesSolver
    from dtcc_sim._flow import solver as flow

    monkeypatch.setattr(flow, "prepare_city_mesh", lambda bounds, settings: (BoxMesh(4, 4, 4), {"request": {}}))
    parameters = NavierStokesParameters(
        wind_speed=.1, wind_dir_deg=270, wall_model="noslip", side_top_boundary="slip", nu=.1,
        inlet_profile="power_law", power_law_alpha=1., u_ref_height=1.,
    )
    solver = NavierStokesSolver(bounds=(0., 0., 1., 1.), params=parameters)
    result = solver.simulate()

    if MPI.COMM_WORLD.rank != 0:
        assert isinstance(result, RankZeroResult)
        assert result.info() is None and result.plot(field="speed") is None
        return
    assert isinstance(result, VolumeMesh)
    fields = {field.name: np.asarray(field.values) for field in result.fields}
    vertices = np.asarray(result.vertices)
    assert fields["velocity"].shape == (len(vertices), 3) and fields["pressure"].size == len(vertices)
    np.testing.assert_allclose(fields["speed"], np.linalg.norm(fields["velocity"], axis=1))
    inlet = np.isclose(vertices[:, 0], vertices[:, 0].min())
    expected = np.zeros((inlet.sum(), 3))
    expected[:, 0] = .1 * (vertices[inlet, 2] - vertices[:, 2].min())
    np.testing.assert_allclose(fields["velocity"][inlet], expected, atol=1e-12)
    ground = np.isclose(vertices[:, 2], vertices[:, 2].min())
    np.testing.assert_allclose(fields["velocity"][ground], 0., atol=1e-12)
    assert np.abs(fields["velocity"][~inlet & ~ground, 0]).min() > 0
    assert result.simulation_diagnostics["equations"] == "navier_stokes"
