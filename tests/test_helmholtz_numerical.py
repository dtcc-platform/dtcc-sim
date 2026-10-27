"""Small public-API solves with independent accuracy and residual checks."""

import numpy as np
import pytest

pytest.importorskip("dolfinx")
from mpi4py import MPI
from dolfinx import mesh as dmesh, io

from dtcc_sim.helmholtz import HelmholtzParameters, HelmholtzSimulator


@pytest.mark.fenics
def test_public_plane_wave_solve_matches_known_solution(tmp_path, record_property):
    mesh = dmesh.create_unit_cube(MPI.COMM_WORLD, 6, 6, 6)
    path = tmp_path / "mesh.xdmf"
    with io.XDMFFile(MPI.COMM_WORLD, str(path), "w") as output:
        output.write_mesh(mesh)
    params = HelmholtzParameters(
        rhs_mode="plane_wave_exact",
        boundary_mode="impedance_only",
        frequency_hz=343 * 1.5 / (2 * np.pi),
        sound_speed=343,
        plane_wave_direction_x=1,
        plane_wave_direction_y=0.3,
        plane_wave_direction_z=0.2,
        n_subdomains=2,
        partitioner="structured",
        nx_sub=2,
        ny_sub=1,
        nz_sub=1,
        overlap_layers=1,
        oversample_layers=1,
        workers=1,
        n_modes_per_patch=2,
        outer_rtol=1e-10,
        verbose=False,
    )
    sim = HelmholtzSimulator(
        mesh_path=path, output_dir=tmp_path / "case", params=params
    )
    result = sim.simulate()
    fields = {f.name: f.values.ravel() for f in result.fields}
    direction = np.array([1.0, 0.3, 0.2])
    direction /= np.linalg.norm(direction)
    exact = np.exp(1j * 1.5 * (result.vertices @ direction))
    numerical = fields["helmholtz_real"] + 1j * fields["helmholtz_imaginary"]
    error = np.linalg.norm(numerical - exact) / np.linalg.norm(exact)
    record_property("relative_vertex_error", float(error))
    record_property("true_relative_residual", sim.result.relative_residual)
    record_property("iterations", sim.result.iterations)
    assert error < 0.02
    assert sim.diagnostics["converged"] and sim.result.relative_residual < 1e-7
    np.testing.assert_allclose(fields["helmholtz_amplitude"], abs(numerical))
