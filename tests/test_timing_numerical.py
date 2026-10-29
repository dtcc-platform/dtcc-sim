"""Timing reports around real numerical solves and output files."""

import json

import numpy as np
import pytest

pytest.importorskip("dolfinx")
from mpi4py import MPI
from petsc4py import PETSc
from dtcc_core import BoxMesh

pytestmark = [pytest.mark.fenics, pytest.mark.simulation]


def check_report(solver, output, expected):
    report = solver.last_timings
    assert report["status"] == "completed"
    assert report == solver.diagnostics["run_timings"]
    assert report["total_s"] > 0
    paths = {tuple(row["path"]) for row in report["stages"]}
    assert expected <= paths
    assert all(row["elapsed_s"] >= row["self_s"] >= 0 for row in report["stages"])
    assert report["ranks"] == MPI.COMM_WORLD.size
    if MPI.COMM_WORLD.rank == 0:
        assert json.loads((output / "timings.json").read_text()) == report
    return report


@pytest.mark.parametrize(
    "equation", ["heat", "navierstokes", "stokes", "reconstruction"]
)
def test_real_solver_report_includes_actual_solve_and_export(equation, tmp_path):
    if not np.issubdtype(PETSc.ScalarType, np.floating):
        pytest.skip("Requires real PETSc")
    from dtcc_sim import (
        HeatSolver,
        HeatParameters,
        NavierStokesSolver,
        NavierStokesParameters,
        StokesSolver,
        StokesParameters,
        SmoothReconstructionSimulator,
        SmoothReconstructionParameters,
    )

    if MPI.COMM_WORLD.size > 1:
        import dolfinx.mesh as dm

        mesh = dm.create_box(
            MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [2.0, 1.0, 1.0]], [6, 3, 3]
        )
        mesh.topology.create_entities(2)
        mesh.topology.create_connectivity(2, 3)
        faces = dm.exterior_facet_indices(mesh.topology)
        midpoints = dm.compute_midpoints(mesh, 2, faces)
        values = np.full(len(faces), -1, dtype=np.int32)
        for axis, coordinate, marker in [
            (0, 0, -3),
            (0, 2, -4),
            (1, 0, -5),
            (1, 1, -6),
            (2, 1, -2),
        ]:
            values[np.isclose(midpoints[:, axis], coordinate)] = marker
        order = np.argsort(faces)
        tags = dm.meshtags(mesh, 2, faces[order], values[order])
        inputs = dict(mesh=mesh, markers=tags)
    else:
        inputs = dict(mesh=BoxMesh(3, 3, 3))
    if equation == "heat":
        solver = HeatSolver(
            **inputs,
            output_dir=tmp_path,
            params=HeatParameters(
                sigma=1.0,
                T_ambient=7.0,
                wall_value=7.0,
                roof_value=7.0,
                ground_value=7.0,
                open_value=7.0,
                wall_bc_type="dirichlet",
                roof_bc_type="dirichlet",
                ground_bc_type="dirichlet",
                open_bc_type="dirichlet",
            ),
        )
        expected = {("solver", "assembly_and_linear_solve")}
    elif equation == "reconstruction":
        solver = SmoothReconstructionSimulator(
            **inputs,
            output_dir=tmp_path,
            point_coords=np.array([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]]),
            point_values=np.array([7.0, 7.0]),
            params=SmoothReconstructionParameters(background_value=7.0),
        )
        expected = {("solver", "assembly"), ("solver", "linear_solve")}
    else:
        solver_type, parameter_type = (
            (NavierStokesSolver, NavierStokesParameters)
            if equation == "navierstokes"
            else (StokesSolver, StokesParameters)
        )

        def inlet(x):
            return np.array(
                [x[2] * (2 - x[2]), np.zeros(x.shape[1]), np.zeros(x.shape[1])]
            )

        solver = solver_type(
            **inputs,
            output_dir=tmp_path,
            params=parameter_type(nu=0.1, wind_speed=1.0, side_top_boundary="slip"),
            inlet_expression=inlet,
        )
        expected = {
            ("solver", "assembly_and_preconditioner_setup"),
            ("solver", "linear_or_nonlinear_solve"),
        }
    result = solver.simulate()
    first = check_report(
        solver, tmp_path, expected | {("mesh_preparation",), ("export",)}
    )
    if MPI.COMM_WORLD.size == 1:
        assert result.simulation_diagnostics["run_timings"] == first
    solver.simulate()
    second = check_report(solver, tmp_path, expected | {("export",)})
    assert second is not first
    assert (
        next(row for row in second["stages"] if row["path"] == ["export"])["calls"] == 1
    )


def test_complex_solver_report_covers_offline_online_and_export(tmp_path):
    if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
        pytest.skip("Requires complex PETSc")
    from dtcc_sim import HelmholtzSolver, HelmholtzParameters

    solver = HelmholtzSolver(
        mesh=BoxMesh(3, 3, 3),
        output_dir=tmp_path,
        params=HelmholtzParameters(
            boundary_mode="impedance_only",
            frequency_hz=50.0,
            n_subdomains=2,
            partitioner="structured",
            nx_sub=2,
            ny_sub=1,
            nz_sub=1,
            workers=1,
            n_modes_per_patch=2,
            overlap_layers=1,
            oversample_layers=1,
            outer_rtol=1e-10,
            verbose=False,
        ),
    )
    result = solver.simulate()
    report = check_report(
        solver,
        tmp_path,
        {
            ("offline_preparation", "assembly"),
            ("offline_preparation", "spectral_and_coarse_setup"),
            ("online_solver", "assembly"),
            ("online_solver", "linear_solve"),
            ("online_solver", "export"),
            ("postprocessing", "export"),
        },
    )
    assert solver.result.converged
    assert result.simulation_diagnostics["run_timings"] == report
    assert solver.result.run_timings == report
    import meshio

    city = meshio.read(tmp_path / "city.vtu")
    assert set(city.cells_dict) == {"triangle"}
    assert np.all(city.cell_data_dict["markers"]["triangle"] == -1)
    assert (
        json.loads((tmp_path / "city.coordinates.json").read_text())
        == solver.result.coordinate_frame
    )
