"""Core mesh-generation repair precedes finite-element conversion and solving."""

import numpy as np
import pytest

pytest.importorskip("dolfinx")
from dtcc_core import datasets
from dtcc_core.model import VolumeMesh
from petsc4py import PETSc

from dtcc_sim._flow import solver as flow
from dtcc_sim import NavierStokesSolver, NavierStokesParameters, StokesSolver, StokesParameters

pytestmark = [pytest.mark.fenics, pytest.mark.simulation]


def corner_mesh():
    mesh = VolumeMesh(
        vertices=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        cells=np.array([[0, 1, 2, 3]], dtype=np.int64),
        boundary_faces=np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
        boundary_markers=np.full(4, -1, dtype=np.int32),
    )
    return mesh


@pytest.mark.parametrize(
    "solver_type,parameter_type",
    [
        (NavierStokesSolver, NavierStokesParameters),
        (StokesSolver, StokesParameters),
    ],
)
def test_generated_mesh_repair_precedes_conversion_and_removes_pressure_defect(
    monkeypatch,
    solver_type,
    parameter_type,
):
    if not np.issubdtype(PETSc.ScalarType, np.floating):
        pytest.skip("Flow requires real PETSc")
    source = corner_mesh()
    requests = []

    def generate(**kwargs):
        requests.append(kwargs)
        assert kwargs["repair_boundary_corners"] is True
        return source.repair_boundary_corners()

    monkeypatch.setattr(datasets, "city_volume_mesh", generate)
    options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1,
        "mat_mumps_icntl_25": 0,
    }
    numerical_options = {
        "petsc_options"
        if parameter_type is NavierStokesParameters
        else "petsc_stokes": options
    }
    params = parameter_type(
        wind_speed=0.0,
        nu=0.1,
        closed_cavity=True,
        inlet_marker=-1,
        mesh_max_mesh_size=7.0,
        mesh_domain_height=23.0,
        mesh_min_building_detail=0.75,
        mesh_raster_cell_size=1.25,
        mesh_raster_radius=2.5,
        **numerical_options,
    )
    solver = solver_type(bounds=(0.0, 0.0, 1.0, 1.0), params=params)
    result = solver.simulate()
    assert requests == [
        {
            "bounds": (0.0, 0.0, 1.0, 1.0),
            "max_mesh_size": 7.0,
            "domain_height": 23.0,
            "min_building_detail": 0.75,
            "raster_cell_size": 1.25,
            "raster_radius": 2.5,
            "repair_boundary_corners": True,
            "lod": "lod1", "lod2_method": "catalogue", "flat_ground": False,
        }
    ]
    assert result.num_vertices == 5 and result.num_cells == 4
    assert result.find_boundary_corners().num_corners == 0
    assert solver.params.mesh_refinement_levels == 0
    assert solver.diagnostics.get("mesh_refinements", 0) == 0
    assert solver.diagnostics["pressure_gauge"] == "zero_mean"
    assert solver.diagnostics["stop_reason"] == "converged"
    assert solver.mesh.topology.index_map(3).size_global == 4
    assert {field.name for field in result.fields} == {"velocity", "pressure", "speed"}
    np.testing.assert_array_equal(source.cells, [[0, 1, 2, 3]])
    np.testing.assert_array_equal(result.vertices[:4], source.vertices)


def test_file_input_preserves_pressure_guard_and_points_to_core_repair(tmp_path):
    if not np.issubdtype(PETSc.ScalarType, np.floating):
        pytest.skip("Flow requires real PETSc")
    path = tmp_path / "corner.xdmf"
    source = corner_mesh()
    source.vertices += [319895.962899, 6398909.716755, 12.5]
    source.save(path)
    solver = NavierStokesSolver(
        mesh_path=str(path),
        params=NavierStokesParameters(
            wind_speed=0.0, nu=0.1, closed_cavity=True, inlet_marker=-1
        ),
    )
    with pytest.raises(
        RuntimeError, match="Generate with repair_boundary_corners=True"
    ):
        solver.simulate()
    assert solver.params.mesh_refinement_levels == 0
    assert solver.diagnostics["stop_reason"] == "solver_failure"
    assert len(solver.diagnostics["uncoupled_pressure_coordinates"]) == 4
    coordinates = np.asarray(solver.diagnostics["uncoupled_pressure_coordinates"])
    np.testing.assert_array_equal(
        np.sort(coordinates, axis=0), np.sort(source.vertices, axis=0)
    )
    assert solver.mesh.topology.index_map(3).size_global == 1


def test_generated_mesh_is_repaired_before_partitioning(monkeypatch):
    from mpi4py import MPI

    if not np.issubdtype(PETSc.ScalarType, np.floating):
        pytest.skip("Flow requires real PETSc")
    calls = []

    def generate(**kwargs):
        assert MPI.COMM_WORLD.rank == 0
        assert kwargs["repair_boundary_corners"] is True
        calls.append(kwargs)
        return corner_mesh().repair_boundary_corners()

    monkeypatch.setattr(datasets, "city_volume_mesh", generate)
    solver = NavierStokesSolver(bounds=(0.0, 0.0, 1.0, 1.0))
    solver._build_mesh_from_bounds()
    assert MPI.COMM_WORLD.allreduce(len(calls), op=MPI.SUM) == 1
    assert solver.mesh.topology.index_map(3).size_global == 4
    if MPI.COMM_WORLD.rank == 0:
        assert solver.volume_mesh_dtcc.num_cells == 4
        assert solver.volume_mesh_dtcc.find_boundary_corners().num_corners == 0
