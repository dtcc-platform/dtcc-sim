"""Physical context geometry alongside simulation fields."""

import json

import meshio
import numpy as np
import pytest

from dtcc_core import BoxMesh
from dtcc_core.model import VolumeMesh
from dtcc_sim.output import _export_city_surface


def test_native_surface_preserves_map_coordinates_and_excludes_outer_boundary(tmp_path):
    offset = np.array([319895.0, 6398909.0, 12.5])
    vertices = (
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [9.0, 9.0, 9.0],
            ]
        )
        + offset
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 4]])
    markers = np.array([-1, 0, 1, -2])
    native = VolumeMesh(
        vertices=vertices.copy(),
        cells=np.array([[0, 1, 2, 3]]),
        boundary_faces=faces.copy(),
        boundary_markers=markers.copy(),
    )
    native.transform.srs = "EPSG:3006"
    frame = {"input_crs": "EPSG:3006", "local_to_input_translation": offset.tolist()}
    destination = tmp_path / "solution.xdmf"
    path = _export_city_surface(destination, native_mesh=native, frame=frame)
    assert path == tmp_path / "city.vtu"
    city = meshio.read(path)
    np.testing.assert_array_equal(city.points, vertices[:4])
    np.testing.assert_array_equal(city.cells_dict["triangle"], faces[:3])
    np.testing.assert_array_equal(
        city.cell_data_dict["markers"]["triangle"], [-1, 0, 1]
    )
    assert not city.point_data
    assert json.loads((tmp_path / "city.coordinates.json").read_text()) == frame
    np.testing.assert_array_equal(native.vertices, vertices)
    np.testing.assert_array_equal(native.boundary_faces, faces)
    np.testing.assert_array_equal(native.boundary_markers, markers)
    assert native.transform.srs == "EPSG:3006"


def test_untagged_run_removes_stale_context_and_preserves_other_files(tmp_path):
    native = BoxMesh(1, 1, 1)
    path = _export_city_surface(tmp_path / "solution.vtu", native_mesh=native, frame={})
    (tmp_path / "keep.txt").write_text("unrelated")
    native.boundary_markers = {}
    assert _export_city_surface(tmp_path / "solution.vtu", native_mesh=native) is None
    assert not path.exists()
    assert not path.with_suffix(".coordinates.json").exists()
    assert (tmp_path / "keep.txt").read_text() == "unrelated"


def test_explicit_city_solution_filename_is_not_overwritten_by_context(tmp_path):
    solution = tmp_path / "city.vtu"
    solution.write_bytes(b"solution fields")
    assert (
        _export_city_surface(solution, native_mesh=BoxMesh(1, 1, 1))
        == tmp_path / "city_surface.vtu"
    )
    assert solution.read_bytes() == b"solution fields"
    assert "triangle" in meshio.read(tmp_path / "city_surface.vtu").cells_dict


@pytest.mark.fenics
@pytest.mark.simulation
@pytest.mark.parametrize(
    "equation", ["heat", "navierstokes", "stokes", "reconstruction"]
)
@pytest.mark.parametrize("input_kind", ["native", "file", "backend"])
def test_city_surface_accompanies_real_solution_and_repeated_exports(
    equation, input_kind, tmp_path
):
    pytest.importorskip("dolfinx")
    from petsc4py import PETSc

    if not np.issubdtype(PETSc.ScalarType, np.floating):
        pytest.skip("Requires real PETSc")
    if equation == "reconstruction" and input_kind == "backend":
        pytest.skip(
            "An untagged reconstruction backend mesh has no city boundary information"
        )
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
    from dtcc_sim.fenics import _mesh_from_dtcc

    native = BoxMesh(3, 3, 3)
    offset = np.array([319895.0, 6398909.0, 12.5])
    native.vertices += offset
    original = native.vertices.copy()
    native.transform.srs = "EPSG:3006"
    if input_kind == "native":
        inputs = dict(mesh=native)
    elif input_kind == "file":
        path = tmp_path / "input.xdmf"
        native.save(path)
        inputs = dict(mesh_path=path, mesh_crs="EPSG:3006")
    else:
        mesh, markers = _mesh_from_dtcc(native)
        inputs = dict(mesh=mesh, markers=markers, mesh_crs="EPSG:3006")
    output = tmp_path / "result"
    if equation == "heat":
        solver = HeatSolver(
            **inputs,
            output_dir=output,
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
    elif equation == "reconstruction":
        solver = SmoothReconstructionSimulator(
            **inputs,
            output_dir=output,
            point_coords=offset + np.array([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]]),
            point_values=np.array([7.0, 7.0]),
            params=SmoothReconstructionParameters(background_value=7.0),
        )
    else:

        def inlet(x):
            z = x[2] - offset[2]
            return np.array([z * (2 - z), np.zeros(x.shape[1]), np.zeros(x.shape[1])])

        solver_type, parameter_type = (
            (NavierStokesSolver, NavierStokesParameters)
            if equation == "navierstokes"
            else (StokesSolver, StokesParameters)
        )
        solver = solver_type(
            **inputs,
            output_dir=output,
            inlet_expression=inlet,
            params=parameter_type(nu=0.1, wind_speed=1.0, side_top_boundary="slip"),
        )
    for run in range(2):
        solver.simulate()
        city = meshio.read(output / "city.vtu")
        assert len(city.cells_dict["triangle"]) > 0
        values = city.cell_data_dict["markers"]["triangle"]
        assert np.all((values == -1) | (values >= 0))
        np.testing.assert_array_equal(city.points.min(axis=0), offset)
        np.testing.assert_array_equal(city.points.max(axis=0), offset + [1.0, 1.0, 0.0])
        frame = json.loads((output / "city.coordinates.json").read_text())
        assert frame == solver.diagnostics["coordinate_frame"]
        assert frame["input_crs"] == "EPSG:3006"
        assert ["export", "city_surface"] in [
            row["path"] for row in solver.last_timings["stages"]
        ]
        np.testing.assert_array_equal(native.vertices, original)
        if run == 0:
            (output / "city.vtu").write_bytes(b"previous output")
            (output / "keep.txt").write_text("unrelated")
        else:
            assert (output / "keep.txt").read_text() == "unrelated"
