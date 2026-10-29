"""Georeferenced flow results, exports and input preservation."""

import numpy as np
import json
import pytest

pytest.importorskip("dolfinx")
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc

from dtcc_core import BoxMesh
from dtcc_core.common import get_python_logger
from dtcc_sim import (
    NavierStokesParameters, NavierStokesSolver, StokesParameters, StokesSolver,
)
from dtcc_sim.fenics import _mesh_from_dtcc
from dtcc_sim._flow import solver as flow

pytestmark = [pytest.mark.fenics, pytest.mark.simulation]


@pytest.fixture(autouse=True)
def real_runtime():
    if not np.issubdtype(PETSc.ScalarType, np.floating):
        pytest.skip("Flow requires real PETSc")


def georeferenced_box():
    native = BoxMesh(3, 3, 3)
    native.vertices *= [2., 1., 1.]
    native.vertices += [319895.962899, 6398909.716755, 12.5]
    return native


def parabolic(x):
    z = x[2] - 12.5
    return np.array([z*(2-z), np.zeros(x.shape[1]), np.zeros(x.shape[1])])


@pytest.mark.parametrize("native_input", [False, True])
@pytest.mark.parametrize("solver_type,parameter_type", [
    (NavierStokesSolver, NavierStokesParameters), (StokesSolver, StokesParameters),
])
def test_flow_preserves_world_coordinates_in_results_and_xdmf(
    solver_type, parameter_type, native_input, tmp_path, caplog,
):
    native = georeferenced_box()
    expected = native.vertices.copy()
    if native_input:
        inputs = dict(mesh=native)
    else:
        mesh, markers = _mesh_from_dtcc(native)
        original = mesh.geometry.x.copy()
        inputs = dict(mesh=mesh, markers=markers)
    params = parameter_type(nu=.1, wind_speed=1., side_top_boundary="slip")
    solver = solver_type(
        **inputs, params=params, inlet_expression=parabolic, output_dir=tmp_path,
    )
    logger = get_python_logger("dtcc-sim")
    logger.addHandler(caplog.handler)
    try:
        result = solver.simulate()
    finally:
        logger.removeHandler(caplog.handler)
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "Flow:" not in messages
    assert "Initializing stationary flow with a Stokes solve" in messages
    record = json.loads((tmp_path/"solution.coordinates.json").read_text())
    assert record == solver.diagnostics["coordinate_frame"]
    np.testing.assert_array_equal(record["local_to_input_translation"], expected.min(axis=0))
    np.testing.assert_array_equal(native.vertices, expected)
    if native_input:
        np.testing.assert_array_equal(result.vertices, expected)
        velocity = next(f.values for f in result.fields if f.name == "velocity")
        z = expected[:, 2] - expected[:, 2].min()
        np.testing.assert_allclose(velocity[:, 0], z*(2-z), atol=1e-7)
    else:
        np.testing.assert_array_equal(mesh.geometry.x, original)
        assert all(field.function_space.mesh is mesh for field in result)
    for name in ("velocity", "pressure"):
        with XDMFFile(MPI.COMM_WORLD, str(tmp_path/f"solution_{name}.xdmf"), "r") as file:
            exported = file.read_mesh()
        np.testing.assert_allclose(exported.geometry.x.min(axis=0), expected.min(axis=0), atol=1e-9, rtol=0)
        np.testing.assert_allclose(exported.geometry.x.max(axis=0), expected.max(axis=0), atol=1e-9, rtol=0)


@pytest.mark.parametrize("refine", [False, True])
def test_flow_restores_input_and_refined_coordinates_on_failure(monkeypatch, refine):
    native = georeferenced_box()
    mesh, tags = _mesh_from_dtcc(native)
    original = mesh.geometry.x.copy()
    solver = NavierStokesSolver(mesh=mesh, markers=tags)

    def fail(started):
        np.testing.assert_allclose(mesh.geometry.x.min(axis=0), 0., atol=1e-9)
        if refine:
            solver.mesh, solver.markers = flow._refine_pressure_mesh(
                mesh, tags, mesh.geometry.x[:1],
            )
        raise RuntimeError("Deliberate solve failure")

    monkeypatch.setattr(solver, "_solve_stationary", fail)
    with pytest.raises(RuntimeError, match="Deliberate solve failure"):
        solver.simulate()
    np.testing.assert_array_equal(mesh.geometry.x, original)
    np.testing.assert_allclose(solver.mesh.geometry.x.min(axis=0), original.min(axis=0), atol=1e-9, rtol=0)
    np.testing.assert_allclose(solver.mesh.geometry.x.max(axis=0), original.max(axis=0), atol=1e-9, rtol=0)


def test_refined_flow_results_use_world_coordinates(monkeypatch, tmp_path):
    native = georeferenced_box()
    mesh, tags = _mesh_from_dtcc(native)
    original = mesh.geometry.x.copy()
    solver = StokesSolver(
        mesh=mesh, markers=tags,
        params=StokesParameters(nu=.1, wind_speed=1., side_top_boundary="slip"),
        inlet_expression=parabolic, output_dir=tmp_path,
    )
    solve = solver._solve_stationary

    def refined_solve(started):
        solver.mesh, solver.markers = flow._refine_pressure_mesh(
            solver.mesh, solver.markers, solver.mesh.geometry.x[:1],
        )
        return solve(started)

    monkeypatch.setattr(solver, "_solve_stationary", refined_solve)
    result = solver.simulate()
    np.testing.assert_array_equal(mesh.geometry.x, original)
    assert all(field.function_space.mesh is solver.mesh for field in result)
    np.testing.assert_allclose(solver.mesh.geometry.x.min(axis=0), original.min(axis=0), atol=1e-9, rtol=0)
    with XDMFFile(MPI.COMM_WORLD, str(tmp_path/"solution_velocity.xdmf"), "r") as file:
        exported = file.read_mesh()
    np.testing.assert_allclose(exported.geometry.x.min(axis=0), original.min(axis=0), atol=1e-9, rtol=0)


def test_distributed_flow_coordinates_and_export(tmp_path):
    import dolfinx.mesh as dm

    mesh = dm.create_box(MPI.COMM_WORLD, [[0., 0., 0.], [2., 1., 1.]], [6, 3, 3])
    mesh.topology.create_entities(2)
    mesh.topology.create_connectivity(2, 3)
    faces = dm.exterior_facet_indices(mesh.topology)
    midpoints = dm.compute_midpoints(mesh, 2, faces)
    labels = np.full(len(faces), -1, dtype=np.int32)
    for axis, value, marker in [(0, 0, -3), (0, 2, -4), (1, 0, -5), (1, 1, -6)]:
        labels[np.isclose(midpoints[:, axis], value)] = marker
    order = np.argsort(faces)
    tags = dm.meshtags(mesh, 2, faces[order], labels[order])
    origin = np.array([319895.962899, 6398909.716755, 12.5])
    mesh.geometry.x[:] += origin
    original = mesh.geometry.x.copy()
    directory = MPI.COMM_WORLD.bcast(str(tmp_path), root=0)

    def inlet(x):
        z = x[2] - 12.5
        return np.array([4*z*(1-z), np.zeros(x.shape[1]), np.zeros(x.shape[1])])

    solver = StokesSolver(
        mesh=mesh, markers=tags,
        params=StokesParameters(nu=.1, wind_speed=1., side_top_boundary="slip"),
        inlet_expression=inlet, output_dir=directory,
    )
    solver.simulate()
    np.testing.assert_array_equal(mesh.geometry.x, original)
    from pathlib import Path
    record = json.loads((Path(directory)/"solution.coordinates.json").read_text())
    np.testing.assert_array_equal(record["local_to_input_translation"], origin)
    with XDMFFile(MPI.COMM_WORLD, f"{directory}/solution_velocity.xdmf", "r") as file:
        exported = file.read_mesh()
    from dtcc_sim.fenics import bounds

    extent = np.array(bounds(exported))
    np.testing.assert_allclose(extent[:3], origin, atol=1e-9, rtol=0)
    np.testing.assert_allclose(extent[3:], origin + [2., 1., 1.], atol=1e-9, rtol=0)


def test_heat_translation_invariance_and_local_postprocessing(monkeypatch, tmp_path):
    from dtcc_sim import HeatParameters, HeatSolver

    results = []
    for index, offset in enumerate(([0., 0., 0.], [319895., 6398909., 12.5])):
        native = BoxMesh(2, 2, 2)
        native.vertices += offset
        native.transform.srs = "EPSG:3006"
        original = native.vertices.copy()
        solver = HeatSolver(
            mesh=native, params=HeatParameters(sigma=1., T_ambient=10.),
            output_dir=tmp_path/str(index),
        )
        attach = solver._attach_temperature_field

        def attach_locally():
            np.testing.assert_array_equal(solver.mesh.geometry.x.min(axis=0), 0.)
            return attach()

        monkeypatch.setattr(solver, "_attach_temperature_field", attach_locally)
        result = solver.simulate()
        np.testing.assert_array_equal(native.vertices, original)
        np.testing.assert_array_equal(result.vertices, original)
        record = solver.diagnostics["coordinate_frame"]
        assert record["local_to_input_translation"] == list(offset)
        assert record["input_crs"] == "EPSG:3006"
        assert json.loads((tmp_path/str(index)/"solution.coordinates.json").read_text()) == record
        with XDMFFile(MPI.COMM_WORLD, str(tmp_path/str(index)/"solution.xdmf"), "r") as file:
            exported = file.read_mesh()
        np.testing.assert_array_equal(exported.geometry.x.min(axis=0), offset)
        results.append(next(f.values for f in result.fields if f.name == "temperature"))
    np.testing.assert_allclose(results[0], results[1], atol=1e-10, rtol=0)


def test_reconstruction_transforms_observations_and_exports_world_coordinates(monkeypatch, tmp_path):
    import dolfinx.mesh as dm
    from dtcc_sim import SmoothReconstructionParameters, SmoothReconstructionSimulator

    results = []
    for index, offset in enumerate(([0., 0., 0.], [319895., 6398909., 12.5])):
        mesh = dm.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
        mesh.geometry.x[:] += offset
        original = mesh.geometry.x.copy()
        observations = np.array([[.25, .25, .25], [.75, .75, .75]]) + offset
        snapshot = observations.copy()
        solver = SmoothReconstructionSimulator(
            mesh=mesh, point_coords=observations, point_values=np.array([10., 20.]),
            params=SmoothReconstructionParameters(), output_dir=tmp_path/str(index),
        )
        locate = solver._locate_points

        def locate_locally(numerical_mesh, points):
            np.testing.assert_array_equal(numerical_mesh.geometry.x.min(axis=0), 0.)
            np.testing.assert_allclose(points, [[.25, .25, .25], [.75, .75, .75]], atol=1e-12)
            return locate(numerical_mesh, points)

        monkeypatch.setattr(solver, "_locate_points", locate_locally)
        result = solver.simulate()
        np.testing.assert_array_equal(mesh.geometry.x, original)
        np.testing.assert_array_equal(observations, snapshot)
        assert solver.diagnostics["coordinate_frame"]["local_to_input_translation"] == list(offset)
        assert json.loads((tmp_path/str(index)/"solution.coordinates.json").read_text()) == solver.diagnostics["coordinate_frame"]
        with XDMFFile(MPI.COMM_WORLD, str(tmp_path/str(index)/"solution.xdmf"), "r") as file:
            exported = file.read_mesh()
        np.testing.assert_array_equal(exported.geometry.x.min(axis=0), offset)
        results.append(result.x.array.copy())
    np.testing.assert_allclose(results[0], results[1], atol=1e-10, rtol=0)


@pytest.mark.parametrize("kind", ["heat", "reconstruction"])
def test_scalar_solver_restores_mesh_on_failure(monkeypatch, kind):
    from dtcc_sim import HeatSolver, SmoothReconstructionSimulator

    native = georeferenced_box()
    mesh, tags = _mesh_from_dtcc(native)
    original = mesh.geometry.x.copy()
    if kind == "heat":
        solver = HeatSolver(mesh=mesh, markers=tags)
        method = "_solve_local_heat"
    else:
        solver = SmoothReconstructionSimulator(
            mesh=mesh, point_coords=np.array([[319896., 6398910., 13.]]),
            point_values=np.array([10.]),
        )
        method = "_solve_local_reconstruction"

    def fail(*args):
        np.testing.assert_array_equal(mesh.geometry.x.min(axis=0), 0.)
        raise RuntimeError("Deliberate numerical failure")

    monkeypatch.setattr(solver, method, fail)
    with pytest.raises(RuntimeError, match="Deliberate numerical failure"):
        solver.simulate()
    np.testing.assert_array_equal(mesh.geometry.x, original)


def test_private_mesh_returns_to_local_coordinates_after_failed_export():
    from dtcc_sim.coordinates import localize_mesh, input_mesh_coordinates

    mesh, _ = _mesh_from_dtcc(georeferenced_box())
    original = mesh.geometry.x.copy()
    localize_mesh(mesh, crs="EPSG:3006")
    local = mesh.geometry.x.copy()
    with pytest.raises(RuntimeError, match="Deliberate export failure"):
        with input_mesh_coordinates(mesh):
            np.testing.assert_array_equal(mesh.geometry.x, original)
            raise RuntimeError("Deliberate export failure")
    np.testing.assert_array_equal(mesh.geometry.x, local)
