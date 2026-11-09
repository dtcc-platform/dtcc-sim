"""Shallow-water API contracts and small real-worker numerical checks."""

from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dtcc_core import RectangleMesh
from dtcc_sim import ShallowWaterParameters, ShallowWaterSolver
from dtcc_sim.shallow_water import ShallowWaterResult
from dtcc_sim.shallow_water import solver as api


@pytest.fixture
def mesh():
    mesh = RectangleMesh(4, 4)
    mesh.markers = np.full(len(mesh.faces), -1, dtype=np.int32)
    return mesh


@pytest.mark.parametrize("settings", [{"gravity": 0}, {"final_time": float("nan")},
                                     {"n_outputs": 0}, {"sigma": -1},
                                     {"max_steps": 0}, {"max_steps": True},
                                     {"max_steps": 1.5}])
def test_invalid_parameters(settings):
    with pytest.raises(ValueError):
        ShallowWaterParameters(**settings)


def test_unavailable_worker_and_invalid_bed_leave_inputs_unchanged(mesh, tmp_path):
    original = mesh.vertices.copy()
    with pytest.raises(ValueError, match="real-PETSc"):
        ShallowWaterSolver(mesh=mesh, output_dir=tmp_path / "missing", solver_python="/no/such/python").run()
    assert not (tmp_path / "missing").exists()
    mesh.vertices[:, 2] = mesh.vertices[:, 0]
    with pytest.raises(ValueError, match="bed_from_mesh"):
        ShallowWaterSolver(mesh=mesh, output_dir=tmp_path / "bed").run()
    np.testing.assert_array_equal(mesh.vertices[:, :2], original[:, :2])
    assert not (tmp_path / "bed").exists()


def test_worker_isolation_failure_and_source_frame(mesh, tmp_path, monkeypatch):
    mesh.vertices[:, :2] += [319995., 6399009.]
    original = mesh.vertices.copy()
    monkeypatch.setenv("CECILIA_EQUILIBRIUM_PRESSURE", "0")
    params = ShallowWaterParameters(x_center=319995.5, y_center=6399009.5)

    def failure(command, **kwargs):
        assert kwargs["env"]["CECILIA_EQUILIBRIUM_PRESSURE"] == "1"
        assert kwargs["env"]["PYTHONPATH"] == str(api._PACKAGE / "_solver")
        case = Path(command[-1]).parent
        config = json.loads((case / "parameters.json").read_text())
        assert config["x_center"] == config["y_center"] == .5
        with np.load(case / "input_mesh.npz") as supplied:
            assert supplied["vertices"][:, :2].min() == 0.
            assert supplied["vertices"][:, :2].max() == 1.
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(api.subprocess, "run", failure)
    solver = ShallowWaterSolver(mesh=mesh, params=params, output_dir=tmp_path)
    with pytest.raises(RuntimeError, match="exited with 7"):
        solver.run()
    case, = (tmp_path / ".shallow_water_runs").iterdir()
    assert json.loads((case / "run.json").read_text())["status"] == "failed"
    with pytest.raises(ValueError, match="completed"):
        ShallowWaterResult.load(case)
    np.testing.assert_array_equal(mesh.vertices, original)
    assert os.environ["CECILIA_EQUILIBRIUM_PRESSURE"] == "0"


real_worker = pytest.mark.skipif(not os.environ.get("DTCC_SWE_PYTHON"), reason="Set DTCC_SWE_PYTHON for real worker tests")


@real_worker
@pytest.mark.simulation
def test_real_worker_repeats_preserves_files_and_conserves_water(mesh, tmp_path):
    from dtcc_core.io import load_mesh

    original = mesh.vertices.copy(); (tmp_path / "notes.txt").write_text("keep")
    params = ShallowWaterParameters(h0=0., compact_pulse=True, amplitude=.2,
                                    sigma=.2, final_time=.02, n_outputs=2)
    solver = ShallowWaterSolver(mesh=mesh, params=params, output_dir=tmp_path)
    for _ in range(2):
        result = solver.simulate()
        assert solver.diagnostics["passed"] and solver.diagnostics["rejected_attempts"] == 0
        assert solver.diagnostics["relative_mass_error"] < 1e-12
        np.testing.assert_array_equal(result.vertices, original)
        np.testing.assert_array_equal(mesh.vertices, original)
        restored = load_mesh(tmp_path / "solution.dtcc")
        for field in result.fields:
            np.testing.assert_array_equal(next(f.values for f in restored.fields if f.name == field.name), field.values)
        assert (tmp_path / "notes.txt").read_text() == "keep"
        assert (tmp_path / "timings.json").is_file()
        assert ShallowWaterResult.load(tmp_path).snapshot_times[-1] == params.final_time
    destination = tmp_path / "series"
    entries = solver.result.export_series(destination)
    assert len(entries) == 3
    (destination / "notes.txt").write_text("keep")
    solver.parameters = replace(params, n_outputs=1)
    solver.simulate(); solver.result.export_series(destination)
    assert not (destination / "water_00002.dtcc").exists()
    assert (destination / "notes.txt").read_text() == "keep"


@real_worker
@pytest.mark.simulation
def test_real_worker_translation_and_core_plot(mesh, tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    params = ShallowWaterParameters(h0=0., amplitude=.2, sigma=.2, compact_pulse=True,
                                    x_center=.5, y_center=.5, final_time=.02, n_outputs=1)
    first = ShallowWaterSolver(mesh=mesh, params=params, output_dir=tmp_path / "local")
    first.simulate()
    mesh.vertices[:, :2] += [319995., 6399009.]
    mesh.transform.srs = "EPSG:3006"
    original = mesh.vertices.copy()
    second = ShallowWaterSolver(mesh=mesh, params=replace(params, x_center=319995.5, y_center=6399009.5), output_dir=tmp_path / "map")
    result = second.simulate()
    with np.load(first.result.case_dir / "solver/state.npz") as a, np.load(second.result.case_dir / "solver/state.npz") as b:
        np.testing.assert_array_equal(a["state"], b["state"])
    np.testing.assert_array_equal(result.vertices, original)
    assert result.transform.srs == "EPSG:3006"
    assert second.diagnostics["coordinate_frame"]["input_crs"] == "EPSG:3006"
    result.info(); axis = result.plot(field="water_depth", show=False)
    assert axis is not None
    depth_map = second.result.plot(snapshot=0, show=False, vmax=.2)
    assert depth_map.name == "rectilinear" and "t = 0 s" in depth_map.get_title()
    assert next(c for c in depth_map.collections if c.get_gid() == "water_depth").norm.vmax == .2
    plt.close("all")


@real_worker
@pytest.mark.simulation
def test_real_worker_preserves_sloping_lake(mesh, tmp_path):
    mesh.vertices[:, 2] = mesh.vertices[:, :2] @ np.array([.1, .03])
    solver = ShallowWaterSolver(mesh=mesh, output_dir=tmp_path,
        params=ShallowWaterParameters(bed_from_mesh=True, water_surface=1., amplitude=0., final_time=.02, n_outputs=1))
    solver.simulate()
    assert solver.diagnostics["max_state_change"] < 1e-10


@real_worker
@pytest.mark.simulation
def test_scheduled_outputs_and_step_limit(mesh, tmp_path):
    params = ShallowWaterParameters(amplitude=0., final_time=1e-4,
                                   n_outputs=5, max_steps=2)
    solver = ShallowWaterSolver(mesh=mesh, params=params, output_dir=tmp_path / "limited")
    with pytest.raises(RuntimeError, match="Maximum step count exhausted.*max_steps=2"):
        solver.run()
    assert not (solver.output_dir / "shallow_water.json").exists()
    case, = (solver.output_dir / ".shallow_water_runs").iterdir()
    assert json.loads((case / "run.json").read_text())["status"] == "failed"

    solver = ShallowWaterSolver(mesh=mesh, params=replace(params, max_steps=None),
                                output_dir=tmp_path / "complete")
    result = solver.run()
    np.testing.assert_allclose(result.snapshot_times, np.linspace(0., 1e-4, 6),
                               rtol=0, atol=1e-15)
    assert result.diagnostics["passed"]


def test_runtime_rejects_incompatible_worker_before_preparation(tmp_path, monkeypatch):
    import sys
    from dtcc_sim.shallow_water._solver.cecilia_swe.runtime import validate_runtime
    from dtcc_sim.shallow_water._solver.cecilia_swe.simulator import ShallowWaterSimulator

    monkeypatch.setitem(sys.modules, "dolfinx", SimpleNamespace(__version__="0.11.0"))
    monkeypatch.setitem(sys.modules, "mpi4py", SimpleNamespace(MPI=SimpleNamespace(COMM_WORLD=SimpleNamespace(size=1))))
    monkeypatch.setitem(sys.modules, "petsc4py", SimpleNamespace(PETSc=SimpleNamespace(ScalarType=np.float64)))
    with pytest.raises(RuntimeError, match="Unsupported worker DOLFINx 0.11.0.*shallow-water-worker.yml"):
        ShallowWaterSimulator(output_dir=tmp_path / "worker").run()
    assert not (tmp_path / "worker").exists()
    monkeypatch.setitem(sys.modules, "dolfinx", SimpleNamespace(__version__="0.10.0"))
    monkeypatch.setitem(sys.modules, "petsc4py", SimpleNamespace(PETSc=SimpleNamespace(ScalarType=np.complex128)))
    with pytest.raises(RuntimeError, match="real Float64 PETSc"):
        validate_runtime()


@real_worker
@pytest.mark.simulation
def test_distant_dry_building_does_not_change_pulse(tmp_path):
    mesh = RectangleMesh(8, 8)
    mesh.markers = np.full(len(mesh.faces), -1, dtype=np.int32)
    centers = mesh.vertices[mesh.faces, :2].mean(axis=1)
    building_face = int(np.argmax(centers.sum(axis=1)))
    params = ShallowWaterParameters(h0=0., amplitude=.2, compact_pulse=True,
                                    x_center=.25, y_center=.25, sigma=.08,
                                    final_time=.01, n_outputs=1)
    states = []
    velocities = []
    for name in ("ground", "building"):
        if name == "building":
            mesh.markers[building_face] = 0
        solver = ShallowWaterSolver(mesh=mesh, params=params, output_dir=tmp_path / name)
        result = solver.run()
        assert result.diagnostics["passed"]
        assert result.diagnostics["rejected_attempts"] == 0
        assert result.diagnostics["relative_mass_error"] < 1e-12
        with np.load(result.case_dir / "solver/state.npz") as raw:
            cells = raw["cell_dofs"]
            coordinates = raw["coordinates"][cells, :2]
            state = np.stack([raw["state"][raw[key]][cells]
                              for key in ("h_dofs", "hu_dofs", "hv_dofs")], axis=-1)
            states.append(state)
            velocity = np.zeros_like(state[:, :, 1:])
            np.divide(state[:, :, 1:], np.maximum(state[:, :, :1], params.h_floor),
                      out=velocity)
            velocities.append(velocity)
            building = int(np.argmin(np.linalg.norm(coordinates.mean(axis=1)
                                                     - centers[building_face], axis=1)))
            assert np.all(raw["initial_state"][raw["h_dofs"]][cells[building]] == 0.)
        if name == "building":
            with np.load(result.case_dir / "solver/limiter-topology.npz") as topology:
                assert len(topology["wall_facets"]) > 0
                # No ground-cell stencil may include the enclosed building.
                ground = np.arange(len(cells)) != building
                assert not np.any(topology["neighbors"][ground] == building)
                assert np.all(topology["neighbors"][building] == building)
    near_pulse = np.all(coordinates.mean(axis=1) < .625, axis=1)
    assert near_pulse.any() and not near_pulse[building]
    np.testing.assert_allclose(states[1][near_pulse], states[0][near_pulse],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(velocities[1][near_pulse], velocities[0][near_pulse],
                               rtol=0, atol=1e-9)


@real_worker
@pytest.mark.simulation
def test_building_and_ground_conserve_water_with_active_pulse(tmp_path):
    mesh = RectangleMesh(8, 8)
    triangles = mesh.vertices[mesh.faces, :2]
    centers = triangles.mean(axis=1)
    building = np.all((centers > .375) & (centers < .625), axis=1)
    assert building.any() and (~building).any()
    mesh.markers = np.where(building, 0, -1).astype(np.int32)
    a = triangles[:, 1] - triangles[:, 0]
    b = triangles[:, 2] - triangles[:, 0]
    areas = abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2
    params = ShallowWaterParameters(h0=.1, amplitude=.2, compact_pulse=True,
                                    x_center=.25, y_center=.5, sigma=.15,
                                    final_time=.01, n_outputs=2)
    result = ShallowWaterSolver(mesh=mesh, params=params, output_dir=tmp_path).run()
    assert result.diagnostics["passed"] and result.diagnostics["rejected_attempts"] == 0
    volumes = []
    for index in range(len(result.snapshot_times)):
        fields = {field.name: field.values for field in result.to_dtcc(index).fields}
        h = fields["water_depth"][:, 0]
        assert np.all(h >= 0)
        volumes.append([areas[region] @ h[region] for region in (building, ~building)])
    volumes = np.array(volumes)
    assert np.all(volumes[0] > 0)
    np.testing.assert_allclose(volumes, np.broadcast_to(volumes[0], volumes.shape),
                               rtol=0, atol=1e-12)
    assert np.max(abs(fields["water_momentum"][~building])) > 1e-8
