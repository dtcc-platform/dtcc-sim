"""Shallow-water API contracts and small real-worker numerical checks."""

from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dtcc_core import RectangleMesh
from dtcc_sim import ShallowWaterParameters, CeciliaSolver
from dtcc_sim.shallow_water import ShallowWaterResult
from dtcc_sim.shallow_water import solver as api


@pytest.fixture
def mesh():
    mesh = RectangleMesh(4, 4)
    mesh.markers = np.full(len(mesh.faces), -1, dtype=np.int32)
    return mesh


@pytest.mark.parametrize("settings", [{"gravity": 0}, {"final_time": float("nan")},
                                     {"n_outputs": 0}, {"sigma": -1}])
def test_invalid_parameters(settings):
    with pytest.raises(ValueError):
        ShallowWaterParameters(**settings)


def test_unavailable_worker_and_invalid_bed_leave_inputs_unchanged(mesh, tmp_path):
    original = mesh.vertices.copy()
    with pytest.raises(ValueError, match="real-PETSc"):
        CeciliaSolver(mesh=mesh, output_dir=tmp_path / "missing", solver_python="/no/such/python").run()
    assert not (tmp_path / "missing").exists()
    mesh.vertices[:, 2] = mesh.vertices[:, 0]
    with pytest.raises(ValueError, match="bed_from_mesh"):
        CeciliaSolver(mesh=mesh, output_dir=tmp_path / "bed").run()
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
    solver = CeciliaSolver(mesh=mesh, params=params, output_dir=tmp_path)
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
    solver = CeciliaSolver(mesh=mesh, params=params, output_dir=tmp_path)
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
    first = CeciliaSolver(mesh=mesh, params=params, output_dir=tmp_path / "local")
    first.simulate()
    mesh.vertices[:, :2] += [319995., 6399009.]
    mesh.transform.srs = "EPSG:3006"
    original = mesh.vertices.copy()
    second = CeciliaSolver(mesh=mesh, params=replace(params, x_center=319995.5, y_center=6399009.5), output_dir=tmp_path / "map")
    result = second.simulate()
    with np.load(first.result.case_dir / "solver/state.npz") as a, np.load(second.result.case_dir / "solver/state.npz") as b:
        np.testing.assert_array_equal(a["state"], b["state"])
    np.testing.assert_array_equal(result.vertices, original)
    assert result.transform.srs == "EPSG:3006"
    assert second.diagnostics["coordinate_frame"]["input_crs"] == "EPSG:3006"
    result.info(); axis = result.plot(field="water_depth", show=False)
    assert axis is not None
    plt.close("all")


@real_worker
@pytest.mark.simulation
def test_real_worker_preserves_sloping_lake(mesh, tmp_path):
    mesh.vertices[:, 2] = mesh.vertices[:, :2] @ np.array([.1, .03])
    solver = CeciliaSolver(mesh=mesh, output_dir=tmp_path,
        params=ShallowWaterParameters(bed_from_mesh=True, water_surface=1., amplitude=0., final_time=.02, n_outputs=1))
    solver.simulate()
    assert solver.diagnostics["max_state_change"] < 1e-10
