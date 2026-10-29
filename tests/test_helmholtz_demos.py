"""Keep the two wave demos linear, runnable and safe to repeat."""

import ast
import os
from pathlib import Path
import runpy

import pytest

from dtcc_sim.output import _output_directory, _solution_path

DEMO_DIR = Path(__file__).resolve().parents[1] / "demos"


def test_wave_demo_inventory_and_structure():
    wave_demos = {
        path.name for path in DEMO_DIR.glob("*.py")
        if "helmholtz" in path.name or "noise" in path.name
    }
    assert wave_demos == {"helmholtz_box.py", "dataset_urban_noise.py"}
    for name in wave_demos:
        source = (DEMO_DIR / name).read_text()
        assert len(source.splitlines()) <= 40
        tree = ast.parse(source)
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in ast.walk(tree)
        )
        assert "argparse" not in source and "__main__" not in source


def test_output_directory_is_internal_and_explicit(tmp_path):
    directory = _output_directory(tmp_path / "nested" / "results")
    assert not directory.exists()
    path = Path(_solution_path(directory, None, "solution.xdmf"))
    assert path.parent == directory and directory.is_dir()
    path.write_text("old result")
    assert _solution_path(directory, None, "solution.xdmf") == str(path)
    assert _solution_path(None, None, "solution.xdmf") is None
    with pytest.raises(ValueError, match="nonempty"):
        _output_directory("")
    with pytest.raises(ValueError, match="not both"):
        _solution_path(directory, "other.xdmf", "solution.xdmf")
    with pytest.raises(NotADirectoryError):
        _output_directory(path)


@pytest.fixture
def demo_runtime(monkeypatch, tmp_path):
    pytest.importorskip("dolfinx")
    from petsc4py import PETSc
    import numpy as np
    import matplotlib

    if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
        pytest.skip("Wave demos require complex PETSc")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    plt.close("all")


def _check_core_plot(mesh):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Path3DCollection
    import numpy as np

    ax = plt.gcf().axes[0]
    samples = [artist for artist in ax.collections if isinstance(artist, Path3DCollection)]
    assert len(samples) == 1
    np.testing.assert_allclose(
        samples[0].get_array(),
        next(field.values for field in mesh.fields if field.name == "helmholtz_amplitude").ravel()
    )
    plt.close(ax.figure)


@pytest.fixture
def solver_logs(caplog):
    from dtcc_core.common import get_python_logger

    logger = get_python_logger("dtcc-sim")
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)


def test_box_demo_runs_twice_without_arguments(demo_runtime, monkeypatch, solver_logs):
    caplog = solver_logs
    import sys

    monkeypatch.setattr(sys, "argv", [str(DEMO_DIR / "helmholtz_box.py")])
    runs = []
    for _ in range(2):
        state = runpy.run_path(str(DEMO_DIR / "helmholtz_box.py"))
        directory = Path(state["output_dir"])
        runs.append(directory)
        assert state["simulation"].result.converged
        assert state["simulation"].result.relative_residual < 1e-7
        assert (directory / "solution.vtu").is_file()
        _check_core_plot(state["result"])
        output = "\n".join(record.getMessage() for record in caplog.records)
        caplog.clear()
        assert "Preparing two-level preconditioner" in output
        assert "GMRES iteration 1; relative preconditioned residual=" in output
        assert "Helmholtz solve converged after" in output
        assert "true relative residual=" in output
        from dolfinx import log
        assert log.get_log_level() == log.LogLevel.WARNING
        if len(runs) == 1:
            (runs[-1] / "keep.txt").write_text("unrelated")
        else:
            assert (runs[-1] / "keep.txt").read_text() == "unrelated"
    assert (runs[-1] / "keep.txt").read_text() == "unrelated"
    assert runs[0] == runs[1]
    assert all((run / "diagnostics.json").is_file() for run in runs)


@pytest.mark.skipif(os.environ.get("DTCC_RUN_DEMOS") != "1", reason="Live city data is opt-in")
def test_noise_demo_runs_twice_and_reuses_mesh(demo_runtime, monkeypatch):
    import json
    import sys
    import numpy as np

    monkeypatch.setattr(sys, "argv", [str(DEMO_DIR / "dataset_urban_noise.py")])
    runs = []
    for index in range(2):
        state = runpy.run_path(str(DEMO_DIR / "dataset_urban_noise.py"))
        run = Path(state["output_dir"])
        runs.append(run)
        assert state["noise"].dataset_context.request.dataset_name == "urban_noise_simulation"
        assert (run / "solution.vtu").is_file()
        import meshio
        city = meshio.read(run / "city.vtu")
        markers = city.cell_data_dict["markers"]["triangle"]
        assert len(markers) and np.all((markers == -1) | (markers >= 0))
        assert np.any(markers >= 0)
        assert set(map(tuple, city.points)) <= set(map(tuple, state["noise"].vertices))
        _check_core_plot(state["noise"])
        provenance = json.loads((run / "mesh_provenance.json").read_text())
        assert provenance["reused"] is bool(index)
        diagnostics = json.loads((run / "diagnostics.json").read_text())
        assert diagnostics["converged"] and diagnostics["relative_residual"] < 1e-5
        if len(runs) == 1:
            (runs[-1] / "keep.txt").write_text("unrelated")
        else:
            assert (runs[-1] / "keep.txt").read_text() == "unrelated"
    assert (runs[-1] / "keep.txt").read_text() == "unrelated"
    assert runs[0] == runs[1]


def test_solution_exports_protect_input_hdf5_companions(tmp_path):
    mesh = tmp_path / "input.xdmf"
    mesh.write_text('<Xdmf><DataItem Format="HDF">solution.h5:/Mesh</DataItem></Xdmf>')
    source = tmp_path / "solution.h5"
    source.write_bytes(b"input mesh data")
    with pytest.raises(ValueError, match="input mesh"):
        _solution_path(tmp_path, None, "solution.xdmf", input_mesh=mesh)
    assert source.read_bytes() == b"input mesh data"

    mesh.write_text('<Xdmf><DataItem Format="HDF">solution_velocity.h5:/Mesh</DataItem></Xdmf>')
    with pytest.raises(ValueError, match="input mesh"):
        _solution_path(tmp_path, None, "solution.xdmf", input_mesh=mesh,
                       stems=("_velocity", "_pressure"))
