"""Argument-free Core plotting and repeatable real shallow-water demos."""

import os
from pathlib import Path
import runpy

import pytest

DEMO_DIR = Path(__file__).resolve().parents[1] / "demos"


@pytest.fixture
def demo_runtime(monkeypatch, tmp_path):
    if not os.environ.get("DTCC_SWE_PYTHON"):
        pytest.skip("Set DTCC_SWE_PYTHON for real worker demos")
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    plt.close("all")


def _check_repeat(name, directory):
    import matplotlib.pyplot as plt
    for _ in range(2):
        state = runpy.run_path(str(DEMO_DIR / name))
        result = state.get("result", state.get("flood"))
        assert any(f.name == "water_depth" for f in result.fields)
        assert result.simulation_diagnostics["passed"]
        assert (directory / "solution.vtu").is_file()
        assert (directory / "solution.dtcc").is_file()
        assert (directory / "notes.txt").read_text() == "keep"
        assert plt.get_fignums()
        plt.close("all")


@pytest.mark.simulation
def test_offline_demo_runs_twice(demo_runtime):
    directory = demo_runtime / "output/shallow_water_pulse"
    directory.mkdir(parents=True); (directory / "notes.txt").write_text("keep")
    _check_repeat("shallow_water_pulse.py", directory)


@pytest.mark.expensive
@pytest.mark.skipif(os.environ.get("DTCC_RUN_DEMOS") != "1", reason="Set DTCC_RUN_DEMOS=1 for live city data")
def test_city_dataset_demo_runs_twice(demo_runtime):
    directory = demo_runtime / "output/dataset_urban_flood"
    directory.mkdir(parents=True); (directory / "notes.txt").write_text("keep")
    _check_repeat("dataset_urban_flood.py", directory)
