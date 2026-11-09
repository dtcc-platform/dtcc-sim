"""Argument-free Core plotting and repeatable real shallow-water demos."""

import os
from pathlib import Path
import runpy

import numpy as np
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
        maps = [ax for number in plt.get_fignums() for ax in plt.figure(number).axes
                if ax.get_gid() == "face_field_map"]
        assert len(maps) == 1 and maps[0].name == "rectilinear"
        assert any(c.get_gid() == "face_field" for c in maps[0].collections)
        if name == "dataset_urban_flood.py":
            assert (state["center_x"], state["center_y"], state["L"]) == (319970., 6397910., 250.)
            assert state["bounds"] == (319845., 6397785., 320095., 6398035.)
            assert result.transform.srs == "EPSG:3006"
            assert result.simulation_diagnostics["time"] == 20.
            assert result.simulation_diagnostics["rejected_attempts"] == 0
            from dtcc_sim.shallow_water import ShallowWaterResult
            transient = ShallowWaterResult.load(directory)
            np.testing.assert_allclose(transient.snapshot_times, [0., 5., 10., 15., 20.],
                                       rtol=0, atol=1e-12)
            initial = {field.name: field.values for field in transient.to_dtcc(0).fields}
            final = {field.name: field.values for field in result.fields}
            ground = result.markers < 0
            assert np.all(initial["water_depth"][~ground] == 0.), "The pulse must start outside buildings"
            newly_wet = ground & (initial["water_depth"][:, 0] == 0.) & (final["water_depth"][:, 0] > 1e-4)
            assert newly_wet.any(), "The pulse must propagate onto initially dry ground"
            owners = {}
            beside_building = np.zeros(len(ground), dtype=bool)
            for cell, face in enumerate(result.faces):
                for i, j in ((0, 1), (1, 2), (2, 0)):
                    edge = tuple(sorted((int(face[i]), int(face[j]))))
                    if edge in owners:
                        other = owners[edge]
                        if ground[cell] != ground[other]:
                            beside_building[cell if ground[cell] else other] = True
                    else:
                        owners[edge] = cell
            assert (newly_wet & beside_building).any(), "The pulse must reach new ground beside buildings"
            assert np.max(final["water_depth"][~ground]) < 1e-12
            assert np.max(abs(final["water_momentum"][ground])) > 1e-6
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


def test_dataset_demo_plots_returned_native_mesh(monkeypatch, tmp_path):
    """Check the demo's API wiring independently of live city acquisition."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from dtcc_core import RectangleMesh, datasets
    from dtcc_core.model import Field, Mesh

    native = RectangleMesh(1, 1)
    native.markers = np.array([-1, 0], dtype=np.int32)
    native.fields = [Field(name="water_depth", unit="m", association="face", dim=1,
                           values=np.array([[.2], [50.]]))]
    calls = []
    def request(**kwargs):
        calls.append(("request", kwargs))
        return native
    original_plot = Mesh.plot
    def plot(self, **kwargs):
        calls.append(("plot", self))
        assert kwargs["projection"] == "xy" and kwargs["field"] == "water_depth"
        np.testing.assert_array_equal(kwargs["context_mask"], [False, True])
        return original_plot(self, show=False, **kwargs)
    monkeypatch.setattr(datasets, "urban_flood_simulation", request)
    monkeypatch.setattr(Mesh, "info", lambda self: calls.append(("info", self)))
    monkeypatch.setattr(Mesh, "plot", plot)
    monkeypatch.chdir(tmp_path)
    state = runpy.run_path(str(DEMO_DIR / "dataset_urban_flood.py"))
    assert state["flood"] is native
    assert [name for name, _ in calls] == ["request", "info", "plot"]
    assert calls[1][1] is native and calls[2][1] is native
    settings = calls[0][1]
    assert settings["bounds"] == (319845., 6397785., 320095., 6398035.)
    assert (settings["h"], settings["min_building_detail"], settings["strict_live"]) == (10., .5, True)
    parameters = settings["parameters"]
    assert (parameters.final_time, parameters.cfl, parameters.n_outputs) == (20., .08, 4)
    assert (parameters.x_center, parameters.y_center) == (320005., 6397890.)
    plt.close("all")
