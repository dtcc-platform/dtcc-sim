"""Offline execution of stationary flow demonstrations."""

from pathlib import Path
import runpy
import numpy as np
import pytest

pytest.importorskip("dolfinx")
from petsc4py import PETSc
from dtcc_core.model import VolumeMesh

pytestmark = [pytest.mark.fenics, pytest.mark.simulation, pytest.mark.slow]


@pytest.mark.parametrize("name", ["channel_flow", "driven_cavity", "bluff_body"])
def test_offline_stationary_demo(name, monkeypatch, tmp_path):
    if not np.issubdtype(PETSc.ScalarType, np.floating):
        pytest.skip("Flow demos require real PETSc")
    source = Path(__file__).resolve().parents[1] / "demos" / f"{name}.py"
    previews = []
    def preview(self, **kwargs):
        assert kwargs["field"] == "speed"
        assert {field.name for field in self.fields} == {"velocity", "pressure", "speed"}
        previews.append(self)
    monkeypatch.setattr(VolumeMesh, "plot", preview)
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(source))
    assert previews
    assert previews[0].simulation_diagnostics["stop_reason"] == "converged"
    assert (tmp_path / "output" / name / "solution_velocity.xdmf").is_file()
    assert (tmp_path / "output" / name / "solution_pressure.h5").is_file()


@pytest.mark.expensive
@pytest.mark.parametrize("name", ["urban_wind_simulation", "urban_wind_stokes_simulation", "dataset_urban_wind", "dataset_urban_wind_stokes"])
def test_city_stationary_demo_on_frozen_mesh(name, monkeypatch, tmp_path):
    import copy
    import os
    from dtcc_core import datasets, io
    mesh_path = os.environ.get("DTCC_WIND_CITY_MESH")
    if not mesh_path:
        pytest.skip("Set DTCC_WIND_CITY_MESH to a tagged city XDMF mesh")
    if not np.issubdtype(PETSc.ScalarType, np.floating):
        pytest.skip("City wind demos require real PETSc")
    native = io.load_volume_mesh(mesh_path)
    prepared = native.repair_boundary_corners()
    def city_mesh(**kwargs):
        assert kwargs["max_mesh_size"] == 12.5
        assert kwargs["repair_boundary_corners"] is True
        bounds = kwargs["bounds"]
        width, height = (bounds.xmax-bounds.xmin, bounds.ymax-bounds.ymin) if hasattr(bounds, "xmin") else (bounds[2]-bounds[0], bounds[3]-bounds[1])
        assert width == 200 and height == 200
        return copy.deepcopy(prepared)
    city_mesh.ArgsModel = datasets.city_volume_mesh.ArgsModel
    city_mesh.validate = datasets.city_volume_mesh.validate
    monkeypatch.setattr(datasets, "city_volume_mesh", city_mesh)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.chdir(tmp_path)
    source = Path(__file__).resolve().parents[1] / "demos" / f"{name}.py"
    directory = tmp_path / "output" / name
    try:
        for index in range(2):
            state = runpy.run_path(str(source))
            result = state["result"]
            assert result.simulation_diagnostics["stop_reason"] == "converged"
            np.testing.assert_array_equal(result.vertices, prepared.vertices)
            np.testing.assert_array_equal(result.cells, prepared.cells)
            np.testing.assert_array_equal(result.vertices[:len(native.vertices)], native.vertices)
            assert result.simulation_diagnostics.get("mesh_refinements", 0) == 0
            assert result.find_boundary_corners().num_corners == 0
            if "solver" in state:
                assert state["solver"].params.mesh_refinement_levels == 0
            if name.startswith("dataset_"):
                context = result.dataset_context
                assert context.request.dataset_name == "urban_wind_simulation"
                assert context.request.parameters["equations"] == (
                    "stokes" if name.endswith("stokes") else "navier_stokes"
                )
            speed = next(field.values for field in result.fields if field.name == "speed")
            np.testing.assert_allclose(plt.gcf().axes[0].collections[0].get_array(), speed.ravel())
            assert (directory / "solution_velocity.xdmf").is_file()
            assert (directory / "solution_pressure.h5").is_file()
            assert (directory / "solution.coordinates.json").is_file()
            if index == 0:
                (directory / "keep.txt").write_text("unrelated")
            else:
                assert (directory / "keep.txt").read_text() == "unrelated"
            plt.close("all")
    finally:
        plt.close("all")
