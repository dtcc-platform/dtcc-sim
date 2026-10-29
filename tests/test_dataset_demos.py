"""Run city dataset and scalar solver examples against live upstream data."""

import os
from pathlib import Path
import runpy

import numpy as np
import pytest


pytestmark = [
    pytest.mark.simulation,
    pytest.mark.skipif(os.environ.get("DTCC_RUN_DEMOS") != "1",
                       reason="Live city and station data is opt-in"),
]


@pytest.mark.parametrize("name,field", [
    ("dataset_urban_heat", "temperature"),
    ("dataset_air_quality_field", "NO2"),
    ("dataset_traffic", "flow"),
    ("urban_heat_cooling", "temperature"),
    ("urban_heat_heatwave", "temperature"),
    ("smooth_reconstruction", "temperature"),
])
def test_city_demo_runs_twice(name, field, monkeypatch, tmp_path):
    if name != "dataset_traffic":
        pytest.importorskip("dolfinx")
        from petsc4py import PETSc
        if not np.issubdtype(PETSc.ScalarType, np.floating):
            pytest.skip("Heat and reconstruction require real PETSc")
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
            assert plt.get_fignums()
            assert any(ax.collections for ax in plt.gcf().axes)
            if name == "dataset_traffic":
                values = result.to_arrays()["attributes"][field]
                assert (directory / "solution.dtcc").is_file()
            else:
                values = next(item.values for item in result.fields if item.name == field)
                ax = plt.gcf().axes[0]
                samples = ax.collections[0]
                positions = np.column_stack(samples._offsets3d)
                by_position = dict(zip(map(tuple, result.vertices), values.ravel()))
                assert 0 < len(positions) <= len(values)
                np.testing.assert_allclose(
                    samples.get_array(), [by_position[tuple(point)] for point in positions],
                )
                assert (directory / "solution.xdmf").is_file()
                assert (directory / "solution.h5").is_file()
                assert (directory / "solution.coordinates.json").is_file()
                assert result.vertices[:, 0].min() > 300000
            assert len(values) and np.isfinite(values).all()
            if name.startswith("dataset_"):
                assert result.dataset_context.request.dataset_name == {
                    "dataset_urban_heat": "urban_heat_simulation",
                    "dataset_air_quality_field": "air_quality_field",
                    "dataset_traffic": "traffic_simulation",
                }[name]
            if index == 0:
                (directory / "keep.txt").write_text("unrelated")
            else:
                assert (directory / "keep.txt").read_text() == "unrelated"
            plt.close("all")
    finally:
        plt.close("all")
