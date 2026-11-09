"""Transient snapshot selection delegates map rendering to native Core meshes."""

from pathlib import Path

import numpy as np
import pytest

from dtcc_core.model import Mesh
from dtcc_sim.shallow_water import ShallowWaterResult


def test_result_delegates_snapshot_axes_and_colour_range(monkeypatch):
    meshes = [Mesh(faces=np.array([[0, 1, 2], [0, 2, 3]]),
                   markers=np.array([-1, 0])) for _ in range(2)]
    result = ShallowWaterResult(Path("unused"))
    monkeypatch.setattr(ShallowWaterResult, "snapshot_times", property(lambda self: (0., 2.5)))
    monkeypatch.setattr(ShallowWaterResult, "to_dtcc", lambda self, index: meshes[index])
    calls = []
    axes = object()
    def capture(self, **kwargs):
        calls.append((self, kwargs))
        return axes
    monkeypatch.setattr(Mesh, "plot", capture)
    assert result.plot(snapshot=0, ax=axes, show=False, vmax=.5, theme="dark") is axes
    assert result.plot(show=False, vmax=.5) is axes
    for (mesh, options), expected, time in zip(calls, meshes, (0., 2.5)):
        assert mesh is expected
        assert options["projection"] == "xy" and options["field"] == "water_depth"
        assert options["vmin"] == 0. and options["vmax"] == .5
        assert options["show"] is False
        assert f"{time:g} s" in options["title"]
        np.testing.assert_array_equal(options["context_mask"], [False, True])
    assert calls[0][1]["ax"] is axes and calls[0][1]["theme"] == "dark"


def test_empty_and_invalid_markers(monkeypatch):
    from dtcc_sim.shallow_water.plotting import _plot_depth

    mesh = Mesh(faces=np.array([[0, 1, 2]]))
    calls = []
    monkeypatch.setattr(Mesh, "plot", lambda self, **kwargs: calls.append(kwargs))
    _plot_depth(mesh, 0, show=False)
    assert calls[0]["context_mask"] is None
    mesh.markers = np.array([0, 1])
    with pytest.raises(ValueError, match="Markers"):
        _plot_depth(mesh, 0, show=False)
