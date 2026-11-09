"""Filled triangle maps preserve values, coordinates and building boundaries."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from dtcc_core.model import Field, Mesh
from dtcc_sim.shallow_water.plotting import _plot_depth as plot_depth


@pytest.fixture
def mesh():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.],
                         [0., 1., 0.], [2., 0., 0.], [2., 1., 0.]])
    vertices[:, :2] += [319845., 6397785.]
    mesh = Mesh(vertices=vertices, faces=np.array([[0, 1, 2], [0, 2, 3],
                                                   [1, 4, 5], [1, 5, 2]]),
                markers=np.array([-1, -1, 0, 0], dtype=np.int32))
    mesh.transform.srs = "EPSG:3006"
    # Large interior depths must not dominate the ground-water colour scale.
    mesh.fields = [Field(name="water_depth", unit="m", association="face", dim=1,
                         values=np.array([[0.], [.2], [50.], [50.]]))]
    yield mesh
    plt.close("all")


def test_filled_faces_scale_time_and_building_outline(mesh, monkeypatch):
    original = (mesh.vertices.copy(), mesh.faces.copy(), mesh.fields[0].values.copy())
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    ax = plot_depth(mesh, 2.5, show=False)
    ax.figure.canvas.draw()
    assert shown == []
    assert ax.name == "rectilinear" and ax.get_aspect() == 1.
    assert "2.5 s" in ax.get_title() and "Water depth" in ax.get_title()
    assert ax.get_xlabel() == "Easting [m]" and ax.get_ylabel() == "Northing [m]"
    layers = {c.get_gid(): c for c in ax.collections}
    water = layers["water_depth"]
    np.testing.assert_array_equal(water.get_array(), [0., .2])
    assert water.norm.vmin == 0. and water.norm.vmax == .2
    assert len(water.get_paths()) == 2
    for path, face in zip(water.get_paths(), mesh.faces[:2]):
        np.testing.assert_array_equal(path.vertices[:3], mesh.vertices[face, :2])
    assert len(layers["buildings"].get_paths()) == 2
    assert len(layers["building_outlines"].get_segments()) == 4
    diagonal = mesh.vertices[[1, 5], :2]
    assert not any(np.array_equal(s, diagonal) for s in layers["building_outlines"].get_segments())
    assert ax.figure.axes[1].get_ylabel() == "Water depth [m]"
    texts = [t.get_text() for t in ax.figure.findobj(matplotlib.text.Text)]
    assert "EPSG:3006" in texts and any(t.endswith(" m") for t in texts)
    for before, after in zip(original, (mesh.vertices, mesh.faces, mesh.fields[0].values)):
        np.testing.assert_array_equal(before, after)


def test_dry_mesh_without_markers_and_shared_colour_scale(mesh):
    mesh.markers = np.array([], dtype=np.int32)
    mesh.fields[0].values[:] = 0.
    fig, ax = plt.subplots()
    assert plot_depth(mesh, 0., ax=ax, show=False) is ax
    assert len(fig.axes) == 1
    assert any(t.get_text() == "All ground cells dry" for t in ax.texts)
    plt.close(fig)
    ax = plot_depth(mesh, 1., vmax=.5, show=False, theme="dark")
    water = next(c for c in ax.collections if c.get_gid() == "water_depth")
    assert water.norm.vmax == .5 and len(ax.figure.axes) == 2
    assert len(water.get_paths()) == len(mesh.faces)


@pytest.mark.parametrize("vmax", [0., -1., float("nan"), float("inf"), True])
def test_invalid_colour_limit(mesh, vmax):
    with pytest.raises(ValueError, match="vmax"):
        plot_depth(mesh, 0., vmax=vmax, show=False)


def test_rejects_3d_axes_and_missing_ground(mesh):
    ax = plt.figure().add_subplot(projection="3d")
    with pytest.raises(ValueError, match="2D axes"):
        plot_depth(mesh, 0., ax=ax, show=False)
    mesh.markers[:] = 0
    with pytest.raises(ValueError, match="ground face"):
        plot_depth(mesh, 0., show=False)
