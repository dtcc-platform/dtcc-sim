"""Geometry, data and metric checks for the AIJ benchmark cases."""

import numpy as np
import pytest

from benchmarks.aij.analysis import metrics
from benchmarks.aij.cases import CASES
from benchmarks.aij.geometry import Box, Spacing, block_cells, facet_markers, graded_axis, grid_axes


def test_graded_axis_keeps_fixed_points_and_grows():
    nodes = graded_axis(0., 10., [4., 5.], lambda x: np.minimum(2., .1 + .2 * np.maximum(4. - x, x - 5.).clip(0)))
    assert nodes[0] == 0. and nodes[-1] == 10.
    assert np.any(np.isclose(nodes, 4.)) and np.any(np.isclose(nodes, 5.))
    steps = np.diff(nodes)
    assert steps.min() > 0 and steps[(nodes[:-1] >= 4.) & (nodes[1:] <= 5.)].max() <= .1 + 1e-12
    growth = steps[1:] / steps[:-1]
    assert growth[nodes[1:-1] > 5.].max() < 1.35


def tetrahedron_volumes(points, cells):
    a, b, c, d = (points[cells[:, i]] for i in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6


def test_block_cells_fill_the_domain_outside_the_building():
    building = Box(-.5, .5, -.5, 0., 2.)
    domain = ((-3., 6.), (-3., 0.), 5.)
    axes = grid_axes([building], domain, Spacing(h=.25, ground=.1, ratio=1.3, maximum=1.))
    for axis, values in zip(axes, [(-.5, .5), (-.5, 0.), (2.,)]):
        assert all(np.any(np.isclose(axis, v)) for v in values)
    points, cells = block_cells(*axes, [building])
    volumes = np.abs(tetrahedron_volumes(points, cells))
    assert volumes.min() > 0
    assert np.isclose(volumes.sum(), 9 * 3 * 5 - 1 * .5 * 2)
    centroids = points[cells].mean(axis=1)
    inside = (np.abs(centroids[:, 0]) < .5) & (centroids[:, 1] > -.5) & (centroids[:, 2] < 2.)
    assert not inside.any()


def test_facet_markers_follow_the_dtcc_scheme():
    building = Box(-.5, .5, -.5, 0., 2.)
    domain = ((-3., 6.), (-3., 0.), 5.)
    midpoints = np.array([[0., -2., 0.], [0., -2., 5.], [-3., -1., 1.], [6., -1., 1.], [0., -3., 1.],
                          [0., 0., 3.], [-.5, -.2, 1.], [.1, -.5, 1.], [0., -.2, 2.]])
    values = facet_markers(midpoints, domain, [building], 1e-9)
    assert values.tolist() == [-1, -2, -3, -4, -5, -6, 0, 0, 1]
    with pytest.raises(ValueError, match="outside"):
        facet_markers(np.array([[1., -1., 1.]]), domain, [building], 1e-9)


def test_metrics():
    observed = np.array([1., 1., 1., .02, np.nan])
    predicted = np.array([1.2, 2.5, .9, .06, 1.])
    result = metrics(predicted, observed)
    assert result["points"] == 4
    assert result["hit_rate"] == .75  # 2.5 misses; 0.06 hits through the absolute tolerance
    assert result["fac2"] == .5  # 2.5 and 0.06/0.02 lie outside a factor two
    perfect = metrics(observed, observed)
    assert perfect["hit_rate"] == 1. and perfect["fractional_bias"] == 0. and perfect["nmse"] == 0.


def test_log_law_fit_recovers_parameters():
    pytest.importorskip("scipy")
    from benchmarks.aij.analysis import fit_log_law, log_law

    z = np.geomspace(5e-3, .3, 12)
    friction_velocity, z0, rms = fit_log_law(z, log_law(z, .25, 1e-4), .2)
    assert np.isclose(friction_velocity, .25, rtol=1e-6) and np.isclose(z0, 1e-4, rtol=1e-5) and rms < 1e-8


@pytest.mark.parametrize("name", sorted(CASES))
def test_case_geometry_is_symmetric_half_domain(name):
    case = CASES[name]
    building = case.building()
    (x0, x1), (y0, y1), top = case.domain()
    assert np.isclose(building.x0, -building.x1) and building.y1 == 0. == y1
    assert np.isclose(-building.y0, case.width / 2) and building.height == case.height
    assert np.isclose(x0, building.x0 - 5 * case.height) and np.isclose(x1, building.x1 + 10 * case.height)
    assert np.isclose(top, 6 * case.height) and np.isclose(y0, building.y0 - 5 * case.height)


@pytest.mark.parametrize("name, sections, points", [
    ("A", {"vertical": 66, "horizontal z=0.125b": 60, "horizontal z=1.25b": 60}, 186),
    ("B", {"vertical": 109, "horizontal": 115}, 224),
])
def test_workbooks_when_available(name, sections, points):
    pytest.importorskip("xlrd")
    from benchmarks.aij.data import read_workbook

    case = CASES[name]
    if not case.path.exists():
        pytest.skip(f"{case.workbook} is not in benchmarks/data/aij")
    data = read_workbook(case.path, case.length)
    assert {key: len(s.number) for key, s in data.sections.items()} == sections
    assert sum(len(s.number) for s in data.sections.values()) == points
    for section in data.sections.values():
        assert np.all(section.points[:, 1] <= 0) and np.all(section.points[:, 2] > 0)
        outside = ((np.abs(section.points[:, 0]) >= case.depth / 2) | (section.points[:, 1] <= -case.width / 2)
                   | (section.points[:, 2] >= case.height))
        assert outside.all()
    assert np.all(np.diff(data.inflow_z) > 0) and data.inflow_z[0] < case.height < data.inflow_z[-1]


@pytest.mark.fenics
def test_block_mesh_markers_and_volume():
    pytest.importorskip("dolfinx")
    import dolfinx.fem as fem
    import dolfinx.mesh as dm
    import ufl
    from mpi4py import MPI

    from benchmarks.aij.geometry import block_mesh

    building = Box(-.5, .5, -.5, 0., 2.)
    domain = ((-3., 6.), (-3., 0.), 5.)
    mesh, markers, _ = block_mesh([building], domain, Spacing(h=.5, ground=.25, ratio=1.4, maximum=1.5))
    volume = mesh.comm.allreduce(fem.assemble_scalar(fem.form(1. * ufl.dx(domain=mesh))), op=MPI.SUM)
    assert np.isclose(volume, 9 * 3 * 5 - 1 * .5 * 2)
    shifted = dm.meshtags(mesh, 2, markers.indices, markers.values + 10)  # UFL reserves negative ids
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=shifted)
    area = lambda tag: mesh.comm.allreduce(fem.assemble_scalar(fem.form(1. * ds(tag + 10))), op=MPI.SUM)
    assert np.isclose(area(0), 2 * (.5 * 2) + 1 * 2)  # windward, leeward and the outer side wall
    assert np.isclose(area(1), .5)
    assert np.isclose(area(-1), 9 * 3 - .5)
    assert np.isclose(area(-6), 9 * 5 - 1 * 2)


def test_probe_heights_map_to_the_computational_ground():
    from benchmarks.aij.run import mesh_point

    mapped = mesh_point(np.array([[0., -1., .005], [.1, 0., .001]]), .002)
    assert np.allclose(mapped[0], [0., -1., .003])
    assert np.isnan(mapped[1, 2]) and mapped[1, 0] == .1
