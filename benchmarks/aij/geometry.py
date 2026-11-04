"""Graded tetrahedral meshes around rectangular buildings on flat ground.

The mesh is a tensor-product grid whose spacing grows geometrically away from
the buildings, with the cells inside buildings removed and every hexahedron
split into six tetrahedra along its main diagonal. Building faces, the ground
and the domain faces are therefore planar and axis aligned. Facets carry the
DTCC boundary markers used by the flow solvers: ground -1, top -2, x faces
-3/-4, y faces -5/-6, walls of building i marker i and its roof N + i.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Box:
    """Axis-aligned building [x0, x1] × [y0, y1] × [0, height]."""

    x0: float
    x1: float
    y0: float
    y1: float
    height: float


@dataclass(frozen=True)
class Spacing:
    """Cell sizes: ``h`` at buildings, ``ground`` at the ground, growing by
    ``ratio`` per cell up to ``maximum``."""

    h: float
    ground: float
    ratio: float = 1.2
    maximum: float = np.inf


def graded_axis(lo, hi, fixed, size):
    """Nodes on [lo, hi] containing ``fixed`` points, spaced by ``size(x)``.

    Each interval between consecutive fixed points gets ceil(∫dx/size) cells
    distributed uniformly in the stretched coordinate ∫dx/size.
    """
    points = np.unique(np.clip(np.concatenate([[lo, hi], np.asarray(fixed, dtype=float)]), lo, hi))
    nodes = [points[:1]]
    for a, b in zip(points[:-1], points[1:]):
        x = np.linspace(a, b, 2001)
        density = 1. / size(x)
        stretched = np.concatenate([[0.], np.cumsum(.5 * (density[1:] + density[:-1]) * np.diff(x))])
        cells = max(1, int(np.ceil(stretched[-1] - 1e-9)))
        nodes.append(np.interp(np.linspace(0., stretched[-1], cells + 1)[1:], stretched, x))
    return np.concatenate(nodes)


def _distance(x, intervals):
    gaps = [np.maximum(np.maximum(a - x, x - b), 0.) for a, b in intervals]
    return np.min(gaps, axis=0) if gaps else np.full_like(x, np.inf)


def grid_axes(buildings, domain, spacing, region=None):
    """Graded x, y and z node coordinates for ``domain = ((x0, x1), (y0, y1), top)``.

    Building edges are grid nodes. The spacing is ``spacing.h`` at the
    buildings and inside the optional horizontal ``region = ((x0, x1), (y0, y1))``.
    """
    (x0, x1), (y0, y1), top = domain
    growth = spacing.ratio - 1.

    def horizontal(axis, lo, hi):
        edges = [(b.x0, b.x1) if axis == 0 else (b.y0, b.y1) for b in buildings]
        refined = edges + ([region[axis]] if region is not None else [])
        size = lambda x: np.minimum(spacing.maximum, spacing.h + growth * _distance(x, refined))
        return graded_axis(lo, hi, [v for interval in edges for v in interval], size)

    roof = max(b.height for b in buildings)
    vertical = lambda z: np.minimum.reduce([
        np.full_like(z, spacing.maximum), spacing.ground + growth * z,
        spacing.h + growth * np.maximum(z - roof, 0.)])
    return horizontal(0, x0, x1), horizontal(1, y0, y1), graded_axis(0., top, [b.height for b in buildings], vertical)


def block_cells(x, y, z, buildings):
    """Vertices and tetrahedra of the grid with building cells removed."""
    nx, ny, nz = len(x), len(y), len(z)
    index = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    centre = [.5 * (c[1:] + c[:-1]) for c in (x, y, z)]
    X, Y, Z = np.meshgrid(*centre, indexing="ij")
    solid = np.zeros(X.shape, dtype=bool)
    for b in buildings:
        solid |= (X > b.x0) & (X < b.x1) & (Y > b.y0) & (Y < b.y1) & (Z < b.height)
    i, j, k = np.nonzero(~solid)
    corner = lambda di, dj, dk: index[i + di, j + dj, k + dk]
    tetrahedra = []
    for order in ([0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]):
        step, path = np.zeros(3, dtype=int), [corner(0, 0, 0)]
        for axis in order:
            step[axis] = 1
            path.append(corner(*step))
        tetrahedra.append(np.stack(path, axis=1))
    cells = np.concatenate(tetrahedra)
    used, cells = np.unique(cells, return_inverse=True)
    cells = cells.reshape(-1, 4)
    points = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)[used]
    return points, cells.astype(np.int64)


def facet_markers(midpoints, domain, buildings, tolerance):
    """DTCC markers for exterior facets from their midpoints."""
    (x0, x1), (y0, y1), top = domain
    n = len(buildings)
    values = np.full(len(midpoints), np.iinfo(np.int32).min, dtype=np.int32)
    for i, b in enumerate(buildings):
        inside = ((midpoints[:, 0] > b.x0 - tolerance) & (midpoints[:, 0] < b.x1 + tolerance)
                  & (midpoints[:, 1] > b.y0 - tolerance) & (midpoints[:, 1] < b.y1 + tolerance)
                  & (midpoints[:, 2] < b.height + tolerance))
        values[inside] = i
        values[inside & np.isclose(midpoints[:, 2], b.height, atol=tolerance)] = n + i
    for axis, coordinate, marker in [(2, 0., -1), (2, top, -2), (0, x0, -3), (0, x1, -4), (1, y0, -5), (1, y1, -6)]:
        values[np.isclose(midpoints[:, axis], coordinate, atol=tolerance)] = marker
    if np.any(values == np.iinfo(np.int32).min):
        raise ValueError("exterior facets outside the domain faces and buildings")
    return values


def block_mesh(buildings, domain, spacing, region=None, comm=None):
    """Distributed DOLFINx mesh and facet markers for buildings in a box domain.

    The mesh is built on rank zero and partitioned. Returns
    ``(mesh, markers, axes)`` with the grid node coordinates in ``axes``.
    """
    import basix.ufl
    import dolfinx.mesh as dm
    import ufl
    from mpi4py import MPI

    comm = MPI.COMM_WORLD if comm is None else comm
    axes = grid_axes(buildings, domain, spacing, region)
    if comm.rank == 0:
        points, cells = block_cells(*axes, buildings)
    else:
        points, cells = np.zeros((0, 3)), np.zeros((0, 4), dtype=np.int64)
    element = ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,)))
    mesh = dm.create_mesh(comm, cells, element, points)
    fd = mesh.topology.dim - 1
    mesh.topology.create_entities(fd)
    mesh.topology.create_connectivity(fd, fd + 1)
    facets = dm.exterior_facet_indices(mesh.topology)
    tolerance = 1e-6 * min(np.min(np.diff(axis)) for axis in axes)
    values = facet_markers(dm.compute_midpoints(mesh, fd, facets), domain, buildings, tolerance)
    order = np.argsort(facets)
    return mesh, dm.meshtags(mesh, fd, facets[order], values[order]), axes
