"""Validated single-cell-type layouts for triangular DG1 states."""

import numpy as np


def triangle_corner_order(target, source):
    """Map scalar DG1 nodes to each input triangle's XY vertex order."""
    target = np.asarray(target)[..., :2]
    source = np.asarray(source)[..., :2]
    if target.shape != source.shape or target.ndim != 3 or target.shape[1:] != (3, 2):
        raise ValueError("Expected matching triangular DG1 coordinates")
    distances = np.linalg.norm(target[:, :, None] - source[:, None, :], axis=-1)
    order = distances.argmin(axis=2)
    if (not np.isfinite(distances).all()
            or np.any(np.take_along_axis(distances, order[..., None], axis=2) > 1e-8)
            or not np.all(np.sort(order, axis=1) == np.arange(3))):
        raise ValueError("DG1 nodes must match the input triangle vertices")
    return order


def triangle_geometry(domain):
    maps = domain.geometry.dofmaps
    if len(maps) != 1 or domain.topology.cell_type.name != "triangle":
        raise NotImplementedError("Shallow water supports one triangular cell type only")
    cells = domain.topology.index_map(domain.topology.dim)
    count = cells.size_local + cells.num_ghosts
    faces = np.asarray(maps[0])
    coordinates = domain.geometry.x
    if (faces.shape != (count, 3) or faces.dtype.kind not in "iu"
            or coordinates.dtype != np.dtype("float64")
            or not np.isfinite(coordinates).all()
            or np.any(faces < 0) or np.any(faces >= len(coordinates))):
        raise ValueError("Expected linear triangles with finite Float64 coordinates")
    return faces


def scalar_cell_dofs(space, nodes_per_cell):
    """Return a scalar DG cell map in backend cell order."""
    triangle_geometry(space.mesh)
    maps = space.dofmaps
    if len(maps) != 1 or maps[0].bs != 1 or maps[0].index_map_bs != 1:
        raise ValueError("Expected one scalar degree-of-freedom map")
    cells = space.mesh.topology.index_map(space.mesh.topology.dim)
    count = cells.size_local + cells.num_ghosts
    dofs = np.asarray(maps[0].list)
    coordinates = space.tabulate_dof_coordinates()
    if (dofs.shape != (count, nodes_per_cell) or dofs.dtype.kind not in "iu"
            or np.any(dofs < 0) or np.any(dofs >= len(coordinates))
            or np.unique(dofs).size != dofs.size
            or coordinates.dtype != np.dtype("float64")
            or not np.isfinite(coordinates).all()):
        raise ValueError("Invalid scalar discontinuous cell layout")
    return dofs


def collapse_components(space, state_size):
    """Select the sole cell-type map and validate the three DG1 components."""
    spaces, components = [], []
    for component in range(3):
        scalar, maps = space.sub(component).collapse()
        cells = scalar_cell_dofs(scalar, 3)
        if len(maps) != 1:
            raise ValueError("Expected one collapse map for triangular DG1")
        dofs = np.asarray(maps[0])
        if (dofs.dtype.kind not in "iu" or dofs.shape != (cells.size,)
                or np.any(dofs < 0) or np.any(dofs >= state_size)
                or np.unique(dofs).size != dofs.size):
            raise ValueError("Invalid DG1 component collapse map")
        if spaces and (not np.array_equal(cells, scalar_cell_dofs(spaces[0], 3))
                       or not np.array_equal(scalar.tabulate_dof_coordinates(),
                                             spaces[0].tabulate_dof_coordinates())):
            raise ValueError("DG1 component cell layouts do not match")
        spaces.append(scalar)
        components.append(dofs)
    if not np.array_equal(np.sort(np.concatenate(components)), np.arange(state_size)):
        raise ValueError("DG1 component maps must partition the state vector")
    return spaces, components
