"""Native DTCC wind-tunnel geometry with an aligned box obstruction."""

import numpy as np
from dtcc_core import BoxMesh
from dtcc_core.model import VolumeMesh


def bluff_body_mesh():
    """A 6 × 2 × 2 m tunnel around [2,3] × [0.5,1.5] × [0,1]."""
    mesh = BoxMesh(24, 8, 8)
    vertices = mesh.vertices * np.array([6., 2., 2.])
    centers = vertices[mesh.cells].mean(axis=1)
    obstacle = ((centers[:, 0] > 2) & (centers[:, 0] < 3)
                & (centers[:, 1] > .5) & (centers[:, 1] < 1.5)
                & (centers[:, 2] < 1))
    cells = mesh.cells[~obstacle]
    used, inverse = np.unique(cells, return_inverse=True)
    vertices, cells = vertices[used], inverse.reshape(-1, 4)
    patterns = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    faces = cells[:, patterns].reshape(-1, 3)
    _, first, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    boundary = faces[first[counts == 1]]
    points = vertices[boundary].mean(axis=1)
    markers = np.zeros(len(boundary), dtype=np.int64)
    markers[np.isclose(points[:, 2], 1.)] = 1
    for axis, lower, upper, low_tag, high_tag in [(0, 0, 6, -3, -4), (1, 0, 2, -5, -6), (2, 0, 2, -1, -2)]:
        markers[np.isclose(points[:, axis], lower)] = low_tag
        markers[np.isclose(points[:, axis], upper)] = high_tag
    return VolumeMesh(vertices=vertices, cells=cells, boundary_faces=boundary, boundary_markers=markers)
