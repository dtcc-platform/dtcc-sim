"""Portable triangular mesh exchange between DTCC and the real PETSc runtime."""

from pathlib import Path
import numpy as np


def validate_mesh(vertices, faces, markers=None):
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError("Mesh vertices must have shape (N, 3)")
    if vertices.dtype.kind not in "fi" or not np.isfinite(vertices).all():
        raise ValueError("Mesh vertices must be finite real coordinates")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or faces.dtype.kind not in "iu":
        raise ValueError("Mesh faces must be integer triangles")
    if np.min(faces) < 0 or np.max(faces) >= len(vertices):
        raise ValueError("Triangle vertex index out of bounds")
    if len(np.unique(vertices[:, :2], axis=0)) != len(vertices):
        raise ValueError("Duplicate XY vertices are unsupported")
    triangles = vertices[faces, :2]
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    area2 = np.abs(edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0])
    max_edge = np.maximum.reduce([np.linalg.norm(triangles[:, i] - triangles[:, j], axis=1)
                                  for i, j in ((0, 1), (1, 2), (2, 0))])
    if np.any(area2 <= 1e-12 * max_edge**2):
        raise ValueError("Degenerate or extremely thin triangle")
    if len(np.unique(np.sort(faces, axis=1), axis=0)) != len(faces):
        raise ValueError("Duplicate faces are unsupported")
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    if np.max(np.unique(edges, axis=0, return_counts=True)[1]) > 2:
        raise ValueError("Nonmanifold triangle edge")
    if markers is None or np.asarray(markers).size == 0:
        markers = np.full(len(faces), -1, dtype=np.int32)
    markers = np.asarray(markers)
    if markers.shape != (len(faces),) or markers.dtype.kind not in "iu" or np.any(markers > np.iinfo(np.int32).max) or np.any(markers < np.iinfo(np.int32).min):
        raise ValueError("Markers must be one int32-compatible integer per face")
    return dict(vertices=vertices.astype(float), faces=faces.astype(np.int64),
                markers=markers.astype(np.int32), min_altitude=float(np.min(area2 / max_edge)))


def save_mesh(mesh, path):
    """Write geometry and building markers from a DTCC Mesh to an NPZ file."""
    data = validate_mesh(mesh.vertices, mesh.faces, mesh.markers)
    with Path(path).open("xb") as stream:
        np.savez(stream, **data)


def load_mesh(path):
    with np.load(path, allow_pickle=False) as data:
        return validate_mesh(data["vertices"], data["faces"], data["markers"])


def build_domain(data, comm):
    import basix.ufl
    import ufl
    from dolfinx import mesh

    coordinate_element = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    domain = mesh.create_mesh(comm, data["faces"], ufl.Mesh(coordinate_element),
                              data["vertices"][:, :2].copy())
    original = np.asarray(domain.topology.original_cell_index, dtype=np.int64)
    if not np.array_equal(np.sort(original), np.arange(len(data["faces"]))):
        raise RuntimeError("Imported cell ordering is not a bijection")
    tags = mesh.meshtags(domain, 2, np.arange(len(original), dtype=np.int32),
                         data["markers"][original])
    return domain, tags, np.argsort(original)
