"""Local numerical coordinates with explicit input-frame provenance."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np


@dataclass(frozen=True)
class MeshCoordinateFrame:
    """Translation from a local computational frame to the input coordinates."""

    origin: tuple[float, float, float]
    crs: str | None = None

    @classmethod
    def from_mesh(cls, mesh, *, crs=None):
        from mpi4py import MPI

        points = mesh.geometry.x
        if points.dtype != np.float64:
            raise ValueError("Solvers require Float64 mesh coordinates")
        local_min = points.min(axis=0) if len(points) else np.full(3, np.inf)
        origin = np.empty(3, dtype=np.float64)
        mesh.comm.Allreduce(local_min, origin, op=MPI.MIN)
        if not np.isfinite(origin).all():
            raise ValueError("Mesh coordinates must be nonempty and finite")
        if not mesh.comm.allreduce(bool(np.isfinite(points).all()), op=MPI.LAND):
            raise ValueError("Mesh coordinates must be finite")
        return cls(tuple(map(float, origin)), crs or None)

    def to_local(self, points):
        points = np.asarray(points, dtype=np.float64)
        return points - np.asarray(self.origin)[: points.shape[-1]]

    def to_input(self, points):
        points = np.asarray(points, dtype=np.float64)
        return points + np.asarray(self.origin)[: points.shape[-1]]

    def input_callback(self, callback):
        """Evaluate an input-coordinate callback from local FEniCSx coordinates."""

        def evaluate(points):
            return callback(self.to_input(np.asarray(points).T).T)

        return evaluate

    def as_dict(self):
        return {
            "local_to_input_translation": list(self.origin),
            "input_crs": self.crs,
            "computation_coordinates": "local",
            "output_coordinates": "input",
        }


def input_crs(native_mesh=None, *, generated=False, declared=None):
    """Resolve declared native CRS or the city provider's known CRS."""
    if declared is not None:
        if not isinstance(declared, str) or not declared.strip():
            raise ValueError("mesh_crs must be a nonempty CRS string")
        declared = declared.strip()
    crs = getattr(getattr(native_mesh, "transform", None), "srs", None)
    known = str(crs) if crs else "EPSG:3006" if generated else None
    if known and declared and known != declared:
        raise ValueError("mesh_crs conflicts with the input mesh CRS")
    return known or declared


def write_coordinate_frame(output_path, frame, *, comm=None):
    """Persist the input CRS and numerical translation beside an exported mesh."""
    record = frame.as_dict() if isinstance(frame, MeshCoordinateFrame) else frame
    path = Path(output_path).with_suffix(".coordinates.json")
    if comm is None:
        path.write_text(json.dumps(record, indent=2) + "\n")
    else:
        failure = None
        if comm.rank == 0:
            try:
                path.write_text(json.dumps(record, indent=2) + "\n")
            except Exception as error:
                failure = f"{type(error).__name__}: {error}"
        failure = comm.bcast(failure, root=0)
        if failure is not None:
            raise OSError(f"Cannot write coordinate frame to {path}: {failure}")
    return path


@contextmanager
def local_mesh_coordinates(mesh, *, crs=None):
    """Compute locally and restore supplied coordinates exactly on every exit."""
    frame = MeshCoordinateFrame.from_mesh(mesh, crs=crs)
    original = mesh.geometry.x.copy()
    mesh.geometry.x[:] = frame.to_local(original)
    try:
        yield frame
    finally:
        mesh.geometry.x[:] = original


def localize_mesh(mesh, *, crs=None):
    """Localize a private backend mesh while retaining exact export coordinates."""
    frame = MeshCoordinateFrame.from_mesh(mesh, crs=crs)
    mesh._dtcc_input_coordinates = mesh.geometry.x.copy()
    mesh._dtcc_coordinate_frame = frame
    mesh.geometry.x[:] = frame.to_local(mesh.geometry.x)
    return frame


@contextmanager
def input_mesh_coordinates(mesh):
    """Temporarily expose a private backend mesh in its input frame for export."""
    frame = getattr(mesh, "_dtcc_coordinate_frame", None)
    if frame is None:
        yield MeshCoordinateFrame((0.0, 0.0, 0.0))
        return
    local = mesh.geometry.x.copy()
    mesh.geometry.x[:] = mesh._dtcc_input_coordinates
    try:
        yield frame
    finally:
        mesh.geometry.x[:] = local
