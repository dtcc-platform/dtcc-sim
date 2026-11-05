"""Validated case configuration and isolated solver outputs."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class ShallowWaterParameters:
    name: str = "wave"
    mesh_path: str | None = None
    mesh_data_path: str | None = None
    mesh_name: str = "mesh"
    cell_tags_name: str = "boundary_markers"
    nx: int = 24
    ny: int = 8
    Lx: float = 3.0
    Ly: float = 1.0
    x_center: float | None = 1.0
    y_center: float | None = 0.75
    final_time: float = 0.05
    cfl: float = 0.08
    gravity: float = 9.81
    bed_from_mesh: bool = False
    water_surface: float | None = None
    compact_pulse: bool = False
    h0: float = 1.0
    h_floor: float = 1e-6
    amplitude: float = 0.2
    sigma: float = 0.1
    tvb_M: float = 0.5
    tvb_nu: float = 1.5
    n_outputs: int = 5

    def __post_init__(self):
        if self.mesh_data_path is not None and (not isinstance(self.mesh_data_path, str) or not self.mesh_data_path):
            raise ValueError("mesh_data_path must be an NPZ filename")
        if self.mesh_data_path and self.mesh_path:
            raise ValueError("Choose one mesh input")
        for name in ("nx", "ny", "n_outputs"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("Lx", "Ly", "final_time", "cfl", "gravity", "h0",
                     "h_floor", "sigma", "tvb_nu", "tvb_M", "amplitude"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0 or (name not in ("tvb_M", "amplitude", "h0") and value == 0):
                raise ValueError(f"Invalid {name}")
        if type(self.bed_from_mesh) is not bool:
            raise ValueError("bed_from_mesh must be Boolean")
        if self.water_surface is not None and (isinstance(self.water_surface,bool) or not isinstance(self.water_surface,(int,float)) or not math.isfinite(self.water_surface)):
            raise ValueError("water_surface must be finite or null")
        if type(self.compact_pulse) is not bool:
            raise ValueError("compact_pulse must be Boolean")
        if self.final_time <= 1e-12:
            raise ValueError("Depth must exceed its floor and final_time must exceed 1e-12")
        for name in ("x_center", "y_center"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError(f"{name} must be finite or null")
        for name in ("name", "mesh_name", "cell_tags_name"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        if self.mesh_path is not None:
            if not isinstance(self.mesh_path, str) or Path(self.mesh_path).suffix.lower() not in (".xdmf", ".xmf"):
                raise ValueError("mesh_path must be an XDMF filename")


class ShallowWaterSimulator:
    def __init__(self, *, parameters=None, output_dir, mesh=None):
        self.parameters = parameters if parameters is not None else ShallowWaterParameters()
        if not isinstance(self.parameters, ShallowWaterParameters):
            raise TypeError("parameters must be ShallowWaterParameters")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.mesh = mesh
        if mesh is not None and (self.parameters.mesh_path or self.parameters.mesh_data_path):
            raise ValueError("Choose either a DTCC mesh object or a mesh file")

    def run(self):
        from mpi4py import MPI
        from petsc4py import PETSc
        import numpy as np

        if MPI.COMM_WORLD.size != 1:
            raise RuntimeError("This interface currently supports one MPI rank")
        if np.issubdtype(PETSc.ScalarType, np.complexfloating):
            raise RuntimeError("Use a real-valued PETSc runtime")
        if self.parameters.mesh_path and not Path(self.parameters.mesh_path).is_file():
            raise FileNotFoundError(self.parameters.mesh_path)
        from . import backend
        from .mesh_io import load_mesh, validate_mesh

        mesh_data = None
        if self.mesh is not None:
            mesh_data = validate_mesh(self.mesh.vertices, self.mesh.faces, self.mesh.markers)
        elif self.parameters.mesh_data_path:
            mesh_data = load_mesh(self.parameters.mesh_data_path)

        if self.parameters.bed_from_mesh and mesh_data is None:
            raise ValueError("Terrain requires an NPZ or DTCC mesh with vertex elevations")
        if mesh_data is not None and not self.parameters.bed_from_mesh and not np.allclose(mesh_data["vertices"][:,2],mesh_data["vertices"][0,2],rtol=0,atol=1e-9):
            raise ValueError("Set bed_from_mesh=true to use nonhorizontal terrain")
        self.output_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "parameters": asdict(self.parameters),
            "upstream_commit": "2cf10273eea2441137393b57040528b1c03e8725",
            "backend_sha256": hashlib.sha256(Path(backend.__file__).read_bytes()).hexdigest(),
        }
        (self.output_dir / "configuration.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if mesh_data is not None:
            np.savez(self.output_dir / "input_mesh.npz", **mesh_data)
        status_path = self.output_dir / "status.json"
        status_path.write_text(json.dumps({"status": "running"}) + "\n")
        try:
            result = backend.run(self.parameters, self.output_dir, mesh_data=mesh_data)
        except Exception as error:
            status_path.write_text(json.dumps({"status": "failed", "error": str(error)}) + "\n")
            raise
        status_path.write_text(json.dumps({"status": "completed"}) + "\n")
        return result
