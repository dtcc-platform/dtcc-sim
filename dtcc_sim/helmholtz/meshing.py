"""Select and generate city meshes using the DTCC volume-mesh dataset."""

from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
import hashlib
import json
import math
import shutil
import tempfile

from .simulator import _mesh_input_hashes


@dataclass(frozen=True)
class CityMeshParameters:
    """City bounds in EPSG:3006 and target mesh dimensions in meters."""

    bounds: tuple[float, float, float, float]
    h: float
    domain_height: float = 80.0
    min_building_detail: float = 0.5
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        if len(self.bounds) != 4 or not all(math.isfinite(x) for x in self.bounds):
            raise ValueError("bounds must contain four finite EPSG:3006 coordinates")
        if self.bounds[0] >= self.bounds[2] or self.bounds[1] >= self.bounds[3]:
            raise ValueError("bounds must be xmin, ymin, xmax, ymax with positive area")
        for name in ("h", "domain_height", "min_building_detail"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        reserved = {
            "bounds",
            "max_mesh_size",
            "domain_height",
            "min_building_detail",
            "format",
            "boundary_face_markers",
        }
        if reserved.intersection(self.options):
            raise ValueError(
                "Use the explicit mesh fields; format and boundary_face_markers are controlled by the workflow"
            )
        if any("output_dir" in name for name in self.options):
            raise ValueError(
                "Mesh debug output directories are not supported in cached generation"
            )

    def resolved_arguments(self):
        """Resolve and record the dataset defaults without generating a mesh."""
        from dtcc_core import datasets

        supplied = dict(
            self.options,
            bounds=list(self.bounds),
            max_mesh_size=self.h,
            domain_height=self.domain_height,
            min_building_detail=self.min_building_detail,
            boundary_face_markers=True,
            format=None,
        )
        unknown = set(supplied).difference(
            datasets.city_volume_mesh.ArgsModel.model_fields
        )
        if unknown:
            raise ValueError(f"Unknown city meshing options: {sorted(unknown)}")
        return datasets.city_volume_mesh.validate(supplied).model_dump(mode="json")


def _mesher_versions():
    versions = {}
    for name in ("dtcc-core", "dtcc-mesher", "dtcc-tetgen-wrapper"):
        try:
            dist = metadata.distribution(name)
            direct = dist.read_text("direct_url.json")
            versions[name] = {
                "version": dist.version,
                "source": json.loads(direct) if direct else None,
            }
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _generate_mesh(arguments, path):
    from dtcc_core import datasets
    import dtcc_core.io  # noqa: F401 - Registers VolumeMesh.save.
    import numpy as np

    mesh = datasets.city_volume_mesh(**arguments)
    vertices = np.asarray(mesh.vertices)
    cells = np.asarray(mesh.cells)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("DTCC generated invalid volume-mesh vertices")
    if cells.ndim != 2 or cells.shape[1] != 4 or not len(cells):
        raise ValueError("DTCC must generate a nonempty tetrahedral mesh")
    if (
        not np.issubdtype(cells.dtype, np.integer)
        or cells.min() < 0
        or cells.max() >= len(vertices)
    ):
        raise ValueError("DTCC generated invalid tetrahedron indices")
    if not len(getattr(mesh, "boundary_markers", ())):
        raise ValueError("DTCC generated no boundary markers")
    mesh.save(path)
    return {"vertices": len(vertices), "tetrahedra": len(cells)}


class CityMeshCache:
    """Reuse only intact meshes with matching full configuration and mesher versions."""

    def __init__(self, directory):
        self.directory = Path(directory).expanduser().resolve()

    def request(self, parameters):
        request = {
            "schema": 1,
            "crs": "EPSG:3006",
            "arguments": parameters.resolved_arguments(),
            "mesher": _mesher_versions(),
        }
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        return request, self.directory / key

    def describe(self, parameters):
        request, directory = self.request(parameters)
        return {
            "h": parameters.h,
            "request": request,
            "mesh_path": str(directory / "mesh.xdmf"),
        }

    def resolve(self, parameters):
        request, directory = self.request(parameters)
        manifest = directory / "mesh.json"
        if directory.exists():
            if not manifest.is_file():
                raise ValueError(f"Incomplete mesh cache entry: {directory}")
            record = json.loads(manifest.read_text())
            if record["request"] != request or record["files"] != _mesh_input_hashes(
                directory / "mesh.xdmf", relative=True
            ):
                raise ValueError(
                    f"Mesh cache metadata or contents changed: {directory}"
                )
            return directory / "mesh.xdmf", record, True
        self.directory.mkdir(parents=True, exist_ok=True)
        lock = directory.with_suffix(".lock")
        try:
            handle = lock.open("x")
        except FileExistsError:
            raise RuntimeError(f"Mesh generation is already locked: {lock}") from None
        with handle:
            handle.write("Mesh generation in progress\n")
        staging = None
        try:
            if directory.exists():
                raise RuntimeError(
                    "Another process created the mesh; retry to reuse it"
                )
            staging = Path(tempfile.mkdtemp(prefix=".building-", dir=self.directory))
            counts = _generate_mesh(request["arguments"], staging / "mesh.xdmf")
            record = {
                "request": request,
                "h": parameters.h,
                "counts": counts,
                "files": _mesh_input_hashes(staging / "mesh.xdmf", relative=True),
            }
            (staging / "mesh.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n"
            )
            staging.rename(directory)
            staging = None
            return directory / "mesh.xdmf", record, False
        finally:
            if staging is not None:
                shutil.rmtree(staging)
            lock.unlink()
