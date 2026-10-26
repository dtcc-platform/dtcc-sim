"""Map complex Helmholtz solutions to DTCC vertex fields."""

from pathlib import Path
import json
import xml.etree.ElementTree as ET


FIELD_NAMES = ("helmholtz_real", "helmholtz_imaginary", "helmholtz_amplitude")


def attach_helmholtz_fields(
    mesh, dof_coordinates, solution, *, unit="", tolerance=1.0e-8
):
    """Attach P1 solution values using a checked bijection of absolute coordinates."""
    import numpy as np
    from scipy.spatial import cKDTree
    from dtcc_core.model import Field

    source = np.asarray(dof_coordinates, dtype=np.float64)
    target = np.asarray(mesh.vertices, dtype=np.float64)
    values = np.asarray(solution)
    if source.ndim != 2 or source.shape[1] != 3 or target.shape != source.shape:
        raise ValueError(
            "P1 DOF coordinates and DTCC vertices must have matching (N, 3) shapes"
        )
    if values.ndim != 1 or len(values) != len(source) or not len(source):
        raise ValueError("One complex solution value is required per P1 DOF")
    if not all(np.isfinite(x).all() for x in (source, target, values)):
        raise ValueError("Coordinates and solution values must be finite")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Coordinate tolerance must be finite and positive")
    if len(np.unique(source, axis=0)) != len(source) or len(
        np.unique(target, axis=0)
    ) != len(target):
        raise ValueError("Duplicate coordinates make the vertex mapping ambiguous")
    distances, indices = cKDTree(source).query(target)
    if distances.max() > tolerance or len(np.unique(indices)) != len(source):
        raise ValueError(
            "Solution DOFs do not map bijectively to the DTCC mesh vertices"
        )
    # Reject geometries where another source point also falls inside the tolerance.
    if len(source) > 1:
        second_distances, _ = cKDTree(source).query(target, k=2)
        if np.any(second_distances[:, 1] <= tolerance):
            raise ValueError("Coordinate tolerance admits an ambiguous vertex mapping")
    mapped = values[indices]
    components = (mapped.real, mapped.imag, np.abs(mapped))
    descriptions = (
        "Real part of the complex Helmholtz solution",
        "Imaginary part of the complex Helmholtz solution",
        "Magnitude of the complex Helmholtz solution",
    )
    attached = [
        Field(
            name=name,
            association="vertex",
            dim=1,
            values=value.reshape(-1, 1),
            unit=unit,
            description=description,
        )
        for name, value, description in zip(FIELD_NAMES, components, descriptions)
    ]
    mesh.fields = [
        field for field in mesh.fields if field.name not in FIELD_NAMES
    ] + attached
    return mesh


def result_to_dtcc(result, *, unit=""):
    """Load the input mesh and map its P1 solution without repeating the solve."""
    if result.mesh_path is None:
        raise ValueError("Result has no input mesh path")
    if result.fe_degree != 1:
        raise ValueError("DTCC vertex-field export currently supports FE degree 1 only")
    manifest = json.loads((result.output_dir / "prepared.json").read_text())
    if result.prepared_identity is not None and manifest != result.prepared_identity:
        raise ValueError("Case preparation identity changed since this result was created")
    parameters = manifest["parameters"]
    if (
        parameters["fe_degree"] != result.fe_degree
        or parameters["mesh_name"] != result.mesh_name
    ):
        raise ValueError("Result mesh name or FE degree does not match its prepared case")
    from mpi4py import MPI
    import numpy as np
    from dolfinx import fem, io, mesh as dmesh
    import dtcc_core.io  # noqa: F401 - Registers VolumeMesh.save.
    from dtcc_core.model import VolumeMesh
    from .simulator import _mesh_input_hashes, _runtime_versions

    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("DTCC Helmholtz export requires one MPI rank")
    saved_runtime = dict(manifest["runtime"])
    current_runtime = _runtime_versions()
    # Field export reads geometry and values without constructing a partition.
    saved_runtime.pop("partitioner", None)
    current_runtime.pop("partitioner", None)
    if (
        manifest["inputs"] != _mesh_input_hashes(result.mesh_path)
        or saved_runtime != current_runtime
    ):
        raise ValueError(
            "Input mesh or runtime changed since this solution was prepared"
        )
    with io.XDMFFile(MPI.COMM_WORLD, str(result.mesh_path), "r") as xdmf:
        mesh = xdmf.read_mesh(name=result.mesh_name)
        if mesh.topology.dim != 3 or mesh.geometry.dofmaps[0].shape[1] != 4:
            raise ValueError("DTCC export requires linear tetrahedral mesh geometry")
        dtcc_mesh = VolumeMesh(
            vertices=np.array(mesh.geometry.x, copy=True),
            cells=np.array(mesh.geometry.dofmaps[0], dtype=np.int64, copy=True),
        )
        mesh.topology.create_connectivity(2, 3)
        grids = ET.parse(result.mesh_path).getroot().findall(".//Grid")
        # Use the numerical reader for both mesh and tags, including legacy XDMF.
        for dim, name in (
            (2, parameters.get("facet_tag_name", "boundary_markers")),
            (3, parameters.get("cell_tag_name", "cell_tags")),
        ):
            # Missing optional tags are allowed; malformed existing tags must fail.
            if not any(grid.get("Name") == name for grid in grids):
                continue
            tags = xdmf.read_meshtags(mesh, name=name)
            if dim == 2:
                facets = dmesh.entities_to_geometry(mesh, 2, tags.indices, False)
                dtcc_mesh.boundary_markers = {
                    tuple(int(v) for v in facet): int(value)
                    for facet, value in zip(facets, tags.values)
                }
            else:
                dtcc_mesh.markers = np.zeros(len(dtcc_mesh.cells), dtype=np.int32)
                dtcc_mesh.markers[tags.indices] = tags.values
    space = fem.functionspace(mesh, ("Lagrange", 1))
    coordinates = space.tabulate_dof_coordinates()
    return attach_helmholtz_fields(dtcc_mesh, coordinates, result.solution, unit=unit)


def _checked_export_path(path):
    """Validate an export destination and protect its companion files."""
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() not in (".vtu", ".xdmf", ".dtcc"):
        raise ValueError("Choose .vtu, .xdmf, or .dtcc for a DTCC Helmholtz result")
    outputs = [path]
    if path.suffix.lower() == ".xdmf":
        outputs.extend(
            (
                path.with_suffix(".h5"),
                path.with_name(path.stem + "_boundary_markers.xdmf"),
            )
        )
    if any(output.exists() for output in outputs):
        raise FileExistsError(f"DTCC result export already exists: {path}")
    return path


def save_dtcc_result(result, path=None, *, unit=""):
    """Save a DTCC volume mesh with three fields; never overwrite an existing export."""
    path = _checked_export_path(path or result.output_dir / "helmholtz.vtu")
    mesh = result_to_dtcc(result, unit=unit)
    _checked_export_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(path)
    return path


def load_result(output_dir):
    """Read the complex array, diagnostics, and mesh provenance of a saved case."""
    import numpy as np
    from .simulator import HelmholtzResult

    directory = Path(output_dir).expanduser().resolve()
    manifest = json.loads((directory / "prepared.json").read_text())
    diagnostics = json.loads((directory / "diagnostics.json").read_text())
    paths = [
        Path(name)
        for name in manifest["inputs"]
        if Path(name).suffix.lower() in (".xdmf", ".xmf")
    ]
    if len(paths) != 1:
        raise ValueError("Prepared case must identify exactly one XDMF mesh")
    diagnostics["gmres_history"] = tuple(diagnostics["gmres_history"])
    return HelmholtzResult(
        solution=np.load(directory / "solution.npy", allow_pickle=False),
        output_dir=directory,
        mesh_path=paths[0],
        fe_degree=manifest["parameters"]["fe_degree"],
        mesh_name=manifest["parameters"]["mesh_name"],
        prepared_identity=manifest,
        **diagnostics,
    )
