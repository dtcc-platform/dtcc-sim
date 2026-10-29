"""Shared output-directory handling for simulation APIs."""

from pathlib import Path
import xml.etree.ElementTree as ET


def _output_directory(directory):
    """Validate a supplied directory without creating it during configuration."""
    if directory is None:
        return None
    if not str(directory).strip():
        raise ValueError("output_dir must be a nonempty directory path")
    path = Path(directory).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"output_dir is not a directory: {path}")
    return path


def _solution_path(directory, explicit_path, filename, *, input_mesh=None, stems=("",)):
    """Resolve an optional export, creating parents and replacing prior outputs."""
    if directory is not None and explicit_path is not None:
        raise ValueError("Specify output_dir or output_path, not both")
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
    elif directory is not None:
        path = directory / filename
    else:
        return None
    if input_mesh is not None:
        mesh_path = Path(input_mesh).expanduser().resolve()
        inputs = {mesh_path}
        for item in ET.parse(mesh_path).iter():
            if item.tag.split("}")[-1] == "DataItem" and item.get("Format") == "HDF":
                reference = (item.text or "").strip().split(":", 1)[0]
                if not reference:
                    raise ValueError(f"Empty HDF5 reference in input mesh: {mesh_path}")
                inputs.add((mesh_path.parent / reference).resolve())
        outputs = set()
        for stem in stems:
            target = path.with_name(path.stem + stem + path.suffix)
            outputs.add(target)
            if target.suffix.lower() in (".xdmf", ".xmf"):
                outputs.add(target.with_suffix(".h5"))
        if inputs.intersection(outputs):
            raise ValueError("Solution output would overwrite an input mesh file")
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _save_city_surface(mesh, path):
    """Export the domain's ground and building facets in input coordinates."""
    from copy import deepcopy
    import numpy as np
    from dtcc_core.model import Mesh

    if isinstance(mesh.boundary_markers, dict):
        faces = np.asarray(list(mesh.boundary_markers), dtype=np.int64)
        markers = np.asarray(list(mesh.boundary_markers.values()))
    else:
        faces = np.asarray(mesh.boundary_faces)
        markers = np.asarray(mesh.boundary_markers)
    if faces.ndim != 2 or faces.shape[1] != 3 or markers.shape != (len(faces),):
        raise ValueError("city.vtu requires tagged triangular boundary faces")
    if not np.issubdtype(faces.dtype, np.integer) or not np.issubdtype(
        markers.dtype, np.integer
    ):
        raise ValueError("City boundary faces and markers must be integers")
    if not len(faces) or faces.min() < 0 or faces.max() >= len(mesh.vertices):
        raise ValueError("City boundary faces must reference valid mesh vertices")
    if np.any(markers < -6):
        raise ValueError("City boundary markers must use the DTCC city mesh convention")
    # Ground is -1; building facets are nonnegative. -2 through -6 are artificial boundaries.
    keep = (markers == -1) | (markers >= 0)
    if not keep.any():
        raise ValueError("City mesh has no tagged ground or building faces to export")
    vertices, indices = np.unique(faces[keep], return_inverse=True)
    surface = Mesh(
        vertices=np.array(mesh.vertices[vertices], copy=True),
        faces=indices.reshape(-1, 3),
        markers=np.array(markers[keep], copy=True),
    )
    surface.transform = deepcopy(mesh.transform)
    surface.save(path)


def _boundary_payload(mesh, markers):
    """Collect owned physical boundary triangles with global geometry indices."""
    import numpy as np
    import dolfinx.mesh as dm

    if markers is None:
        return None
    dimension = mesh.topology.dim - 1
    mesh.topology.create_entities(dimension)
    mesh.topology.create_connectivity(dimension, mesh.topology.dim)
    exterior = dm.exterior_facet_indices(mesh.topology)
    keep = ((markers.values == -1) | (markers.values >= 0)) & np.isin(
        markers.indices, exterior
    )
    facets = markers.indices[keep]
    if not len(facets):
        return None
    geometry = dm.entities_to_geometry(mesh, dimension, facets)
    if geometry.ndim != 2 or geometry.shape[1] != 3:
        raise ValueError(
            "City surface export requires triangular facets with linear geometry"
        )
    local_vertices, inverse = np.unique(geometry, return_inverse=True)
    global_vertices = mesh.geometry.index_map().local_to_global(
        local_vertices.astype(np.int32)
    )
    return (
        global_vertices,
        np.array(mesh.geometry.x[local_vertices], copy=True),
        global_vertices[inverse].reshape(-1, 3),
        np.array(markers.values[keep], copy=True),
    )


def _merge_boundary_payloads(payloads):
    """Build a native boundary-only volume-mesh view from distributed facets."""
    import numpy as np
    from dtcc_core.model import VolumeMesh

    payloads = [payload for payload in payloads if payload is not None]
    if not payloads:
        return None
    ids, first = np.unique(
        np.concatenate([item[0] for item in payloads]), return_index=True
    )
    vertices = np.concatenate([item[1] for item in payloads])[first]
    faces = np.searchsorted(ids, np.concatenate([item[2] for item in payloads]))
    markers = np.concatenate([item[3] for item in payloads])
    return VolumeMesh(vertices=vertices, boundary_faces=faces, boundary_markers=markers)


def _export_city_surface(
    output_path,
    *,
    native_mesh=None,
    mesh=None,
    markers=None,
    mesh_path=None,
    frame=None,
):
    """Write physical city geometry beside a solution, collectively for MPI meshes."""
    from .timing import phase

    if output_path is None:
        return None
    with phase("city_surface"):
        return _export_city_surface_impl(
            output_path,
            native_mesh=native_mesh,
            mesh=mesh,
            markers=markers,
            mesh_path=mesh_path,
            frame=frame,
        )


def _export_city_surface_impl(
    output_path, *, native_mesh, mesh, markers, mesh_path, frame
):
    import numpy as np
    from .coordinates import write_coordinate_frame

    comm = getattr(mesh, "comm", None)
    rank = comm.rank if comm is not None else 0
    path = Path(output_path).parent / "city.vtu"
    if path == Path(output_path):
        path = path.with_name("city_surface.vtu")
    failure = None
    native = native_mesh if rank == 0 else None
    if rank == 0 and native is None and markers is None and mesh_path is not None:
        try:
            from dtcc_core.io import load_volume_mesh

            native = load_volume_mesh(mesh_path)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
    use_native = native is not None or failure is not None
    if comm is not None:
        use_native = comm.bcast(use_native, root=0)
    if not use_native:
        payload, local_failure = None, None
        try:
            if mesh is not None:
                payload = _boundary_payload(mesh, markers)
        except Exception as exc:
            local_failure = f"{type(exc).__name__}: {exc}"
        gathered = (
            comm.gather((payload, local_failure), root=0)
            if comm is not None
            else [(payload, local_failure)]
        )
        if rank == 0:
            failure = next((error for _, error in gathered if error is not None), None)
            if failure is None:
                try:
                    native = _merge_boundary_payloads([item[0] for item in gathered])
                except Exception as exc:
                    failure = f"{type(exc).__name__}: {exc}"
    exported = False
    if rank == 0 and failure is None:
        try:
            tags = getattr(native, "boundary_markers", ())
            if tags is None:
                tags = ()
            values = np.asarray(list(tags.values()) if isinstance(tags, dict) else tags)
            if values.size and np.any((values == -1) | (values >= 0)):
                _save_city_surface(native, path)
                if frame is not None:
                    write_coordinate_frame(path, frame)
                exported = True
            else:
                path.unlink(missing_ok=True)
                path.with_suffix(".coordinates.json").unlink(missing_ok=True)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
    if comm is not None:
        failure, exported = comm.bcast((failure, exported), root=0)
    if failure is not None:
        raise RuntimeError(f"Could not export city surface: {failure}")
    return path if exported else None
