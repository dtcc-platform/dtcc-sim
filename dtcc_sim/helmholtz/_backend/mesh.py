from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, io, mesh as dmesh
from dolfinx.fem import petsc as fem_petsc
import ufl

from .params import PARAMS


def tprint(msg: str) -> None:
    if MPI.COMM_WORLD.rank == 0:
        print(msg, flush=True)


@dataclass
class MeshData:
    mesh: dmesh.Mesh
    facet_tags: Any | None
    cell_tags: Any | None


def _as_scalar(value: complex | float) -> PETSc.ScalarType:
    return PETSc.ScalarType(value)


def _safe_name(path_str: str) -> str:
    return str(Path(path_str).expanduser().resolve())


def _build_python_mesh() -> MeshData:
    comm = MPI.COMM_WORLD

    if PARAMS.python_mesh_kind == "unit_cube_uniform":
        cell_type = (
            dmesh.CellType.tetrahedron
            if PARAMS.unit_cube_cell_type == "tetrahedron"
            else dmesh.CellType.hexahedron
        )
        mesh = dmesh.create_unit_cube(
            comm,
            int(PARAMS.unit_cube_nx),
            int(PARAMS.unit_cube_ny),
            int(PARAMS.unit_cube_nz),
            cell_type=cell_type,
        )
        return MeshData(mesh=mesh, facet_tags=None, cell_tags=None)

    if PARAMS.python_mesh_kind == "unit_square_uniform":
        cell_type = (
            dmesh.CellType.triangle
            if PARAMS.unit_square_cell_type == "triangle"
            else dmesh.CellType.quadrilateral
        )
        mesh = dmesh.create_unit_square(
            comm,
            int(PARAMS.unit_square_nx),
            int(PARAMS.unit_square_ny),
            cell_type=cell_type,
        )
        return MeshData(mesh=mesh, facet_tags=None, cell_tags=None)

    raise ValueError(f"Unsupported python_mesh_kind: {PARAMS.python_mesh_kind!r}")


def _read_external_xdmf() -> MeshData:
    mesh_path = Path(PARAMS.mesh_path).expanduser().resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    comm = MPI.COMM_WORLD
    facet_tags = None
    cell_tags = None

    with io.XDMFFile(comm, str(mesh_path), "r") as xdmf:
        mesh = xdmf.read_mesh(name=PARAMS.mesh_name)

        tdim = mesh.topology.dim
        fdim = tdim - 1
        mesh.topology.create_entities(fdim)
        mesh.topology.create_connectivity(tdim, fdim)
        mesh.topology.create_connectivity(fdim, tdim)

        try:
            cell_tags = xdmf.read_meshtags(mesh, name=PARAMS.cell_tag_name)
        except Exception:
            cell_tags = None

        try:
            facet_tags = xdmf.read_meshtags(mesh, name=PARAMS.facet_tag_name)
        except Exception:
            facet_tags = None

    return MeshData(mesh=mesh, facet_tags=facet_tags, cell_tags=cell_tags)


def load_mesh() -> tuple[dmesh.Mesh, Any | None, Any | None]:
    if PARAMS.mesh_source == "python_builder":
        out = _build_python_mesh()
    elif PARAMS.mesh_source == "external_file":
        out = _read_external_xdmf()
    else:
        raise ValueError(f"Unsupported mesh_source: {PARAMS.mesh_source!r}")

    mesh = out.mesh
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_entities(fdim)
    mesh.topology.create_connectivity(tdim, fdim)
    mesh.topology.create_connectivity(fdim, tdim)

    if PARAMS.verbose:
        gdim = mesh.geometry.dim
        n_cells = mesh.topology.index_map(tdim).size_local
        n_facets = mesh.topology.index_map(fdim).size_local
        tprint(
            f"[mesh] source={PARAMS.mesh_source} "
            f"gdim={gdim} tdim={tdim} local_cells={n_cells} local_facets={n_facets}"
        )
        if PARAMS.mesh_source == "external_file":
            tprint(f"[mesh] file={_safe_name(PARAMS.mesh_path)}")
        tprint(f"[mesh] facet_tags={'yes' if out.facet_tags is not None else 'no'}")
        tprint(f"[mesh] cell_tags={'yes' if out.cell_tags is not None else 'no'}")

    return out.mesh, out.facet_tags, out.cell_tags


def mesh_bounds(mesh: dmesh.Mesh) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(mesh.geometry.x, dtype=np.float64)
    mins = x.min(axis=0)
    maxs = x.max(axis=0)
    return mins, maxs


def build_function_space(mesh: dmesh.Mesh) -> fem.FunctionSpace:
    if PARAMS.fe_family != "Lagrange":
        raise ValueError(f"Unsupported FE family: {PARAMS.fe_family!r}")
    return fem.functionspace(mesh, (PARAMS.fe_family, int(PARAMS.fe_degree)))


def _build_piecewise_constant_cell_function(
    mesh: dmesh.Mesh,
    cell_tags: Any | None,
    values_by_tag: dict[int, float],
    default_value: float,
    *,
    name: str,
) -> fem.Function:
    Q = fem.functionspace(mesh, ("DG", 0))
    coeff = fem.Function(Q, name=name)
    coeff.x.array[:] = _as_scalar(default_value)

    if cell_tags is None:
        return coeff

    local_cells = np.asarray(cell_tags.indices, dtype=np.int32)
    local_values = np.asarray(cell_tags.values, dtype=np.int32)

    for tag, val in values_by_tag.items():
        hits = local_cells[local_values == int(tag)]
        for c in hits.tolist():
            dof = Q.dofmap.cell_dofs(int(c))[0]
            coeff.x.array[dof] = _as_scalar(float(val))

    return coeff


def _build_piecewise_constant_facet_function(
    mesh: dmesh.Mesh,
    facet_tags: Any | None,
    values_by_tag: dict[int, float],
    default_value: float,
    *,
    name: str,
) -> fem.Function:
    Vc = fem.functionspace(mesh, ("CG", 1))
    coeff = fem.Function(Vc, name=name)
    coeff.x.array[:] = _as_scalar(default_value)

    if facet_tags is None:
        return coeff

    fdim = mesh.topology.dim - 1
    for tag, val in values_by_tag.items():
        facets = np.asarray(facet_tags.find(int(tag)), dtype=np.int32)
        if facets.size == 0:
            continue
        dofs = fem.locate_dofs_topological(Vc, fdim, facets)
        coeff.x.array[np.asarray(dofs, dtype=np.int32)] = _as_scalar(float(val))

    return coeff


def build_coefficients(
    mesh: dmesh.Mesh,
    facet_tags: Any | None,
    cell_tags: Any | None,
):
    mode = PARAMS.coefficient_mode

    if mode == "constant":
        Acoef = fem.Constant(mesh, _as_scalar(float(PARAMS.A_const)))
        Vcoef = fem.Constant(mesh, _as_scalar(float(PARAMS.V_const)))
        beta = fem.Constant(mesh, _as_scalar(float(PARAMS.beta_const)))
        return Acoef, Vcoef, beta

    if mode == "cell_tags":
        Acoef = _build_piecewise_constant_cell_function(
            mesh, cell_tags, PARAMS.A_by_cell_tag, float(PARAMS.A_const), name="Acoef"
        )
        Vcoef = _build_piecewise_constant_cell_function(
            mesh, cell_tags, PARAMS.V_by_cell_tag, float(PARAMS.V_const), name="Vcoef"
        )
        beta = _build_piecewise_constant_facet_function(
            mesh, facet_tags, PARAMS.beta_by_facet_tag, float(PARAMS.beta_const), name="beta"
        )
        return Acoef, Vcoef, beta

    if mode == "functions":
        raise NotImplementedError(
            "coefficient_mode='functions' is reserved for explicit city coefficient fields."
        )

    raise ValueError(f"Unsupported coefficient_mode: {mode!r}")


def _remap_requested_facet_tags(
    mesh: dmesh.Mesh,
    facet_tags: Any,
    requested_tags: tuple[int, ...],
):
    fdim = mesh.topology.dim - 1

    all_facets = []
    all_vals = []
    tag_to_local = {}

    for j, tag in enumerate(requested_tags, start=1):
        facets = np.asarray(facet_tags.find(int(tag)), dtype=np.int32)
        if facets.size == 0:
            continue
        all_facets.append(facets)
        all_vals.append(np.full(facets.size, j, dtype=np.int32))
        tag_to_local[int(tag)] = j

    if len(all_facets) == 0:
        return None, {}

    facets_arr = np.concatenate(all_facets)
    vals_arr = np.concatenate(all_vals)

    order = np.argsort(facets_arr)
    facets_arr = facets_arr[order]
    vals_arr = vals_arr[order]

    mt = dmesh.meshtags(mesh, fdim, facets_arr, vals_arr)
    return mt, tag_to_local


def _impedance_boundary_measures(mesh, facet_tags):
    if PARAMS.boundary_mode not in ("impedance_only", "mixed"):
        raise ValueError(f"Unsupported boundary_mode: {PARAMS.boundary_mode!r}")
    if facet_tags is None:
        if PARAMS.boundary_mode == "mixed":
            raise RuntimeError("Mixed boundary mode requires facet tags")
        return (ufl.ds(domain=mesh),)
    remapped, tag_to_local = _remap_requested_facet_tags(
        mesh, facet_tags, tuple(int(t) for t in PARAMS.impedance_facet_tags)
    )
    if remapped is None:
        return ()
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=remapped)
    return tuple(ds(tag) for tag in tag_to_local.values())


def _global_bilinear_form(mesh, V, facet_tags, Acoef, Vcoef, beta):
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    k = float(PARAMS.k())
    dx = ufl.dx(domain=mesh)
    a = (
        ufl.inner(Acoef * ufl.grad(u), ufl.grad(v)) * dx
        - (k**2) * ufl.inner((Vcoef**2) * u, v) * dx
    )
    for measure in _impedance_boundary_measures(mesh, facet_tags):
        a += (-1j * k) * ufl.inner(beta * u, v) * measure
    return a


def assemble_global_matrix(
    mesh: dmesh.Mesh,
    V: fem.FunctionSpace,
    facet_tags: Any | None,
    Acoef,
    Vcoef,
    beta,
    bcs: list[Any] | None = None,
) -> PETSc.Mat:
    a = _global_bilinear_form(mesh, V, facet_tags, Acoef, Vcoef, beta)
    A = fem_petsc.assemble_matrix(fem.form(a), bcs=[] if bcs is None else bcs)
    A.assemble()
    return A


def supervisor_source_point(mesh: dmesh.Mesh) -> np.ndarray:
    mins, maxs = mesh_bounds(mesh)
    center_xy = 0.5 * (mins[:2] + maxs[:2])

    zmin = mins[2] if mesh.geometry.dim >= 3 else 0.0
    zmax = maxs[2] if mesh.geometry.dim >= 3 else 0.0
    z0 = PARAMS.source_z_weight_min * zmin + PARAMS.source_z_weight_max * zmax

    if mesh.geometry.dim >= 3:
        return np.array([center_xy[0], center_xy[1], z0], dtype=np.float64)
    return np.array([center_xy[0], center_xy[1]], dtype=np.float64)


def source_point(mesh: dmesh.Mesh) -> np.ndarray:
    if PARAMS.source_location_mode == "manual":
        if mesh.geometry.dim >= 3:
            return np.array(
                [float(PARAMS.source_x), float(PARAMS.source_y), float(PARAMS.source_z)],
                dtype=np.float64,
            )
        return np.array(
            [float(PARAMS.source_x), float(PARAMS.source_y)],
            dtype=np.float64,
        )

    if PARAMS.source_location_mode == "mesh_bounds":
        return supervisor_source_point(mesh)

    raise ValueError(f"Unsupported source_location_mode: {PARAMS.source_location_mode!r}")


def _warn_if_source_outside(mesh: dmesh.Mesh, point: np.ndarray) -> None:
    """Warn about an exterior Gaussian center without changing the forcing."""
    from dolfinx import geometry

    coordinates = np.zeros((1, 3), dtype=mesh.geometry.x.dtype)
    coordinates[0, :len(point)] = point
    tree = geometry.bb_tree(mesh, mesh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, coordinates)
    cells = geometry.compute_colliding_cells(mesh, candidates, coordinates)
    inside = mesh.comm.allreduce(bool(len(cells.links(0))), op=MPI.LOR)
    if not inside:
        warnings.warn(
            f"Gaussian source centre {point.tolist()} is outside the computational mesh "
            f"(source_location_mode={PARAMS.source_location_mode!r}). "
            "The source position is unchanged; the finite-width Gaussian may still "
            "overlap the domain. Check source placement when interpreting this run.",
            RuntimeWarning,
            stacklevel=2,
        )


def _gaussian_source_expression(mesh: dmesh.Mesh):
    x = ufl.SpatialCoordinate(mesh)
    x0 = source_point(mesh)
    _warn_if_source_outside(mesh, x0)
    sigma = float(PARAMS.gaussian_source_sigma)
    amp = complex(PARAMS.source_amplitude_real, PARAMS.source_amplitude_imag)

    r2 = sum((x[i] - float(x0[i])) ** 2 for i in range(mesh.geometry.dim))
    return _as_scalar(amp) * ufl.exp(-r2 / (2.0 * sigma**2))


def _plane_wave_direction(mesh):
    d = np.array(
        [
            float(PARAMS.plane_wave_direction_x),
            float(PARAMS.plane_wave_direction_y),
            float(PARAMS.plane_wave_direction_z),
        ],
        dtype=np.float64,
    )
    d = d[: mesh.geometry.dim]
    nrm = np.linalg.norm(d)
    if nrm <= 0.0:
        raise ValueError("Plane-wave direction must be nonzero")
    return d / nrm


def _plane_wave_expression(mesh):
    x = ufl.SpatialCoordinate(mesh)
    d = _plane_wave_direction(mesh)
    phase = sum(float(d[j]) * x[j] for j in range(mesh.geometry.dim))
    return ufl.exp(1j * float(PARAMS.k()) * phase)


def build_rhs(mesh: dmesh.Mesh, V: fem.FunctionSpace, facet_tags: Any | None, bcs=None) -> PETSc.Vec:
    v = ufl.TestFunction(V)
    dx = ufl.dx(domain=mesh)

    rhs_mode = PARAMS.rhs_mode

    if rhs_mode in ("gaussian_point_source", "point_source"):
        f_expr = _gaussian_source_expression(mesh)
        L = -ufl.inner(f_expr, v) * dx

    elif rhs_mode == "plane_wave_exact":
        if PARAMS.coefficient_mode != "constant":
            raise ValueError("plane_wave_exact requires constant coefficients")
        u_exact = _plane_wave_expression(mesh)
        Acoef, Vcoef, beta = build_coefficients(mesh, facet_tags, None)
        k = float(PARAMS.k())
        f_expr = -ufl.div(Acoef * ufl.grad(u_exact)) - (k**2) * (Vcoef**2) * u_exact
        # Manufactured data include natural flux everywhere and the Robin
        # contribution on exactly the facets used by the bilinear form.
        metadata = {"quadrature_degree": max(8, 2 * int(PARAMS.fe_degree) + 4)}
        dx_exact = ufl.Measure("dx", domain=mesh, metadata=metadata)
        ds_exact = ufl.Measure("ds", domain=mesh, metadata=metadata)
        normal = ufl.FacetNormal(mesh)
        L = ufl.inner(f_expr, v) * dx_exact
        L += ufl.inner(ufl.dot(Acoef * ufl.grad(u_exact), normal), v) * ds_exact
        for measure in _impedance_boundary_measures(mesh, facet_tags):
            L += (-1j * k) * ufl.inner(beta * u_exact, v) * measure(metadata=metadata)

    else:
        raise ValueError(f"Unsupported rhs_mode: {rhs_mode!r}")

    b = fem_petsc.assemble_vector(fem.form(L))
    if rhs_mode == "plane_wave_exact" and bcs:
        a = _global_bilinear_form(mesh, V, facet_tags, Acoef, Vcoef, beta)
        fem_petsc.apply_lifting(b, [fem.form(a)], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return b


def build_dirichlet_bcs_if_needed(
    mesh: dmesh.Mesh,
    V: fem.FunctionSpace,
    facet_tags: Any | None,
) -> list[Any]:
    bcs: list[Any] = []

    if PARAMS.boundary_mode == "mixed":
        tags = tuple(int(t) for t in PARAMS.dirichlet_facet_tags)
        if len(tags) > 0:
            if facet_tags is None:
                raise RuntimeError("Dirichlet facet tags requested, but facet_tags is missing")

            fdim = mesh.topology.dim - 1
            facets = []
            for tag in tags:
                facets.append(np.asarray(facet_tags.find(tag), dtype=np.int32))
            facets_arr = np.unique(np.concatenate(facets)) if facets else np.zeros(0, dtype=np.int32)

            if facets_arr.size > 0:
                dofs = fem.locate_dofs_topological(V, fdim, facets_arr)
                value = fem.Function(V)
                if PARAMS.rhs_mode == "plane_wave_exact":
                    direction = _plane_wave_direction(mesh)
                    k = float(PARAMS.k())
                    value.interpolate(lambda x: np.exp(1j * k * (direction @ x[:len(direction)])))
                else:
                    value.x.array[:] = 0.0
                bcs.append(fem.dirichletbc(value, dofs))

    if PARAMS.use_anchor_point:
        tprint("[mesh] WARNING: use_anchor_point=True requested, but anchor application is not implemented yet.")

    return bcs


def build_system():
    mesh, facet_tags, cell_tags = load_mesh()
    V = build_function_space(mesh)
    Acoef, Vcoef, beta = build_coefficients(mesh, facet_tags, cell_tags)
    bcs = build_dirichlet_bcs_if_needed(mesh, V, facet_tags)
    A = assemble_global_matrix(mesh, V, facet_tags, Acoef, Vcoef, beta, bcs=bcs)
    b = build_rhs(mesh, V, facet_tags, bcs=bcs)
    if len(bcs) > 0:
        fem_petsc.set_bc(b, bcs)

    if PARAMS.verbose:
        n = V.dofmap.index_map.size_local
        mins, maxs = mesh_bounds(mesh)
        tprint(f"[mesh] FE space dofs N = {n}")
        tprint(f"[mesh] coefficient_mode = {PARAMS.coefficient_mode}")
        tprint(f"[mesh] rhs_mode = {PARAMS.rhs_mode}")
        tprint(f"[mesh] bounds min = {mins}")
        tprint(f"[mesh] bounds max = {maxs}")
        if PARAMS.rhs_mode == "plane_wave_exact":
            tprint(f"[mesh] plane wave direction = {_plane_wave_direction(mesh)}")
        else:
            tprint(f"[mesh] source x0 = {source_point(mesh)}")
        tprint(f"[mesh] k          = {float(PARAMS.k()):.6e}")

    return mesh, facet_tags, cell_tags, V, Acoef, Vcoef, beta, A, b
