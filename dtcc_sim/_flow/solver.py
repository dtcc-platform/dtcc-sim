"""Stationary incompressible Taylor–Hood flow with kinematic pressure.

Velocity, pressure and speed are exported in m/s, m²/s² and m/s.
Navier–Stokes retains full convection and the explicitly supplied viscosity.
"""

from __future__ import annotations

from ..timing import phase as timing_phase, timed, timed_run
from ..meshing import prepare_city_mesh, publish_mesh_provenance
from ..output import RankZeroResult, _output_directory, _solution_path, _export_city_surface
from ..coordinates import input_crs, write_coordinate_frame
from .parameters import KAPPA, _FlowParameters

from enum import IntEnum
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

import dolfinx
import dolfinx.fem
import dolfinx.mesh
from dolfinx.fem import (
    Constant,
    Function,
    dirichletbc,
    form as _fem_form,
    locate_dofs_topological,
)
from dolfinx.fem import petsc as _fem_petsc
from mpi4py import MPI
from petsc4py import PETSc
import basix.ufl
import ufl
from ufl import (
    FacetNormal,
    Measure,
    dot,
    inner,
    grad,
    div,
    dx,
    ds,
)

from dtcc_sim.fenics import _mesh_from_dtcc
from dtcc_sim.fenics import (
    FunctionSpace,
    _ensure_parent_dir,
    info,
    warning,
    load_mesh_with_markers,
    _local_mesh_coordinates,
)


# ---------------------------------------------------------------------------
# Boundary categories
# ---------------------------------------------------------------------------


class BndCat(IntEnum):
    """Collapsed boundary categories for the urban wind domain."""

    WALL = 1
    ROOF = 2
    GROUND = 3
    INLET = 4
    OUTLET = 5
    TOP = 6
    SIDE = 7  # lateral bbox faces that are neither inlet nor outlet


# Canonical dtcc bbox marker mapping.
BBOX_MARKER_NORMALS: Dict[int, Tuple[float, float, float]] = {
    -3: (-1.0, 0.0, 0.0),  # xmin face
    -4: (1.0, 0.0, 0.0),  # xmax face
    -5: (0.0, -1.0, 0.0),  # ymin face
    -6: (0.0, 1.0, 0.0),  # ymax face
    -2: (0.0, 0.0, 1.0),  # top face
}


DEFAULT_STATIONARY_PETSC = {
    "ksp_type": "fgmres", "ksp_rtol": 1e-10, "ksp_atol": 1e-13,
    "ksp_max_it": 500, "ksp_gmres_restart": 100,
    "pc_type": "fieldsplit", "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "lower",
    "pc_fieldsplit_schur_precondition": "selfp", "pc_use_amat": True,
    "fieldsplit_u_ksp_type": "preonly", "fieldsplit_u_pc_type": "asm",
    "fieldsplit_u_pc_asm_overlap": 1,
    "fieldsplit_u_sub_ksp_type": "preonly", "fieldsplit_u_sub_pc_type": "ilu",
    "fieldsplit_u_sub_pc_factor_levels": 0,
    "fieldsplit_p_ksp_type": "preonly", "fieldsplit_p_pc_type": "hypre",
    "fieldsplit_p_pc_hypre_type": "boomeramg",
}


VARIABLE_VISCOSITY_PETSC = {
    "fieldsplit_u_sub_pc_factor_levels": 1,
    "fieldsplit_u_sub_pc_factor_mat_ordering_type": "rcm",
}
"""Velocity-block ILU(1) for spatially varying eddy viscosity.

Reverse Cuthill–McKee ordering of each local block keeps the ILU(1) fill, and
with it the factorization and triangular-solve cost, far below the natural
ordering of the mixed space.
"""


def _global_max_positive_marker(
    mesh: dolfinx.mesh.Mesh, markers: dolfinx.mesh.MeshTags
) -> int:
    vals = np.asarray(markers.values, dtype=np.int32)
    local_max = int(vals[vals >= 0].max()) if np.any(vals >= 0) else -1
    return int(mesh.comm.allreduce(local_max, op=MPI.MAX))


def infer_num_buildings(mesh: dolfinx.mesh.Mesh, markers: dolfinx.mesh.MeshTags) -> int:
    """Marker scheme: walls 0..N-1, roofs N..2N-1."""
    max_pos = _global_max_positive_marker(mesh, markers)
    if max_pos < 0:
        return 0
    return (max_pos + 1) // 2


def categorize_boundary(
    mesh: dolfinx.mesh.Mesh,
    markers: dolfinx.mesh.MeshTags,
    num_buildings: int,
    inlet_marker,
    outlet_marker,
    ground_tag: int = -1,
    top_tag: int = -2,
) -> dolfinx.mesh.MeshTags:
    """Collapse per-building markers into wind-specific categories.

    ``inlet_marker`` and ``outlet_marker`` are one bbox marker or a sequence
    of markers; ``outlet_marker`` may be None. Returns MeshTags with values
    from :class:`BndCat`.
    """
    fdim = mesh.topology.dim - 1
    idx = np.asarray(markers.indices, dtype=np.int32)
    vals = np.asarray(markers.values, dtype=np.int32)

    N = int(num_buildings)
    cat = np.empty_like(vals, dtype=np.int32)

    mask_wall = (vals >= 0) & (vals < N)
    mask_roof = (vals >= N) & (vals < 2 * N)
    mask_ground = vals == ground_tag
    mask_top = vals == top_tag
    mask_inlet = np.isin(vals, _marker_list(inlet_marker))
    mask_outlet = np.isin(vals, _marker_list(outlet_marker))

    # Remaining bbox faces (not inlet, outlet or top)
    mask_side = (vals < ground_tag) & ~mask_top & ~mask_inlet & ~mask_outlet

    cat[mask_wall] = int(BndCat.WALL)
    cat[mask_roof] = int(BndCat.ROOF)
    cat[mask_ground] = int(BndCat.GROUND)
    cat[mask_top] = int(BndCat.TOP)
    cat[mask_side] = int(BndCat.SIDE)
    # Inlet/outlet must come last so explicit overrides take priority
    cat[mask_inlet] = int(BndCat.INLET)
    cat[mask_outlet] = int(BndCat.OUTLET)

    # catch-all
    mask_other = ~(
        mask_wall
        | mask_roof
        | mask_ground
        | mask_top
        | mask_inlet
        | mask_outlet
        | mask_side
    )
    cat[mask_other] = int(BndCat.SIDE)

    order = np.argsort(idx)
    return dolfinx.mesh.meshtags(mesh, fdim, idx[order], cat[order])


# ---------------------------------------------------------------------------
# Helpers — inlet / outlet selection
# ---------------------------------------------------------------------------


def _marker_list(markers) -> list:
    """One marker, a sequence of markers or None as a list of ints."""
    if markers is None:
        return []
    if np.ndim(markers) == 0:
        return [int(markers)]
    return [int(marker) for marker in markers]


LATERAL_FLOW_TOLERANCE = 1e-8
"""Faces with |wind · n| below this are parallel to the wind."""


def classify_lateral_faces(params: _FlowParameters) -> Tuple[list, list]:
    """Inflow and outflow lateral bbox markers for the wind direction.

    Every lateral face the wind enters through receives the inflow profile
    and every face it leaves through is a traction outlet, so oblique winds
    keep their crosswind component. Faces parallel to the wind stay side
    faces. Explicit markers or a closed cavity select single faces as in
    :func:`select_inlet_outlet`.
    """
    if params.closed_cavity or params.inlet_marker is not None or params.outlet_marker is not None:
        inlet, outlet = select_inlet_outlet(params)
        return [inlet], _marker_list(outlet)
    wind = np.array(params.wind_vector_xy)
    inlets, outlets = [], []
    for marker, normal in BBOX_MARKER_NORMALS.items():
        if abs(normal[2]) > .5:
            continue
        flux = float(np.dot(normal[:2], wind))
        if flux < -LATERAL_FLOW_TOLERANCE:
            inlets.append(marker)
        elif flux > LATERAL_FLOW_TOLERANCE:
            outlets.append(marker)
    return inlets, outlets


def select_inlet_outlet(
    params: _FlowParameters,
    mesh: Optional[dolfinx.mesh.Mesh] = None,
    markers: Optional[dolfinx.mesh.MeshTags] = None,
) -> Tuple[int, Optional[int]]:
    """Choose inlet and outlet bbox markers from wind direction.

    Returns
    -------
    (inlet_marker, outlet_marker)  — outlet_marker is None for closed cavities.
    """
    if params.inlet_marker is not None and params.outlet_marker is not None:
        return params.inlet_marker, params.outlet_marker

    # Closed cavity: no outlet
    if params.closed_cavity:
        outlet_override: Optional[int] = None
    else:
        outlet_override = params.outlet_marker  # could be None → auto-detect

    _ = mesh, markers

    marker_normals = {
        marker: normal
        for marker, normal in BBOX_MARKER_NORMALS.items()
        if abs(normal[2]) <= 0.5
    }

    wx, wy = params.wind_vector_xy
    w = np.array([wx, wy])

    # Only consider lateral faces.
    best_inlet_marker = -3
    best_outlet_marker = -3
    best_inlet_dot = 0.0  # most negative dot → inlet
    best_outlet_dot = 0.0  # most positive dot → outlet

    first = True
    for marker, (nx, ny, nz) in marker_normals.items():
        if abs(nz) > 0.5:
            continue
        n_xy = np.array([nx, ny])
        d = float(np.dot(n_xy, w))
        if first:
            best_inlet_dot = d
            best_outlet_dot = d
            best_inlet_marker = marker
            best_outlet_marker = marker
            first = False
        else:
            if d < best_inlet_dot:
                best_inlet_dot = d
                best_inlet_marker = marker
            if d > best_outlet_dot:
                best_outlet_dot = d
                best_outlet_marker = marker

    inlet = (
        params.inlet_marker if params.inlet_marker is not None else best_inlet_marker
    )
    outlet: Optional[int]
    if params.closed_cavity:
        outlet = None
    elif outlet_override is not None:
        outlet = outlet_override
    else:
        outlet = best_outlet_marker

    return inlet, outlet


# ---------------------------------------------------------------------------
# Helpers — marker validation
# ---------------------------------------------------------------------------


def _validate_bbox_markers(
    mesh: dolfinx.mesh.Mesh,
    markers: dolfinx.mesh.MeshTags,
    inlet_marker,
    outlet_marker,
    has_outlet: bool,
) -> None:
    """Validate that expected boundary markers exist in the mesh tags.

    Raises ``RuntimeError`` with a helpful message when required markers
    are missing.
    """
    available = set(int(v) for v in markers.values)
    # Gather globally (some markers may only appear on a subset of ranks)
    all_markers_local = np.array(sorted(available), dtype=np.int32)
    all_counts = mesh.comm.allgather(all_markers_local)
    global_available = set()
    for arr in all_counts:
        global_available.update(int(v) for v in arr)

    for marker in _marker_list(inlet_marker):
        if marker not in global_available:
            raise RuntimeError(
                f"Inlet marker {marker} not found in mesh tags.  "
                f"Available markers: {sorted(global_available)}.  "
                f"Expected bbox markers: {sorted(BBOX_MARKER_NORMALS.keys())}."
            )

    for marker in _marker_list(outlet_marker) if has_outlet else []:
        if marker not in global_available:
            raise RuntimeError(
                f"Outlet marker {marker} not found in mesh tags.  "
                f"Available markers: {sorted(global_available)}.  "
                f"Set closed_cavity=True if no outlet face exists."
            )

    # Warn if none of the expected bbox markers are present
    expected_bbox = set(BBOX_MARKER_NORMALS.keys())
    if not expected_bbox.intersection(global_available):
        warning(
            f"None of the expected bounding-box markers "
            f"{sorted(expected_bbox)} found in mesh tags "
            f"{sorted(global_available)}.  Inlet/outlet selection may be wrong."
        )


# ---------------------------------------------------------------------------
# Helpers — inlet velocity profile
# ---------------------------------------------------------------------------


def make_inlet_velocity_expression(
    params: _FlowParameters,
) -> callable:
    """Return a callable ``f(x) -> (3, N)`` for dolfinx interpolation.

    ``x`` has shape ``(gdim, N)``; ``z = x[2]``.
    """
    wx, wy = params.wind_vector_xy
    U_ref = params.wind_speed
    profile = params.inlet_profile

    if profile == "uniform":

        def _expr(x: np.ndarray) -> np.ndarray:
            n = x.shape[1]
            vals = np.zeros((3, n), dtype=np.float64)
            vals[0, :] = U_ref * wx
            vals[1, :] = U_ref * wy
            return vals

    elif profile == "power_law":
        alpha = params.power_law_alpha
        z_ref = params.u_ref_height

        def _expr(x: np.ndarray) -> np.ndarray:
            n = x.shape[1]
            z = np.maximum(x[2], 0.0)
            mag = U_ref * (z / z_ref) ** alpha
            vals = np.zeros((3, n), dtype=np.float64)
            vals[0, :] = mag * wx
            vals[1, :] = mag * wy
            return vals

    elif profile == "log_law":
        z0 = params.z0
        zw = params.wall_offset
        z_ref = params.u_ref_height
        denom = np.log((z_ref + zw) / z0)

        def _expr(x: np.ndarray) -> np.ndarray:
            n = x.shape[1]
            z = np.maximum(x[2], 0.0)
            mag = U_ref * np.log((z + zw) / z0) / denom
            vals = np.zeros((3, n), dtype=np.float64)
            vals[0, :] = mag * wx
            vals[1, :] = mag * wy
            return vals

    else:
        raise ValueError(f"Unknown inlet profile: {profile}")

    return _expr


# ---------------------------------------------------------------------------
# Helpers — output conversion
# ---------------------------------------------------------------------------


def _owned_vertex_values(mesh: dolfinx.mesh.Mesh, function: Function) -> np.ndarray:
    """Values of ``function`` at the geometry nodes owned by this rank."""
    owned = mesh.geometry.index_map().size_local
    size = int(np.prod(function.ufl_shape)) if function.ufl_shape else 1
    if owned == 0:
        return np.zeros((0, size))
    function.x.scatter_forward()
    dofmap = np.asarray(mesh.geometry.dofmaps[0])
    cells = np.empty(len(mesh.geometry.x), dtype=np.int32)
    cells[dofmap.ravel()] = np.repeat(np.arange(len(dofmap), dtype=np.int32), dofmap.shape[1])
    points = np.ascontiguousarray(mesh.geometry.x[:owned], dtype=np.float64)
    return np.asarray(function.eval(points, cells[:owned]), dtype=np.float64).reshape(owned, size)


def _dolfinx_to_volume_mesh(
    dolfinx_mesh: dolfinx.mesh.Mesh,
    u_sol: Function,
    p_sol: Function,
    volume_mesh_dtcc: Any,
) -> Any:
    """Gather dolfinx solution fields onto a dtcc-core VolumeMesh on rank zero.

    Every rank evaluates velocity and pressure directly at its owned mesh
    vertices (no P2→P1 interpolation); rank zero, which holds the VolumeMesh,
    collects them and attaches ``velocity``, ``pressure`` and ``speed`` fields
    in dtcc-core vertex ordering. Returns the VolumeMesh on rank zero and None
    on other ranks. Collective over the mesh communicator.
    """
    comm = dolfinx_mesh.comm
    owned = dolfinx_mesh.geometry.index_map().size_local
    local = (
        np.array(dolfinx_mesh.geometry.x[:owned], dtype=np.float64),
        _owned_vertex_values(dolfinx_mesh, u_sol),
        _owned_vertex_values(dolfinx_mesh, p_sol),
    )
    parts = comm.gather(local, root=0)
    if comm.rank != 0:
        return None
    coordinates, velocity, pressure = (np.concatenate([part[i] for part in parts]) for i in range(3))
    return _attach_wind_fields(volume_mesh_dtcc, coordinates, velocity, pressure.ravel())


def _attach_wind_fields(volume_mesh_dtcc: Any, fem_verts: np.ndarray, u_arr: np.ndarray, p_arr: np.ndarray) -> Any:
    """Attach vertex velocity, pressure and speed matched by coordinates."""
    from dtcc_core.model import Field as DtccField

    # ---- map to dtcc-core vertex ordering ----
    dtcc_verts = np.asarray(volume_mesh_dtcc.vertices, dtype=np.float64)  # (M,3)

    # Match relative coordinates so keys are independent of translation.
    decimals = 6
    dtcc_keys = np.round(dtcc_verts - np.min(dtcc_verts, axis=0), decimals)
    fem_keys = np.round(fem_verts - np.min(fem_verts, axis=0), decimals)

    u_mapped = _reorder_by_coords(fem_keys, u_arr, dtcc_keys)
    p_mapped = _reorder_by_coords(fem_keys, p_arr.reshape(-1, 1), dtcc_keys).ravel()

    speed = np.linalg.norm(u_mapped, axis=1)

    # Attach fields
    volume_mesh_dtcc.fields = [
        DtccField(
            association="vertex",
            name="velocity",
            dim=3,
            values=u_mapped,
            unit="m/s",
            description="Velocity field from urban wind CFD solver",
        ),
        DtccField(
            association="vertex",
            name="pressure",
            dim=1,
            values=p_mapped,
            unit="m^2/s^2",
            description="Pressure field from urban wind CFD solver",
        ),
        DtccField(
            association="vertex",
            name="speed",
            dim=1,
            values=speed,
            unit="m/s",
            description="Wind speed |u| from urban wind CFD solver",
        ),
    ]
    _validate_wind_volume_fields(volume_mesh_dtcc)
    return volume_mesh_dtcc


def _validate_wind_volume_fields(volume_mesh_dtcc: Any) -> None:
    """Fail loudly if required wind fields do not match the mesh contract."""
    vertices = np.asarray(volume_mesh_dtcc.vertices, dtype=np.float64)
    n_vertices = int(vertices.shape[0])
    fields = {field.name: field for field in getattr(volume_mesh_dtcc, "fields", [])}
    required = {
        "velocity": {"dim": 3, "unit": "m/s"},
        "pressure": {"dim": 1, "unit": "m^2/s^2"},
        "speed": {"dim": 1, "unit": "m/s"},
    }

    missing = [name for name in required if name not in fields]
    if missing:
        raise RuntimeError(
            "VolumeMesh output is missing required field(s): "
            f"{', '.join(missing)}."
        )

    for name, expected in required.items():
        field = fields[name]
        if int(field.dim) != expected["dim"]:
            raise RuntimeError(
                f"Field {name!r} has dim={field.dim}; "
                f"expected dim={expected['dim']}."
            )
        if field.unit != expected["unit"]:
            raise RuntimeError(
                f"Field {name!r} has unit={field.unit!r}; "
                f"expected {expected['unit']!r}."
            )
        values = np.asarray(field.values)
        if values.size != n_vertices * expected["dim"]:
            raise RuntimeError(
                f"Field {name!r} has {values.size} scalar values; "
                f"expected {n_vertices * expected['dim']} for {n_vertices} "
                f"vertices and dim={expected['dim']}."
            )
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"Field {name!r} contains non-finite values.")

    velocity = np.asarray(fields["velocity"].values, dtype=np.float64).reshape(
        n_vertices, 3
    )
    speed = np.asarray(fields["speed"].values, dtype=np.float64).reshape(n_vertices)
    expected_speed = np.linalg.norm(velocity, axis=1)
    if not np.allclose(speed, expected_speed, rtol=1.0e-8, atol=1.0e-10):
        raise RuntimeError(
            "The speed field is inconsistent with velocity magnitude."
        )


def _evaluate_function_at_vertices(
    mesh: dolfinx.mesh.Mesh, func: Function
) -> np.ndarray:
    """Evaluate a finite-element function at mesh vertices.

    Returns values in local mesh vertex ordering. Shape is (n_vertices, value_dim),
    with value_dim=1 for scalar functions.
    """
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(0, tdim)
    v2c = mesh.topology.connectivity(0, tdim)

    x = np.asarray(mesh.geometry.x, dtype=np.float64)
    n_vertices = x.shape[0]

    cells = np.full(n_vertices, -1, dtype=np.int32)
    for vi in range(n_vertices):
        incident = v2c.links(vi)
        if len(incident) > 0:
            cells[vi] = int(incident[0])

    valid = cells >= 0
    if np.any(valid):
        probe = np.asarray(func.eval(x[valid][:1], cells[valid][:1]), dtype=np.float64)
        if probe.ndim == 0:
            value_dim = 1
        elif probe.ndim == 1:
            # dolfinx may return shape (value_dim,) for a single point.
            value_dim = int(probe.shape[0]) if probe.shape[0] > 1 else 1
        else:
            value_dim = probe.shape[1]

        values = np.zeros((n_vertices, value_dim), dtype=np.float64)
        sampled = np.asarray(func.eval(x[valid], cells[valid]), dtype=np.float64)
        if sampled.ndim == 0:
            sampled = sampled.reshape(1, 1)
        elif sampled.ndim == 1:
            sampled = sampled.reshape(-1, value_dim)
        values[valid] = sampled
    else:
        values = np.zeros((n_vertices, 1), dtype=np.float64)

    if not np.all(valid):
        warning(
            f"_evaluate_function_at_vertices: could evaluate {int(np.sum(valid))}/{n_vertices} "
            "vertices; unmatched entries are zero."
        )
    return values


def _reorder_by_coords(
    src_coords: np.ndarray,
    src_vals: np.ndarray,
    dst_coords: np.ndarray,
    decimals: int = 6,
) -> np.ndarray:
    """Reorder *src_vals* so they align with *dst_coords* row order.

    Both coordinate arrays should already be rounded to *decimals* places.
    Falls back to a dict-based lookup when shapes don't match exactly.
    """
    n_dst = dst_coords.shape[0]
    n_src = src_coords.shape[0]
    out = np.zeros((n_dst, src_vals.shape[1]), dtype=src_vals.dtype)

    # Build dict: rounded tuple → index
    src_map: Dict[tuple, int] = {}
    for i in range(n_src):
        key = tuple(src_coords[i])
        src_map[key] = i

    matched = 0
    for j in range(n_dst):
        key = tuple(dst_coords[j])
        idx = src_map.get(key)
        if idx is not None:
            out[j] = src_vals[idx]
            matched += 1

    if matched < n_dst:
        warning(
            f"_reorder_by_coords: matched {matched}/{n_dst} vertices "
            f"(src has {n_src}); unmatched vertices will be zero."
        )
    return out


class _PressureSpaceError(RuntimeError):
    """Uncoupled pressure basis functions in the constrained mixed space."""

    def __init__(self, coordinates):
        self.coordinates = coordinates
        super().__init__(
            f"Taylor–Hood pressure space has {len(coordinates)} uncoupled pressure DOFs. "
            "Generate with repair_boundary_corners=True or repair the native mesh "
            "with repair_boundary_corners() before finite-element conversion. "
            "For an imported mesh, explicitly request mesh_refinement_levels=1 "
            "to refine affected cell stars."
        )


def _refine_pressure_mesh(mesh, markers, coordinates):
    """Refine cell stars supporting uncoupled pressure basis functions."""
    tdim = mesh.topology.dim
    mesh.topology.create_entities(1)
    mesh.topology.create_connectivity(tdim, 1)
    keys = {tuple(row) for row in np.round(coordinates, 8)}
    selected_vertices = np.array([tuple(row) in keys for row in np.round(mesh.geometry.x, 8)])
    cells = np.flatnonzero(np.any(selected_vertices[mesh.geometry.dofmaps[0]], axis=1)).astype(np.int32)
    edges = dolfinx.mesh.compute_incident_entities(mesh.topology, cells, tdim, 1)
    refined, parents, parent_facets = dolfinx.mesh.refine(
        mesh, edges=edges, option=dolfinx.mesh.RefinementOption.parent_cell_and_facet,
    )
    refined.topology.create_entities(tdim-1)
    refined.topology.create_connectivity(tdim-1, tdim)
    tags = dolfinx.mesh.transfer_meshtag(markers, refined, parents, parent_facets)
    info(f"Refining pressure support at {len(coordinates)} vertices; cells {mesh.topology.index_map(tdim).size_global} -> {refined.topology.index_map(tdim).size_global}")
    return refined, tags


CLOSURE_QUADRATURE_DEGREE = 5
"""Quadrature degree for forms containing wall distances, closures or wall laws."""


def _wall_distance(mesh, categories, tags, name):
    """Poisson wall distance to the facets with the given categories.

    Solves ``-Δφ = 1`` with ``φ = 0`` on the selected facets and natural
    conditions elsewhere, and returns ``φ`` with the UFL distance
    ``d = 2φ / (sqrt(|∇φ|² + 2φ) + |∇φ|)``. The distance is exact for a single
    plane wall with a parallel natural boundary.
    """
    fd = mesh.topology.dim - 1
    facets = np.unique(np.concatenate([categories.find(int(tag)) for tag in tags])).astype(np.int32)
    if mesh.comm.allreduce(len(facets), op=MPI.SUM) == 0:
        names = ", ".join(BndCat(int(tag)).name.lower() for tag in tags)
        raise ValueError(f"The {name} distance requires {names} boundary facets")
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))
    bc = dirichletbc(PETSc.ScalarType(0), locate_dofs_topological(V, fd, facets), V)
    phi, w = ufl.TrialFunction(V), ufl.TestFunction(V)
    problem = _fem_petsc.LinearProblem(
        inner(grad(phi), grad(w)) * dx, Constant(mesh, PETSc.ScalarType(1)) * w * dx, bcs=[bc],
        petsc_options_prefix=f"dtcc_{name}_distance_",
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "pc_hypre_type": "boomeramg",
                       "ksp_rtol": 1e-12, "ksp_atol": 1e-14, "ksp_max_it": 500},
    )
    solution = problem.solve()
    solution = solution[0] if isinstance(solution, tuple) else solution
    reason = problem.solver.getConvergedReason()
    if reason <= 0:
        raise RuntimeError(f"The {name} distance solve failed with KSP reason {reason}")
    solution.name = f"{name}_potential"
    potential = ufl.max_value(solution, 0.)
    slope = ufl.sqrt(dot(grad(solution), grad(solution)))
    scale = float(mesh.comm.allreduce(np.ptp(mesh.geometry.x, axis=0).max() if len(mesh.geometry.x) else 0., op=MPI.MAX))
    distance = 2 * potential / ufl.max_value(ufl.sqrt(slope**2 + 2 * potential) + slope, 1e-12 * max(scale, 1.))
    return solution, distance


SHARP_WALL_ANGLE_DEG = 30.
"""Wall facets meeting at a node at more than this angle keep their own normals."""


def _wall_normals(mesh, categories, solids):
    """Weak-wall normals averaged over the facets sharing each vertex.

    A polyhedral approximation of a curved wall has facet normals that jump
    at every edge, so flow tangent to the true wall crosses the facets and
    is penalized. Each wall vertex instead receives the area-weighted mean of
    the outward normals of all incident weak-wall facets, accumulated across
    MPI ranks, and the continuous P1 interpolant is used on smooth walls.
    Vertices whose incident facets deviate more than SHARP_WALL_ANGLE_DEG
    from that mean (building edges, wall–ground corners) are left at zero;
    where the interpolant is short or far from the facet normal the facet
    normal is used. On planar walls the result is the facet normal.
    """
    fd, tdim, gdim = mesh.topology.dim - 1, mesh.topology.dim, mesh.geometry.dim
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (gdim,)))
    total, sharp = Function(V), Function(V)
    mesh.topology.create_connectivity(fd, 0)
    mesh.topology.create_connectivity(fd, tdim)
    mesh.topology.create_connectivity(tdim, 0)
    cell_vertices = mesh.topology.connectivity(tdim, 0).array.reshape(-1, tdim + 1)
    vertex_dofs = np.empty(mesh.topology.index_map(0).size_local + mesh.topology.index_map(0).num_ghosts, dtype=np.int32)
    vertex_dofs[cell_vertices] = V.dofmap.list[:len(cell_vertices)]
    facets = np.unique(np.concatenate([categories.find(int(tag)) for tag in solids])).astype(np.int32)
    facets = facets[facets < mesh.topology.index_map(fd).size_local]
    facet_vertices = mesh.topology.connectivity(fd, 0)
    facet_cells = mesh.topology.connectivity(fd, tdim)
    if len(facets):
        dofs = vertex_dofs[np.array([facet_vertices.links(f) for f in facets])]
        points = mesh.geometry.x[dolfinx.mesh.entities_to_geometry(mesh, fd, facets).reshape(len(facets), -1)[:, :3]]
        cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
        cells = np.array([facet_cells.links(f)[0] for f in facets], dtype=np.int32)
        centroids = mesh.geometry.x[mesh.geometry.dofmaps[0][cells]].mean(axis=1)
        cross[np.einsum("ij,ij->i", cross, points.mean(axis=1) - centroids) < 0] *= -1
    else:
        dofs, cross = np.zeros((0, 3), dtype=np.int32), np.zeros((0, 3))
    sums = total.x.array.reshape(-1, gdim)
    for corner in range(3):
        np.add.at(sums, dofs[:, corner], .5 * cross)  # area-weighted outward normals
    total.x.scatter_reverse(dolfinx.la.InsertMode.add)
    total.x.scatter_forward()
    lengths = np.linalg.norm(sums, axis=1, keepdims=True)
    means = np.divide(sums, lengths, out=np.zeros_like(sums), where=lengths > 0)
    if len(facets):
        normals = cross / np.linalg.norm(cross, axis=1, keepdims=True)
        limit = np.cos(np.deg2rad(SHARP_WALL_ANGLE_DEG))
        flags = sharp.x.array.reshape(-1, gdim)
        for corner in range(3):
            deviating = np.einsum("ij,ij->i", normals, means[dofs[:, corner]]) < limit
            np.add.at(flags[:, 0], dofs[deviating, corner], 1.)
    sharp.x.scatter_reverse(dolfinx.la.InsertMode.add)
    sharp.x.scatter_forward()
    smooth = sharp.x.array.reshape(-1, gdim)[:, 0] == 0
    sums[:] = np.where(smooth[:, None], means, 0.)
    total.x.scatter_forward()
    total.name = "wall_normal"
    n = FacetNormal(mesh)
    length = ufl.sqrt(dot(total, total))
    averaged = total / ufl.max_value(length, 1e-12)
    use = ufl.And(ufl.ge(length, .5), ufl.ge(dot(averaged, n), np.cos(np.deg2rad(SHARP_WALL_ANGLE_DEG))))
    return total, ufl.conditional(use, averaged, n)


def _tangential_inflow(mesh, categories, solids, inlet):
    """Remove the wall-normal inflow on inlet edges shared with weak walls.

    The inflow profile is horizontal, while sloping terrain and walls only
    admit tangential velocity. Each inflow value on a weak-wall facet is
    projected onto the tangent plane of the mean normal of all weak-wall
    facets containing it, and also onto distinct wall directions where facets
    meet at more than 30 degrees. The incident facets are collected from all
    ranks and ordered geometrically, so the result does not depend on the
    mesh partition.
    """
    fd, gdim = mesh.topology.dim - 1, mesh.geometry.dim
    V = inlet.function_space
    index_map = V.dofmap.index_map
    inlet_facets = categories.find(int(BndCat.INLET)).astype(np.int32)
    inlet_dofs = locate_dofs_topological(V, fd, inlet_facets)
    inlet_set = set(inlet_dofs.tolist())
    facets = np.concatenate([categories.find(int(tag)) for tag in solids]).astype(np.int32)
    if not inlet_set:
        facets = facets[:0]
    nodes = dolfinx.mesh.entities_to_geometry(mesh, fd, facets)
    nodes = nodes.reshape(len(facets), nodes.shape[-1] if nodes.ndim == 2 else 3)
    inlet_nodes = np.unique(dolfinx.mesh.entities_to_geometry(mesh, fd, inlet_facets))
    touching = np.isin(nodes, inlet_nodes).any(axis=1)
    facets, nodes = facets[touching], nodes[touching]
    points = mesh.geometry.x[nodes]
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    midpoints = points.mean(axis=1)
    records = []
    for facet, normal, midpoint in zip(facets, normals, midpoints):
        dofs = [int(d) for d in locate_dofs_topological(V, fd, np.array([facet], dtype=np.int32), remote=False)
                if int(d) in inlet_set]
        for dof in index_map.local_to_global(np.array(dofs, dtype=np.int32)) if dofs else []:
            records.append([float(dof), *midpoint, *normal])
    gathered = [row for rows in mesh.comm.allgather(records) for row in rows]
    scale = float(mesh.comm.allreduce(np.ptp(mesh.geometry.x, axis=0).max() if len(mesh.geometry.x) else 0., op=MPI.MAX))
    incident = {}
    for row in gathered:
        key = tuple(np.round(np.asarray(row[1:4]) / max(scale, 1.), 9))
        incident.setdefault(int(row[0]), {})[key] = np.asarray(row[4:7])
    owned = np.array([dof for dof in inlet_dofs if dof < index_map.size_local], dtype=np.int32)
    values = inlet.x.array.reshape(-1, gdim)
    for dof, global_dof in zip(owned, index_map.local_to_global(owned) if len(owned) else []):
        facet_normals = incident.get(int(global_dof))
        if not facet_normals:
            continue
        ordered = [facet_normals[key] for key in sorted(facet_normals)]
        aligned = [n if np.dot(n, ordered[0]) >= 0 else -n for n in ordered]
        mean = np.sum(aligned, axis=0)
        basis = [mean / np.linalg.norm(mean)]
        for normal in aligned:
            residual = normal - sum(np.dot(normal, b) * b for b in basis)
            length = np.linalg.norm(residual)
            if length > .5:  # a distinct wall direction, not a nearly coplanar facet
                basis.append(residual / length)
        for b in basis:
            values[dof] -= np.dot(values[dof], b) * b
    inlet.x.scatter_forward()


class _MixedFlowSystem:
    """Taylor–Hood stationary momentum and continuity with traction boundaries.

    ``stage`` scales convection and selects the closure stage: 0 gives the
    linear initialization (equilibrium eddy viscosity and linearized wall law),
    1 the full model.
    """

    def __init__(self, mesh, params, markers, categories, inlet_expression):
        self.mesh, self.params = mesh, params
        gdim = mesh.geometry.dim
        velocity = basix.ufl.element("Lagrange", mesh.basix_cell(), 2, shape=(gdim,))
        pressure = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
        self.W = W = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([velocity, pressure]))
        self.w = Function(W, name="flow")
        self.convection = Constant(mesh, PETSc.ScalarType(0))
        self.stage = Constant(mesh, PETSc.ScalarType(0))
        self.u, self.p = u, p = ufl.split(self.w)
        v, q = ufl.TestFunctions(W)
        # Closure fields are not polynomials; a fixed rule keeps their forms affordable.
        if params.uses_wall_distance:
            metadata = {"quadrature_degree": CLOSURE_QUADRATURE_DEGREE}
            dx = Measure("dx", domain=mesh, metadata=metadata)
            self.ds = Measure("ds", domain=mesh, subdomain_data=categories, metadata=metadata)
        else:
            dx = ufl.dx
            self.ds = Measure("ds", domain=mesh, subdomain_data=categories)
        n = FacetNormal(mesh)
        solids = [BndCat.WALL, BndCat.ROOF, BndCat.GROUND]
        if inlet_expression is None and not params.uses_wall_distance:
            inlet_expression = make_inlet_velocity_expression(params)
        self._closure(categories, solids, inlet_expression is None)
        nu = self.nu
        viscous = inner(self.stress(u), grad(v)) if self.symmetric else nu * inner(grad(u), grad(v))
        self.F = (viscous - p * div(v) + q * div(u)
                  + self.convection * inner(grad(u) * u, v)) * dx
        if self.symmetric:
            open_tags = [BndCat.OUTLET] + ([BndCat.SIDE, BndCat.TOP] if params.side_top_boundary == "open" else [])
            for tag in open_tags:
                self.F -= nu * dot(grad(u).T * n, v) * self.ds(int(tag))
        self.bcs = []
        Vu, _ = W.sub(0).collapse()
        inlet = Function(Vu)
        if inlet_expression is None:
            inlet.interpolate(dolfinx.fem.Expression(self._abl_inlet(), Vu.element.interpolation_points))
        else:
            inlet.interpolate(inlet_expression)
        fd = mesh.topology.dim - 1
        if params.wall_model != "noslip":
            _tangential_inflow(mesh, categories, solids, inlet)
        dofs = locate_dofs_topological((W.sub(0), Vu), fd, categories.find(int(BndCat.INLET)))
        self.bcs.append(dirichletbc(inlet, dofs, W.sub(0)))
        self.wall_normal_field, self.wall_normal = None, n
        if params.wall_model == "noslip":
            zero = Function(Vu)
            for tag in solids:
                dofs = locate_dofs_topological((W.sub(0), Vu), fd, categories.find(int(tag)))
                self.bcs.append(dirichletbc(zero, dofs, W.sub(0)))
        else:
            self.wall_normal_field, self.wall_normal = _wall_normals(mesh, categories, solids)
            for tag in solids:
                facets = categories.find(int(tag))
                geometry = dolfinx.mesh.entities_to_geometry(mesh, fd, facets)
                points = mesh.geometry.x[geometry]
                normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
                for axis in range(gdim):
                    aligned = facets[np.abs(normals[:, axis]) > 1 - 1e-10]
                    sub = W.sub(0).sub(axis)
                    S, _ = sub.collapse()
                    dofs = locate_dofs_topological((sub, S), fd, aligned)
                    self.bcs.append(dirichletbc(Function(S), dofs, sub))
                self._normal_boundary(u, p, v, q, n, self.wall_normal, nu, int(tag), params.gamma_normal)
                tangential = u - dot(u, self.wall_normal) * self.wall_normal
                self.F += self._wall_friction(tangential) * inner(tangential, v) * self.ds(int(tag))
        if params.side_top_boundary == "slip":
            all_markers = sorted(set().union(*mesh.comm.allgather(set(int(m) for m in markers.values))))
            for marker in all_markers:
                facets = np.intersect1d(markers.find(int(marker)), np.concatenate([categories.find(int(BndCat.SIDE)), categories.find(int(BndCat.TOP))])).astype(np.int32)
                count = mesh.comm.allreduce(len(facets), op=MPI.SUM)
                if not count:
                    continue
                normal = BBOX_MARKER_NORMALS.get(int(marker))
                if normal is None:
                    raise ValueError("Slip faces require canonical axis-aligned bbox markers")
                geometry = dolfinx.mesh.entities_to_geometry(mesh, fd, facets)
                points = mesh.geometry.x[geometry]
                normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
                invalid = bool(len(normals) and np.max(np.linalg.norm(np.cross(normals, normal), axis=1)) > 1e-8)
                if mesh.comm.allreduce(invalid, op=MPI.LOR):
                    raise ValueError("Slip marker faces must be axis aligned")
                axis = int(np.argmax(np.abs(normal)))
                sub = W.sub(0).sub(axis)
                S, _ = sub.collapse()
                dofs = locate_dofs_topological((sub, S), fd, facets)
                self.bcs.append(dirichletbc(Function(S), dofs, sub))
            if self.top_shear:
                wx, wy = params.wind_vector_xy
                stress = params.friction_velocity**2
                self.F -= stress * (wx * v[0] + wy * v[1]) * self.ds(int(BndCat.TOP))
        if params.grad_div_gamma:
            self.F += params.grad_div_gamma * div(u) * div(v) * dx
        if getattr(params, "backflow_beta", 0.):
            for tag in [BndCat.OUTLET, BndCat.SIDE, BndCat.TOP]:
                if tag != BndCat.OUTLET and params.side_top_boundary != "open":
                    continue
                self.F += self.convection * params.backflow_beta * ufl.max_value(-dot(u, n), 0) * inner(u, v) * self.ds(int(tag))
        self.J = ufl.derivative(self.F, self.w)
        du, dp = ufl.TrialFunctions(W)
        self.P = self.J + (1 / nu) * dp * q * dx
        self.nullspace = None
        self.null_vector = None
        self.splits = []
        self.history = []
        self.problem = None

    def _closure(self, categories, solids, abl_inlet):
        """Define wall distances and the effective viscosity ``nu``.

        The ABL closures use ``l_m = κ(d + z_w)``, optionally with a Blackadar
        limit. ``abl`` gives the equilibrium ``ν_t = u* l_m``; ``mixing_length``
        gives ``ν_t = l_m² |S|`` with ``|S| = sqrt(2 ε(u):ε(u))``.
        """
        params, mesh = self.params, self.mesh
        self.wall_distance = self.ground_distance = None
        self.symmetric = params.turbulence_model != "constant"
        self.top_shear = (params.turbulence_model != "constant" and params.abl_top_shear
                          and params.side_top_boundary == "slip")
        if params.uses_wall_distance:
            _, self.wall_distance = _wall_distance(mesh, categories, solids, "wall")
            if abl_inlet:
                _, self.ground_distance = _wall_distance(mesh, categories, [BndCat.GROUND], "ground")
        if params.turbulence_model == "constant":
            self.nu_t, self.nu = None, params.nu_eff
            return
        friction = params.friction_velocity
        length = KAPPA * (self.wall_distance + params.wall_offset)
        if params.mixing_length_limit is not None:
            length = length / (1 + length / params.mixing_length_limit)
        equilibrium = friction * length
        if params.turbulence_model == "abl":
            self.nu_t = equilibrium
        else:
            strain = ufl.sym(grad(self.u))
            floor = max(1e-6 * friction / (KAPPA * params.wall_offset), 1e-10)
            shear = ufl.sqrt(2 * inner(strain, strain) + floor**2)
            self.nu_t = (1 - self.stage) * equilibrium + self.stage * length**2 * shear
        self.nu = params.nu + self.nu_t

    def stress(self, u):
        """Viscous stress: gradient form for constant viscosity, symmetric otherwise."""
        if self.symmetric:
            return self.nu * (grad(u) + grad(u).T)
        return self.nu * grad(u)

    def _wall_friction(self, tangential):
        """Tangential wall-friction coefficient.

        The log-law wall applies ``C_d |u_t| u_t`` with
        ``C_d = (κ / ln(z_w/z0))²``; initialization uses its equilibrium
        linearization ``κ u* / ln(z_w/z0)``.
        """
        params = self.params
        if params.wall_model == "friction":
            return params.beta_wall
        log_ratio = float(np.log(params.wall_offset / params.z0))
        drag = (KAPPA / log_ratio) ** 2
        linear = KAPPA * params.friction_velocity / log_ratio
        floor = max(1e-6 * params.wind_speed, 1e-12)
        return (1 - self.stage) * linear + self.stage * drag * ufl.sqrt(inner(tangential, tangential) + floor**2)

    def _abl_inlet(self):
        """Equilibrium log-law inflow at terrain-relative height."""
        params = self.params
        wx, wy = params.wind_vector_xy
        speed = (params.friction_velocity / KAPPA) * ufl.ln((self.ground_distance + params.wall_offset) / params.z0)
        return ufl.as_vector((speed * wx, speed * wy, 0. * speed))

    def closure_diagnostics(self):
        """Closure constants and cell-sampled eddy-viscosity and distance statistics."""
        params = self.params
        result = {"turbulence_model": params.turbulence_model, "wall_model": params.wall_model,
                  "viscous_form": "symmetric" if self.symmetric else "gradient"}
        if not params.uses_wall_distance:
            return {**result, "nu_t": params.nu_t}
        result.update(
            von_karman=KAPPA, roughness_length=params.z0, friction_velocity=params.friction_velocity,
            wall_offset=params.wall_offset, top_shear_stress=params.friction_velocity**2 if self.top_shear else 0.,
            terrain_relative_inlet=self.ground_distance is not None,
        )
        if params.wall_model == "log_law":
            result["wall_function_height"] = params.wall_function_height
            result["drag_coefficient"] = (KAPPA / float(np.log(params.wall_offset / params.z0)))**2
        if params.turbulence_model == "mixing_length":
            result["mixing_length_limit"] = params.mixing_length_limit
        cells = dolfinx.fem.functionspace(self.mesh, ("DG", 0))
        def statistics(expression):
            sampled = Function(cells)
            sampled.interpolate(dolfinx.fem.Expression(expression, cells.element.interpolation_points))
            local = sampled.x.array[:self.mesh.topology.index_map(self.mesh.topology.dim).size_local]
            comm = self.mesh.comm
            volume = self.integral(Constant(self.mesh, PETSc.ScalarType(1)) * ufl.dx)
            return {"min": float(comm.allreduce(np.min(local) if len(local) else np.inf, op=MPI.MIN)),
                    "max": float(comm.allreduce(np.max(local) if len(local) else -np.inf, op=MPI.MAX)),
                    "mean": self.integral(expression * ufl.dx) / volume}
        result["wall_distance_m"] = statistics(self.wall_distance)
        if params.turbulence_model != "constant":
            result["eddy_viscosity_m2_s"] = statistics(self.nu_t)
        return result

    def set_stage(self, value):
        """Select linear initialization (0) or the full nonlinear model (1)."""
        self.convection.value = PETSc.ScalarType(value)
        self.stage.value = PETSc.ScalarType(value)

    def _normal_boundary(self, u, p, v, q, n, m, nu, tag, penalty):
        """Nitsche no penetration along the wall normal ``m``.

        The traction ``σ(u) n - p n`` on a facet with normal ``n`` is
        projected on ``m``; on planar walls ``m = n``.
        """
        um, vm = dot(u, m), dot(v, m)
        if penalty is None:
            height = 3 * ufl.CellVolume(self.mesh) / ufl.FacetArea(self.mesh)
            penalty = self.params.nitsche_penalty * (nu / height + self.params.wind_speed)
        def normal_stress(w):
            return dot(self.stress(w) * n, m) if self.symmetric else nu * dot(grad(w) * n, m)
        alignment = dot(n, m)
        self.F += (-(normal_stress(u) - p * alignment) * vm
                   - (normal_stress(v) + q * alignment) * um + penalty * um * vm) * self.ds(tag)

    @timed("assembly_and_preconditioner_setup")
    def configure(self, options):
        """Create SNES and verify whether constant pressure is a true null mode."""
        self.effective_options = dict(options)
        self.problem = _fem_petsc.NonlinearProblem(
            self.F, self.w, J=self.J,
            P=self.P if options.get("pc_type") == "fieldsplit" and not options.get("pc_use_amat", False) else None,
            bcs=self.bcs, petsc_options_prefix="dtcc_stationary_", petsc_options=options,
        )
        problem = self.problem
        self._adaptive_linear = problem.solver.getUseEW()
        self._linear_rtol = problem.solver.getKSP().getTolerances()[0]
        self.splits = [_FlowSolver._mixed_subspace_is(self.W, i) for i in range(2)]
        if options.get("pc_type") == "fieldsplit":
            problem.solver.getKSP().getPC().setFieldSplitIS(("u", self.splits[0]), ("p", self.splits[1]))
        _fem_petsc.assemble_matrix(problem.A, problem.J, bcs=self.bcs)
        problem.A.assemble()
        coupling = problem.A.createSubMatrix(self.splits[0], self.splits[1])
        try:
            _, columns, values = coupling.getValuesCSR()
            local = np.bincount(columns, weights=np.abs(values)**2, minlength=coupling.getSize()[1])
            norms = np.empty_like(local)
            self.mesh.comm.Allreduce(local, norms, op=MPI.SUM)
            norms = np.sqrt(norms)
            threshold = 64 * np.finfo(float).eps * max(float(np.max(norms)), 1e-30)
            if np.any(norms <= threshold):
                Q, parent = self.W.sub(1).collapse()
                parent = np.asarray(parent).reshape(-1)
                owned = np.flatnonzero(parent < self.W.dofmap.index_map.size_local)
                order = owned[np.argsort(parent[owned])]
                counts = self.mesh.comm.allgather(len(order))
                offset = sum(counts[:self.mesh.comm.rank])
                bad = norms[offset:offset+len(order)] <= threshold
                points = Q.tabulate_dof_coordinates()[order[bad]]
                points = np.asarray([point for group in self.mesh.comm.allgather(points.tolist()) for point in group]).reshape(-1, 3)
                raise _PressureSpaceError(points)
        finally:
            coupling.destroy()
        candidate = Function(self.W)
        _, pmap = self.W.sub(1).collapse()
        candidate.x.array[pmap] = 1.
        candidate.x.scatter_forward()
        vec = candidate.x.petsc_vec.copy()
        vec.normalize()
        null = PETSc.NullSpace().create(vectors=[vec], comm=self.mesh.comm)
        check = vec.duplicate()
        pressure_scale = float(np.linalg.norm(norms))
        null_tolerance = max(1e-12, 128 * np.finfo(PETSc.RealType).eps) * pressure_scale
        problem.A.mult(vec, check)
        right_null = check.norm() <= null_tolerance
        problem.A.multTranspose(vec, check)
        left_null = check.norm() <= null_tolerance
        check.destroy()
        if right_null:
            if not left_null:
                null.destroy()
                vec.destroy()
                raise RuntimeError("Pressure gauge has no compatible left nullspace")
            self.nullspace, self.null_vector = null, vec
            problem.A.setNullSpace(null)
            problem.A.setTransposeNullSpace(null)
            override = (getattr(self.params, "petsc_options", None)
                        if self.params.equations == "navier_stokes"
                        else getattr(self.params, "petsc_stokes", None)) or {}
            if (options.get("pc_type") == "fieldsplit"
                    and options.get("pc_fieldsplit_schur_precondition") == "selfp"
                    and options.get("pc_use_amat")
                    and "fieldsplit_p_pc_type" not in override):
                ksp = problem.solver.getKSP()
                ksp.setOperators(problem.A)
                ksp.getPC().setUp()
                ksp.getPC().getFieldSplitSubKSP()[1].getPC().setType("jacobi")
                self.effective_options["fieldsplit_p_pc_type"] = "jacobi"
            def residual(snes, x, b):
                _fem_petsc.assemble_residual(snes, x, b, self.w, problem.F, problem.J, self.bcs)
                null.remove(b)
            problem.solver.setFunction(residual, problem.b)
        else:
            null.destroy(); vec.destroy()
        _fem_petsc.set_bc(self.w.x.petsc_vec, self.bcs)
        self.w.x.scatter_forward()
        def monitor(snes, iteration, norm):
            record = dict(iteration=iteration, residual=float(norm), linear_iterations=snes.getLinearSolveIterations(),
                          linear_rtol=float(snes.getKSP().getTolerances()[0]), convection=float(self.convection.value))
            self.history.append(record)
            if self.mesh.comm.rank == 0:
                info(f"Newton iteration {iteration}: residual={norm:.6e}, linear iterations={record['linear_iterations']}, convection={record['convection']:g}")
        problem.solver.setMonitor(monitor)
        def linear_monitor(ksp, iteration, norm):
            if self.mesh.comm.rank == 0 and (iteration == 0 or iteration % 25 == 0):
                info(f"Linear iteration {iteration}: residual={norm:.6e}")
        problem.solver.getKSP().setMonitor(linear_monitor)

    def residual(self):
        """Assemble the residual, including Dirichlet mismatch and gauge compatibility."""
        w = self.w
        w.x.scatter_forward()
        b = _fem_petsc.assemble_vector(_fem_form(self.F))
        try:
            _fem_petsc.apply_lifting(b, [_fem_form(self.J)], [self.bcs], x0=[w.x.petsc_vec], alpha=-1.)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            _fem_petsc.set_bc(b, self.bcs, w.x.petsc_vec, -1.)
            compatibility = abs(self.null_vector.dot(b)) if self.nullspace else 0.
            if self.nullspace:
                self.nullspace.remove(b)
            return float(b.norm()), float(compatibility)
        finally:
            b.destroy()

    @timed("linear_or_nonlinear_solve")
    def solve(self, convection):
        self.set_stage(convection)
        self.problem.solver.setUseEW(self._adaptive_linear and bool(convection))
        if not convection:
            self.problem.solver.getKSP().setTolerances(rtol=self._linear_rtol)
        self.problem.solve()
        snes = self.problem.solver
        ksp = snes.getKSP()
        result = dict(snes_reason=int(snes.getConvergedReason()), nonlinear_iterations=snes.getIterationNumber(), linear_iterations=snes.getLinearSolveIterations(), linear_reason=int(ksp.getConvergedReason()))
        if result['snes_reason'] <= 0 or (result['linear_iterations'] > 0 and result['linear_reason'] <= 0):
            raise RuntimeError(f"Stationary flow solve failed: {result}")
        return result

    def fields(self):
        u, p = self.w.sub(0).collapse(), self.w.sub(1).collapse()
        u.name, p.name = "velocity", "pressure"
        if self.nullspace:
            mean = self.integral(p * dx) / self.integral(Constant(self.mesh, PETSc.ScalarType(1.)) * dx)
            p.x.array[:] -= mean
            p.x.scatter_forward()
        return u, p

    def integral(self, expression):
        return float(self.mesh.comm.allreduce(dolfinx.fem.assemble_scalar(_fem_form(expression)), op=MPI.SUM))

    def close(self):
        if self.problem is not None:
            problem = self.problem
            problem.solver.cancelMonitor()
            problem.solver.getKSP().cancelMonitor()
            problem.solver.setFunction(_fem_petsc.assemble_residual, problem.b, kargs=dict(u=self.w, residual=problem.F, jacobian=problem.J, bcs=self.bcs))
            self.problem = None
            del problem
        if self.nullspace:
            self.nullspace.destroy()
            self.null_vector.destroy()
        for index_set in self.splits:
            index_set.destroy()
        self.splits = []
        self.nullspace = self.null_vector = None


class _FlowSolver:
    """Shared mesh, output and mixed stationary solve contracts."""

    def __init__(
        self,
        *,
        bounds: Optional[Any] = None,
        mesh_path: Optional[str] = None,
        mesh: Optional[dolfinx.mesh.Mesh] = None,
        markers: Optional[dolfinx.mesh.MeshTags] = None,
        output_dir: Optional[str] = None,
        params: Optional[_FlowParameters] = None,
        inlet_expression: Optional[callable] = None,
        mesh_crs: Optional[str] = None,
    ) -> None:
        self.output_dir = _output_directory(output_dir)
        self.bounds = bounds
        self.mesh_path = mesh_path
        self.mesh = mesh
        self.mesh_crs = input_crs(mesh, generated=bounds is not None, declared=mesh_crs)
        self.markers = markers
        self.params = params if params is not None else self.parameter_type()
        self._validate_parameters()
        self._validate_runtime()
        self.inlet_expression = inlet_expression

        # populated during simulation
        self.volume_mesh_dtcc: Optional[Any] = None
        self.num_buildings: int = 0
        self.category_markers: Optional[dolfinx.mesh.MeshTags] = None
        self.diagnostics: Dict[str, Any] = {}

    def _validate_parameters(self):
        if not isinstance(self.params, self.parameter_type):
            raise TypeError(f"{type(self).__name__} requires {self.parameter_type.__name__}")
        self.params = self.parameter_type.model_validate(self.params.model_dump())
        expected = self.parameter_type.model_fields["equations"].default
        if self.params.equations != expected:
            raise ValueError(f"{type(self).__name__} requires equations={expected!r}")

    @staticmethod
    def _validate_runtime():
        if not np.issubdtype(PETSc.ScalarType, np.floating):
            raise RuntimeError(
                "Stokes and Navier–Stokes require real PETSc. "
                "Run 'conda activate fenicsx-env' before running wind simulations; "
                "fenicsx-env-complex is for Helmholtz."
            )

    # ------------------------------------------------------------------ mesh
    def _boundary_category_counts(self, mesh: dolfinx.mesh.Mesh) -> Dict[str, int]:
        """Return global counts for collapsed boundary categories."""
        if self.category_markers is None:
            return {}
        values = np.asarray(self.category_markers.values, dtype=np.int32)
        names = {
            BndCat.WALL: "wall",
            BndCat.ROOF: "roof",
            BndCat.GROUND: "ground",
            BndCat.INLET: "inlet",
            BndCat.OUTLET: "outlet",
            BndCat.TOP: "top",
            BndCat.SIDE: "side",
        }
        counts: Dict[str, int] = {}
        for category, name in names.items():
            local_count = int(np.count_nonzero(values == int(category)))
            counts[name] = int(mesh.comm.allreduce(local_count, op=MPI.SUM))
        return counts

    @staticmethod
    def _function_stats(function: Function) -> Dict[str, Any]:
        """Return global finite/range/L2 diagnostics for a FE function array."""
        mesh = function.function_space.mesh
        values = np.real(np.asarray(function.x.array))
        finite_local = bool(np.all(np.isfinite(values)))
        finite = bool(mesh.comm.allreduce(int(finite_local), op=MPI.MIN))
        if values.size:
            local_min = float(np.min(values))
            local_max = float(np.max(values))
            local_l2_sq = float(np.dot(values, values))
        else:
            local_min = float("inf")
            local_max = float("-inf")
            local_l2_sq = 0.0
        global_min = float(mesh.comm.allreduce(local_min, op=MPI.MIN))
        global_max = float(mesh.comm.allreduce(local_max, op=MPI.MAX))
        global_l2_sq = float(mesh.comm.allreduce(local_l2_sq, op=MPI.SUM))
        return {
            "finite": finite,
            "min": global_min,
            "max": global_max,
            "l2_norm": float(np.sqrt(max(global_l2_sq, 0.0))),
        }

    @staticmethod
    def _function_dof_count(function: Function) -> int:
        """Return the global scalar DOF count for a function."""
        dofmap = function.function_space.dofmap
        return int(dofmap.index_map.size_global * dofmap.index_map_bs)

    @staticmethod
    def _ksp_info(ksp: PETSc.KSP) -> Dict[str, Any]:
        """Return the KSP convergence information currently exposed by PETSc."""
        return {
            "converged_reason": int(ksp.getConvergedReason()),
            "iterations": int(ksp.getIterationNumber()),
            "residual_norm": float(ksp.getResidualNorm()),
            "pc_failed_reason": int(ksp.getPC().getFailedReason()),
        }

    def _build_mesh_from_bounds(self) -> None:
        comm = MPI.COMM_WORLD
        rank = comm.rank

        info("Building a volume mesh for stationary flow …")
        # --- serial work: only rank 0 builds the mesh and writes to disk ---
        import tempfile

        tmp_path: Optional[str] = None
        failure = None
        if rank == 0:
            try:
                volume_mesh, self.mesh_record = prepare_city_mesh(self.bounds, self.params)
                self.volume_mesh_dtcc = volume_mesh

                with tempfile.NamedTemporaryFile(suffix=".xdmf", delete=False) as tmp:
                    tmp_path = tmp.name
                info(f"Saving the generated flow mesh to {tmp_path}")
                volume_mesh.save(tmp_path)

            except Exception as exc:
                if comm.size == 1:
                    raise
                failure = f"{type(exc).__name__}: {exc}"
        failure = comm.bcast(failure, root=0)
        if failure is not None:
            raise RuntimeError(f"Could not prepare city mesh: {failure}")

        # broadcast the temp path so all ranks open the *same* file
        tmp_path = comm.bcast(tmp_path, root=0)
        self.mesh_record = comm.bcast(getattr(self, "mesh_record", None), root=0)
        comm.barrier()  # ensure file is written before any rank reads

        # --- parallel read: all ranks collectively load the mesh ---
        self.mesh, self.markers = load_mesh_with_markers(tmp_path)

        # barrier *after* read so no rank deletes before others finish
        comm.barrier()

        # rank 0 cleans up
        if rank == 0:
            Path(tmp_path).unlink(missing_ok=True)
            Path(tmp_path).with_suffix(".h5").unlink(missing_ok=True)

    @timed("mesh_preparation")
    def _load_if_needed(self) -> None:
        from dtcc_core.model import VolumeMesh

        if isinstance(self.mesh, VolumeMesh):
            self.volume_mesh_dtcc = self.mesh
            if self.markers is not None:
                raise ValueError("DTCC meshes carry their boundary markers; omit markers")
            self.mesh, self.markers = _mesh_from_dtcc(self.mesh)

        if self.mesh is not None and self.markers is not None:
            return
        if self.bounds is not None:
            self._build_mesh_from_bounds()
        elif self.mesh_path is not None:
            info(f"Loading the flow mesh from {self.mesh_path}")
            self.mesh, self.markers = load_mesh_with_markers(self.mesh_path)
        else:
            raise ValueError(
                f"{type(self).__name__} requires bounds, mesh_path, or (mesh, markers)."
            )

    # ------------------------------------------------------- weather helper
    @staticmethod
    def _circular_mean_deg(angles_deg: np.ndarray) -> float:
        """Compute the circular (directional) mean of angles in degrees."""
        theta = np.deg2rad(angles_deg)
        return float(
            (
                np.rad2deg(np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta))))
                + 360.0
            )
            % 360.0
        )

    @staticmethod
    def _circular_median_deg(angles_deg: np.ndarray) -> float:
        """Approximate circular median: circular mean is used.

        A true circular median is complex; the circular mean is a
        reasonable substitute for small station counts.
        """
        return _FlowSolver._circular_mean_deg(angles_deg)

    def _nearest_station_index(self, pts: np.ndarray) -> int:
        """Return the index of the station closest to the domain centre."""
        if self.bounds is None:
            return 0
        try:
            b = self.bounds
            # bounds may be a tuple (xmin, ymin, xmax, ymax) or an object
            if hasattr(b, "center"):
                cx, cy = b.center.x, b.center.y
            elif hasattr(b, "__len__") and len(b) >= 4:
                cx = 0.5 * (b[0] + b[2])
                cy = 0.5 * (b[1] + b[3])
            else:
                return 0
            dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
            return int(np.argmin(dists))
        except Exception:
            return 0

    @timed("data_acquisition")
    def _maybe_fetch_weather(self) -> None:
        """Optionally override wind_speed / wind_dir_deg from SMHI weather.

        Uses the dtcc-core ``SensorCollection.to_arrays()`` API.
        """
        if not self.params.use_weather:
            return

        comm = MPI.COMM_WORLD
        rank = comm.rank
        weather_data: Optional[Tuple[float, float]] = None
        weather_error: Optional[str] = None

        # Only rank 0 performs the HTTP fetch
        if rank == 0:
            try:
                import dtcc_core.datasets as datasets
            except ImportError as exc:
                weather_error = (
                    "use_weather=True requires dtcc_core.datasets "
                    f"so reference wind can be fetched. Original error: {exc}"
                )
                datasets = None

            if weather_error is None:
                info("Fetching wind conditions from weather data …")
                try:
                    sensors = datasets.weather(
                        bounds=self.bounds,
                        parameters=["wind_speed", "wind_direction"],
                    )

                    pts_ws, ws_vals = sensors.to_arrays("wind_speed")
                    pts_wd, wd_vals = sensors.to_arrays("wind_direction")

                    if len(ws_vals) > 0 and len(wd_vals) > 0:
                        agg = self.params.weather_aggregation
                        if agg == "nearest":
                            idx_ws = self._nearest_station_index(pts_ws)
                            idx_wd = self._nearest_station_index(pts_wd)
                            ws = float(ws_vals[idx_ws])
                            wd = float(wd_vals[idx_wd])
                        elif agg == "mean":
                            ws = float(np.mean(ws_vals))
                            wd = self._circular_mean_deg(wd_vals)
                        elif agg == "median":
                            ws = float(np.median(ws_vals))
                            wd = self._circular_median_deg(wd_vals)
                        else:
                            ws = float(ws_vals[0])
                            wd = float(wd_vals[0])
                        weather_data = (ws, wd)
                    else:
                        weather_error = (
                            "use_weather=True but weather data did "
                            "not include both wind_speed and wind_direction "
                            "values for the requested bounds."
                        )
                except Exception as exc:
                    weather_error = (
                        "use_weather=True but the weather fetch "
                        "failed. Disable use_weather or provide manual "
                        "wind_speed and wind_dir_deg. "
                        f"Original error: {exc}"
                    )

        # Broadcast result or provider failure to all ranks before raising.
        weather_data, weather_error = comm.bcast(
            (weather_data, weather_error),
            root=0,
        )

        if weather_error is not None:
            raise RuntimeError(weather_error)

        if weather_data is not None:
            ws, wd = weather_data
            info(f"Weather gives wind speed {ws:.1f} m/s from {wd:.0f}°")
            self.params = self.params.model_copy(
                update={"wind_speed": ws, "wind_dir_deg": wd}
            )
            return

        raise RuntimeError(
            "use_weather=True but no usable wind data was returned."
        )

    # ----------------------------------------------------- output helpers
    @staticmethod
    @timed("export_preparation")
    def _prepare_xdmf_fields(
        mesh: dolfinx.mesh.Mesh,
        u_sol: Function,
        p_sol: Function,
    ):
        """Sample output fields while numerical coordinates remain local."""
        # XDMF requires functions of the same degree as the mesh (P1).
        # Interpolate higher-order velocity down to P1 for output.
        V1_out = FunctionSpace(mesh, "Lagrange", 1, dim=3)
        u_out = Function(V1_out, name="velocity")
        u_out.interpolate(u_sol)

        p_out = Function(p_sol.function_space, name="pressure")
        p_out.x.array[:] = p_sol.x.array
        p_out.x.scatter_forward()
        return u_out, p_out

    @staticmethod
    def _write_solution_xdmf(mesh, u_sol, p_sol, output_path, *, prepared=None):
        """Write velocity and pressure in the input coordinate frame."""
        from dolfinx.io import XDMFFile
        import os

        u_out, p_out = prepared or _FlowSolver._prepare_xdmf_fields(mesh, u_sol, p_sol)

        base, ext = os.path.splitext(output_path)
        vel_path = f"{base}_velocity{ext}"
        pres_path = f"{base}_pressure{ext}"
        _ensure_parent_dir(vel_path)
        _ensure_parent_dir(pres_path)

        with XDMFFile(mesh.comm, vel_path, "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_function(u_out)
        with XDMFFile(mesh.comm, pres_path, "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_function(p_out)
        if mesh.comm.rank == 0:
            info(
                f"Saved velocity to {vel_path} and pressure to {pres_path}"
            )

    def _native_output(self, mesh: dolfinx.mesh.Mesh) -> bool:
        """Whether a native VolumeMesh, held on rank zero, receives the fields."""
        return bool(mesh.comm.allreduce(int(self.volume_mesh_dtcc is not None), op=MPI.MAX))

    def _finalize_result(
        self,
        mesh: dolfinx.mesh.Mesh,
        u_sol: Function,
        p_sol: Function,
        *,
        output_path: Optional[str],
        native_result=None,
        xdmf_fields=None,
    ) -> Any:
        """Handle optional XDMF output and return type conversion."""
        result = (u_sol, p_sol)
        if self._native_output(mesh):
            if mesh.comm.rank == 0:
                result = native_result
                result.simulation_diagnostics = dict(self.diagnostics)
            else:
                result = RankZeroResult()
        with timing_phase("export"):
            if output_path is not None:
                self._write_solution_xdmf(mesh, u_sol, p_sol, output_path, prepared=xdmf_fields)
                write_coordinate_frame(output_path, self.coordinate_frame, comm=mesh.comm)
                _export_city_surface(
                    output_path, native_mesh=self.volume_mesh_dtcc, mesh=mesh,
                    markers=self.markers, frame=self.coordinate_frame,
                )
            publish_mesh_provenance(self, result, output_path)
        return result

    @staticmethod
    def _mixed_subspace_is(
        W: dolfinx.fem.FunctionSpace, subspace: int
    ) -> PETSc.IS:
        """Create a global IS for owned dofs of a mixed subspace."""
        _, parent_map = W.sub(subspace).collapse()
        local_parent = np.unique(np.asarray(parent_map, dtype=np.int32))
        n_owned = W.dofmap.index_map.size_local
        local_parent = local_parent[local_parent < n_owned]
        global_parent = np.asarray(
            W.dofmap.index_map.local_to_global(local_parent), dtype=PETSc.IntType
        )
        return PETSc.IS().createGeneral(global_parent, comm=W.mesh.comm)

    @timed_run
    def simulate(self, *, output_path: Optional[str] = None) -> Any:
        """Solve stationary incompressible flow and export only accepted fields."""
        self._validate_parameters()
        self._validate_runtime()
        output_path = _solution_path(
            self.output_dir, output_path, "solution.xdmf", input_mesh=self.mesh_path,
            stems=("_velocity", "_pressure"),
        )
        self.diagnostics = {"equations": self.params.equations, "stop_reason": "initializing"}
        started = time.perf_counter()
        self._load_if_needed()
        original_mesh = self.mesh
        crs = input_crs(self.volume_mesh_dtcc, generated=self.bounds is not None, declared=self.mesh_crs)
        with _local_mesh_coordinates(original_mesh, crs=crs) as frame:
            self.coordinate_frame = frame
            self.diagnostics["coordinate_frame"] = frame.as_dict()
            try:
                mesh, u, p = self._solve_stationary(started)
                with timing_phase("postprocessing"):
                    native_result = (
                        _dolfinx_to_volume_mesh(mesh, u, p, self.volume_mesh_dtcc)
                        if self._native_output(mesh) else None
                    )
                    xdmf_fields = self._prepare_xdmf_fields(mesh, u, p) if output_path else None
            finally:
                if self.mesh is not original_mesh:
                    self.mesh.geometry.x[:] = frame.to_input(self.mesh.geometry.x)
                coordinates = self.diagnostics.get("uncoupled_pressure_coordinates")
                if coordinates is not None:
                    self.diagnostics["uncoupled_pressure_coordinates"] = (
                        frame.to_input(np.asarray(coordinates).reshape(-1, 3))
                    ).tolist()
        return self._finalize_result(
            mesh, u, p, output_path=output_path,
            native_result=native_result, xdmf_fields=xdmf_fields,
        )

    @timed("solver")
    def _solve_stationary(self, started):
        """Solve and check flow on the private local coordinate frame."""
        mesh, params = self.mesh, self.params
        self._maybe_fetch_weather()
        self._validate_parameters()
        params = self.params
        inlet_marker, outlet_marker = select_inlet_outlet(params)
        inlet_markers, outlet_markers = classify_lateral_faces(params)
        self._has_outlet = bool(outlet_markers)
        _validate_bbox_markers(mesh, self.markers, inlet_markers, outlet_markers, self._has_outlet)
        self.num_buildings = infer_num_buildings(mesh, self.markers)
        self.category_markers = categorize_boundary(mesh, self.markers, self.num_buildings, inlet_markers, outlet_markers)
        if self.inlet_expression is not None:
            inlet = self.coordinate_frame.input_callback(self.inlet_expression)
        elif params.uses_wall_distance:
            inlet = None  # terrain-relative equilibrium profile built from the ground distance
        else:
            inlet = make_inlet_velocity_expression(params)
        system = _MixedFlowSystem(mesh, params, self.markers, self.category_markers, inlet)
        is_nonlinear = params.equations == "navier_stokes"
        override = (params.petsc_options if is_nonlinear else params.petsc_stokes) or {}
        linear = {
            "ksp_type": "fgmres", "ksp_rtol": 1e-10, "ksp_atol": 1e-13,
            "ksp_max_it": 500, "ksp_gmres_restart": 100,
        }
        if override.get("pc_type", "fieldsplit") == "fieldsplit":
            linear.update(DEFAULT_STATIONARY_PETSC)
            if params.turbulence_model != "constant":
                linear.update(VARIABLE_VISCOSITY_PETSC)
        options = {
            **linear, "snes_type": "newtonls", "snes_linesearch_type": "bt",
            "snes_ksp_ew": is_nonlinear and override.get("ksp_type", linear["ksp_type"]) != "preonly",
            "snes_ksp_ew_version": 2, "snes_ksp_ew_rtol0": .1, "snes_ksp_ew_rtolmax": .1,
            "snes_rtol": params.nonlinear_rtol if is_nonlinear else params.residual_rtol * .1,
            "snes_atol": params.nonlinear_atol if is_nonlinear else params.residual_atol * .1,
            "snes_max_it": params.nonlinear_max_it if is_nonlinear else 1,
            **override,
        }
        if options["ksp_type"] == "preonly":
            for key in ("ksp_rtol", "ksp_atol", "ksp_max_it", "ksp_gmres_restart"):
                options.pop(key, None)
        elif options["ksp_type"] not in {"gmres", "fgmres"}:
            options.pop("ksp_gmres_restart", None)
        if options["snes_type"] != "newtonls" or options["snes_linesearch_type"] != "bt":
            raise ValueError("Stationary flow requires SNES newtonls with bt line search")
        self.diagnostics["linear_solver_options"] = dict(options)
        timings = {}
        try:
            for refinement in range(params.mesh_refinement_levels + 1):
                try:
                    system.configure(options)
                    self.diagnostics["linear_solver_options"] = dict(system.effective_options)
                    break
                except _PressureSpaceError as exc:
                    self.diagnostics["uncoupled_pressure_coordinates"] = exc.coordinates.tolist()
                    if refinement >= params.mesh_refinement_levels:
                        raise
                    system.close()
                    mesh, self.markers = _refine_pressure_mesh(mesh, self.markers, exc.coordinates)
                    self.mesh = mesh
                    self.category_markers = categorize_boundary(mesh, self.markers, self.num_buildings, inlet_markers, outlet_markers)
                    system = _MixedFlowSystem(mesh, params, self.markers, self.category_markers, inlet)
                    self.diagnostics["mesh_refinements"] = self.diagnostics.get("mesh_refinements", 0) + 1
            timings["setup_s"] = time.perf_counter() - started
            initial, compatible = system.residual()
            limit = max(params.residual_atol, params.residual_rtol * max(initial, 1.))
            if compatible > limit:
                raise RuntimeError(f"Incompatible prescribed boundary flux: pressure compatibility residual={compatible:.6e}")
            info("Initializing stationary flow with a Stokes solve")
            stage_start = time.perf_counter()
            initialization = system.solve(0.)
            init_residual, compatible = system.residual()
            timings["initialization_s"] = time.perf_counter() - stage_start
            if max(init_residual, compatible) > limit:
                raise RuntimeError(f"Stokes initialization residual {max(init_residual, compatible):.6e} exceeds {limit:.6e}")
            self.diagnostics["initialization"] = {**initialization, "residual_norm": init_residual}
            final = initialization
            if is_nonlinear:
                system.set_stage(1.)
                initial, _ = system.residual()
                limit = max(params.residual_atol, params.residual_rtol * max(initial, 1.))
                info("Solving stationary Navier–Stokes with full convection")
                stage_start = time.perf_counter()
                final = system.solve(1.)
                timings["nonlinear_s"] = time.perf_counter() - stage_start
            with timing_phase("validation"):
                residual, compatible = system.residual()
                u, p = system.fields()
                finite_local = bool(np.isfinite(u.x.array).all() and np.isfinite(p.x.array).all())
                finite = mesh.comm.allreduce(finite_local, op=MPI.LAND)
                volume = system.integral(Constant(mesh, PETSc.ScalarType(1.)) * dx)
                div_rms = np.sqrt(max(0., system.integral(div(u)**2 * dx)) / volume)
                u_rms = np.sqrt(max(0., system.integral(inner(u, u) * dx)) / volume)
                hmin = float(mesh.hmin())
                dimensionless_divergence = hmin * div_rms / max(u_rms, 1e-14)
                n = FacetNormal(mesh)
                net_flux = system.integral(dot(u, n) * ds)
                absolute_flux = system.integral(abs(dot(u, n)) * ds)
                flux_imbalance = abs(net_flux) / max(absolute_flux, 1e-14)
                solid_measure = system.ds(int(BndCat.WALL)) + system.ds(int(BndCat.ROOF)) + system.ds(int(BndCat.GROUND))
                solid_area = system.integral(Constant(mesh, PETSc.ScalarType(1.)) * solid_measure)
                wall_normal = system.wall_normal  # the normal along which no penetration is imposed
                wall_rms = np.sqrt(max(0., system.integral(dot(u, wall_normal)**2 * solid_measure)) / max(solid_area, 1e-14))
                open_measure = sum((system.ds(int(tag)) for tag in [BndCat.OUTLET, BndCat.SIDE, BndCat.TOP]), system.ds(int(BndCat.INLET)))
                inflow_flux = system.integral(ufl.max_value(-dot(u, n), 0.) * open_measure)
                leakage = {}
                for name, tags in [("ground", [BndCat.GROUND]), ("building", [BndCat.WALL, BndCat.ROOF])]:
                    measure = sum((system.ds(int(tag)) for tag in tags[1:]), system.ds(int(tags[0])))
                    area = system.integral(Constant(mesh, PETSc.ScalarType(1.)) * measure)
                    leakage[name] = {
                        "flux": system.integral(abs(dot(u, wall_normal)) * measure),
                        "facet_flux": system.integral(abs(dot(u, n)) * measure),
                        "rms": float(np.sqrt(max(0., system.integral(dot(u, wall_normal)**2 * measure)) / max(area, 1e-14))),
                    }
                    leakage[name]["fraction"] = leakage[name]["flux"] / max(inflow_flux, 1e-14)
                slip_measure = system.ds(int(BndCat.SIDE)) + system.ds(int(BndCat.TOP))
                slip_area = system.integral(Constant(mesh, PETSc.ScalarType(1.)) * slip_measure)
                slip_rms = np.sqrt(max(0., system.integral(dot(u, n)**2 * slip_measure)) / max(slip_area, 1e-14)) if params.side_top_boundary == "slip" else None
            convergence = {
                "residual_norm": residual, "initial_residual_norm": initial,
                "residual_normalization": max(initial, 1.), "residual_limit": limit,
                "pressure_compatibility_residual": compatible,
                "divergence_rms": float(div_rms), "dimensionless_divergence": float(dimensionless_divergence),
                "divergence_tolerance": params.divergence_tolerance,
                "net_boundary_flux": net_flux, "absolute_boundary_flux": absolute_flux,
                "flux_imbalance": flux_imbalance, "flux_imbalance_tolerance": params.flux_imbalance_tolerance,
                "wall_normal_velocity_rms": float(wall_rms), "slip_normal_velocity_rms": slip_rms,
                "inflow_flux": inflow_flux,
                "ground_leakage_flux": leakage["ground"]["flux"], "ground_normal_velocity_rms": leakage["ground"]["rms"],
                "ground_leakage_fraction": leakage["ground"]["fraction"],
                "ground_facet_flux": leakage["ground"]["facet_flux"],
                "ground_leakage_tolerance": params.ground_leakage_tolerance,
                "building_leakage_flux": leakage["building"]["flux"], "building_normal_velocity_rms": leakage["building"]["rms"],
                "building_leakage_fraction": leakage["building"]["fraction"],
                "building_facet_flux": leakage["building"]["facet_flux"],
                "building_leakage_tolerance": params.building_leakage_tolerance,
                "finite_fields": bool(finite),
            }
            gated = ["residual_norm", "dimensionless_divergence", "flux_imbalance",
                     "ground_leakage_fraction", "building_leakage_fraction"]
            accepted = (finite and np.isfinite([convergence[k] for k in gated]).all()
                        and max(residual, compatible) <= limit
                        and dimensionless_divergence <= params.divergence_tolerance
                        and flux_imbalance <= params.flux_imbalance_tolerance
                        and leakage["ground"]["fraction"] <= params.ground_leakage_tolerance
                        and leakage["building"]["fraction"] <= params.building_leakage_tolerance)
            timings["total_s"] = time.perf_counter() - started
            self.diagnostics.update({
                "scheme": "mixed_stationary", "stop_reason": "converged" if accepted else "quality_failure",
                "nonlinear_iterations": final["nonlinear_iterations"] if is_nonlinear else 0,
                "snes_reason": final["snes_reason"], "wind_speed": params.wind_speed,
                "wind_dir_deg": params.wind_dir_deg, "wind_vector_xy": params.wind_vector_xy,
                "inlet_scale": 1., "convection_scale": 1. if is_nonlinear else 0.,
                "inlet_marker": inlet_marker, "outlet_marker": outlet_marker,
                "inlet_markers": inlet_markers, "outlet_markers": outlet_markers,
                "closed_cavity": params.closed_cavity, "wall_model": params.wall_model,
                "side_top_boundary": params.side_top_boundary, "inlet_profile": params.inlet_profile,
                "rho": params.rho, "nu": params.nu, "nu_t": params.nu_t, "nu_eff": params.nu_eff if params.turbulence_model == "constant" else None,
                "num_buildings": self.num_buildings, "mesh_hmin": hmin,
                "turbulence": system.closure_diagnostics(),
                "pressure_gauge": "zero_mean" if system.nullspace else "traction_determined",
                "degrees_of_freedom": {"velocity": self._function_dof_count(u), "pressure": self._function_dof_count(p)},
                "boundary_category_counts": self._boundary_category_counts(mesh),
                "fields": {"velocity": {"dim": 3, "unit": "m/s", **self._function_stats(u)},
                           "pressure": {"dim": 1, "unit": "m^2/s^2", **self._function_stats(p)},
                           "speed": {"dim": 1, "unit": "m/s"}},
                "convergence": convergence, "linear_solves": {"mixed": self._ksp_info(system.problem.solver.getKSP())},
                "timings": timings,
            })
            if not accepted:
                raise RuntimeError(f"Stationary flow quality checks failed: {convergence}")
            info(f"Stationary flow converged: residual={residual:.6e} dimensionless_divergence={dimensionless_divergence:.6e} flux_imbalance={flux_imbalance:.6e} elapsed={timings['total_s']:.3f}s")
        except Exception as exc:
            snes = system.problem.solver if system.problem else None
            self.diagnostics.update({"stop_reason": "quality_failure" if self.diagnostics.get("stop_reason") == "quality_failure" else "solver_failure",
                                     "error": str(exc), "elapsed_s": time.perf_counter() - started,
                                     "snes_reason": int(snes.getConvergedReason()) if snes else None,
                                     "linear_solves": {"mixed": self._ksp_info(snes.getKSP())} if snes else {}})
            warning(f"Stationary flow solve failed: {exc}")
            raise
        finally:
            self.diagnostics["nonlinear_history"] = list(system.history)
            system.close()
        return mesh, u, p
