"""Urban Wind CFD Solver (IPCS / ABCN Fractional-Step Method)

Solves the incompressible Navier–Stokes equations in pseudo-time using an
Incremental Pressure-Correction Scheme (IPCS) with Adams–Bashforth /
Crank–Nicolson (ABCN) time discretization on DTCC city air-volume meshes.

Physical model
--------------
    ∂u/∂t + (u·∇)u = −∇p̃ + ν_eff Δu      (momentum)
    ∇·u = 0                                  (continuity)

where p̃ = p/ρ is the *kinematic* pressure (units m²/s²).  The density ρ
does not appear in the equations; all output pressures are kinematic.

Output
------
A ``dtcc_core.model.VolumeMesh`` with attached ``Field`` objects:
- ``velocity`` (dim=3, unit m/s)
- ``pressure`` (dim=1, unit m²/s² — kinematic pressure)
- ``speed``    (dim=1, unit m/s)

References
----------
- Simo & Armero (1994) — Unconditional stability and long-term behavior of
  transient algorithms for the incompressible Navier–Stokes equations.
- Oasis / OasisX — Mikael Mortensen's fractional step solvers for FEniCS(x).
"""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, Field

import dolfinx
import dolfinx.fem
import dolfinx.la
import dolfinx.mesh
from dolfinx.fem import (
    Constant,
    Expression,
    Function,
    dirichletbc,
    form as _fem_form,
    locate_dofs_topological,
)
from dolfinx.fem import petsc as _fem_petsc
from dolfinx.mesh import locate_entities_boundary, meshtags
from mpi4py import MPI
from petsc4py import PETSc
import basix
import basix.ufl
import ufl
from ufl import (
    TrialFunction,
    TestFunction,
    SpatialCoordinate,
    FacetNormal,
    Measure,
    as_vector,
    dot,
    inner,
    grad,
    div,
    dx,
    ds,
)

from dtcc_sim.fenics import (
    FunctionSpace,
    info,
    warning,
    load_mesh_with_markers,
    BoxMesh,
    bounds as mesh_bounds,
    offset_to_origin,
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


# Marker → bbox face mapping (dtcc-core volumemesh convention)
BBOX_MARKER_NORMALS: Dict[int, Tuple[float, float, float]] = {
    -3: (-1.0, 0.0, 0.0),  # xmin face
    -4: (1.0, 0.0, 0.0),  # xmax face
    -5: (0.0, -1.0, 0.0),  # ymin face
    -6: (0.0, 1.0, 0.0),  # ymax face
    -2: (0.0, 0.0, 1.0),  # top face
}


# ---------------------------------------------------------------------------
# Default PETSc options
# ---------------------------------------------------------------------------

DEFAULT_VELOCITY_PETSC: Dict[str, Any] = {
    "ksp_type": "gmres",
    "ksp_rtol": 1.0e-6,
    "ksp_monitor": None,
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
}

DEFAULT_PRESSURE_PETSC: Dict[str, Any] = {
    "ksp_type": "cg",
    "ksp_rtol": 1.0e-6,
    "ksp_monitor": None,
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
}


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


class UrbanWindParameters(BaseModel):
    """Parameters for the urban wind CFD solver.

    Groups
    ------
    Physical, wind forcing, mesh, FE/time-stepping, solver scheme,
    boundary-condition model, PETSc options.
    """

    # ---- 2.1 Physical ----
    rho: float = Field(1.2, description="Air density [kg/m³]", gt=0)
    nu: float = Field(1.5e-5, description="Kinematic viscosity [m²/s]", gt=0)
    nu_t: float = Field(
        0.0, description="Eddy viscosity placeholder [m²/s] (v1 laminar → 0)", ge=0
    )

    # ---- 2.2 Wind forcing ----
    use_weather: bool = Field(
        False,
        description="Fetch wind from dtcc_core.datasets.weather()",
    )
    weather_aggregation: Literal["nearest", "mean", "median"] = Field(
        "nearest", description="Weather aggregation strategy"
    )
    wind_speed: float = Field(5.0, description="Reference wind speed [m/s]", ge=0)
    wind_dir_deg: float = Field(
        270.0,
        description=(
            "Wind direction in meteorological convention: direction "
            "wind is coming FROM in degrees (0=N, 90=E, 180=S, 270=W)"
        ),
    )

    # ---- 2.3 Mesh parameters ----
    mesh_max_mesh_size: float = Field(25.0, description="Max mesh element size [m]")
    mesh_domain_height: float = Field(80.0, description="Domain height [m]")
    mesh_raster_cell_size: float = Field(2.0, description="Terrain raster cell size")
    mesh_raster_radius: float = Field(
        3.0, description="Terrain raster interpolation radius"
    )

    # ---- 2.4 FE / time-stepping ----
    velocity_degree: int = Field(2, description="Velocity FE degree (Taylor-Hood P2)")
    pressure_degree: int = Field(1, description="Pressure FE degree (Taylor-Hood P1)")
    dt: float = Field(0.2, description="Pseudo-time step [s]", gt=0)
    max_steps: int = Field(2000, description="Maximum number of time steps", gt=0)
    simulation_mode: Literal["steady", "statistical_steady"] = Field(
        "steady",
        description=(
            "Stopping mode. 'steady' targets fixed-point convergence of "
            "the flow field. 'statistical_steady' targets stationarity of "
            "flow statistics for inherently unsteady bluff-body flows."
        ),
    )
    steady_tolerance: float = Field(
        1.0e-4, description="Relative velocity-change tolerance (steady mode)"
    )
    divergence_tolerance: float = Field(
        5.0e-3,
        description="L2(div u) tolerance used in steady-mode stopping",
        ge=0.0,
    )
    flux_imbalance_tolerance: float = Field(
        5.0e-2,
        description="Net open-boundary flux imbalance tolerance in steady mode",
        ge=0.0,
    )
    min_steps: int = Field(50, description="Minimum steps before early stopping", ge=0)
    steady_window: int = Field(
        5, description="Consecutive converged steps before stopping", ge=1
    )
    stat_warmup_steps: int = Field(
        100,
        description="Warm-up steps before evaluating statistical stationarity",
        ge=0,
    )
    stat_window: int = Field(
        50,
        description="Window size (steps) for statistical stationarity checks",
        ge=5,
    )
    stat_tolerance: float = Field(
        1.0e-2,
        description="Relative change tolerance between consecutive statistic windows",
        ge=0.0,
    )
    stat_divergence_tolerance: float = Field(
        1.0e-2,
        description="Mean L2(div u) tolerance for statistical-steady stopping",
        ge=0.0,
    )
    stat_flux_imbalance_tolerance: float = Field(
        5.0e-2,
        description="Mean flux-imbalance tolerance for statistical-steady stopping",
        ge=0.0,
    )
    cfl_target: float = Field(
        2.0,
        description=(
            "Advisory CFL target used for diagnostics/warnings. "
            "The solver does not auto-adjust dt yet."
        ),
        gt=0.0,
    )
    adaptive_dt: bool = Field(
        False,
        description="Enable CFL-based adaptive pseudo-time stepping.",
    )
    dt_min: float = Field(
        1.0e-3,
        description="Minimum adaptive pseudo-time step [s]",
        gt=0.0,
    )
    dt_max: Optional[float] = Field(
        None,
        description="Maximum adaptive pseudo-time step [s] (None -> initial dt).",
    )
    cfl_reduce_safety: float = Field(
        0.8,
        description="Safety factor used when reducing dt from CFL overflow.",
        gt=0.0,
        le=1.0,
    )
    cfl_increase_factor: float = Field(
        1.05,
        description="Gentle multiplicative dt growth when CFL is well below target.",
        gt=1.0,
    )
    cfl_hard_limit: float = Field(
        8.0,
        description=(
            "Hard CFL threshold. If exceeded, extra damping and aggressive "
            "dt reduction are applied."
        ),
        gt=0.0,
    )
    log_every_steps: int = Field(
        10,
        description="Emit solver progress every N steps",
        ge=1,
    )
    log_initial_steps: int = Field(
        5,
        description="Always emit solver progress for the first N steps",
        ge=0,
    )

    # ---- 2.5 Solver scheme ----
    scheme: Literal["IPCS_ABCN"] = Field("IPCS_ABCN", description="Solver scheme")
    convective_form: Literal["standard", "skew_symmetric"] = Field(
        "standard",
        description=(
            "Convection treatment.  'standard' is the natural Galerkin form "
            "(u_conv·∇)u·v which transports energy through open boundaries "
            "and is stable for inlet/outlet flows.  'skew_symmetric' conserves "
            "kinetic energy exactly but prevents convective energy removal at "
            "outlets, so it requires sufficient eddy viscosity (nu_t) to avoid "
            "energy accumulation and divergence."
        ),
    )
    convection_linearization: Literal["picard", "ab2"] = Field(
        "picard",
        description=(
            "Linearization of the convective velocity: "
            "'picard' uses u_n (robust, recommended for steady state), "
            "'ab2' uses 1.5*u_n - 0.5*u_{n-1} (time-accurate but may oscillate)."
        ),
    )
    velocity_relaxation: float = Field(
        1.0,
        description=(
            "Under-relaxation factor ω for the velocity update: "
            "u^{n+1} = ω u* + (1-ω) u^n.  Use <1 (e.g. 0.5) "
            "to damp oscillations for complex geometries."
        ),
        gt=0.0,
        le=1.0,
    )
    spike_rel_threshold: float = Field(
        0.5,
        description=(
            "If relative update exceeds this threshold, apply extra "
            "damping via spike_relaxation for that step."
        ),
        ge=0.0,
    )
    spike_relaxation: float = Field(
        0.1,
        description=(
            "Temporary velocity relaxation used on spike steps "
            "(rel > spike_rel_threshold)."
        ),
        gt=0.0,
        le=1.0,
    )
    grad_div_gamma: float = Field(
        0.0,
        description=(
            "Grad-div stabilization strength γ for the tentative-velocity "
            "solve: adds γ(∇·u, ∇·v). Helps suppress divergence-induced "
            "instabilities in high-Re complex flows."
        ),
        ge=0.0,
    )
    backflow_beta: float = Field(
        0.0,
        description=(
            "Backflow stabilization on open boundaries (outlet/side/top). "
            "Adds beta*max(-u_conv·n,0)*(u,v) weak damping to suppress "
            "spurious recirculation blow-ups."
        ),
        ge=0.0,
    )
    side_top_boundary: Literal["open", "slip"] = Field(
        "open",
        description=(
            "Boundary model for SIDE/TOP categories. "
            "'open' uses natural traction (allows through-flow). "
            "'slip' enforces no-penetration with free tangential motion."
        ),
    )

    # ---- 2.6 Boundary-condition model ----
    wall_model: Literal["noslip", "friction"] = Field(
        "noslip", description="Wall treatment model"
    )
    beta_wall: float = Field(
        0.5, description="Tangential friction coefficient (friction model)"
    )
    gamma_normal: Optional[float] = Field(
        None, description="Normal penalty (None → auto from ν_eff/h)"
    )
    inlet_profile: Literal["uniform", "power_law", "log_law"] = Field(
        "uniform", description="Inlet velocity profile shape"
    )
    inlet_ramp_steps: int = Field(
        0,
        description=(
            "Number of initial pseudo-time steps to linearly ramp inlet "
            "velocity from 0 to full value. Useful for startup stability."
        ),
        ge=0,
    )
    inlet_ramp_time: Optional[float] = Field(
        None,
        description=(
            "Physical pseudo-time [s] for inlet ramping. "
            "If set, it overrides inlet_ramp_steps."
        ),
        gt=0.0,
    )
    spike_min_step: int = Field(
        20,
        description="First step at which spike damping based on rel is active.",
        ge=0,
    )
    z0: float = Field(0.5, description="Roughness length [m] for log-law", gt=0)
    u_ref_height: float = Field(
        10.0, description="Reference measurement height [m]", gt=0
    )
    power_law_alpha: float = Field(0.2, description="Power-law exponent α", gt=0)

    # Manual marker overrides
    inlet_marker: Optional[int] = Field(
        None, description="Force inlet boundary marker (skip auto detection)"
    )
    outlet_marker: Optional[int] = Field(
        None,
        description=(
            "Force outlet boundary marker (skip auto detection). "
            "Use the string 'none' or set closed_cavity=True for "
            "fully enclosed domains with no outlet."
        ),
    )
    closed_cavity: bool = Field(
        False,
        description=(
            "If True, no pressure-outlet BC is applied; pressure "
            "is constrained via a PETSc null-space (zero-mean).  "
            "Use for fully enclosed domains (lid-driven cavity) or "
            "when no outlet face exists.  Note: combining this with "
            "a net-inflow inlet BC may violate incompressibility; "
            "ensure the inlet profile integrates to zero net flux or "
            "pair with an outlet."
        ),
    )

    # ---- 2.7 PETSc options ----
    petsc_velocity: Optional[Dict[str, Any]] = Field(
        None, description="PETSc options for velocity sub-solve"
    )
    petsc_pressure: Optional[Dict[str, Any]] = Field(
        None, description="PETSc options for pressure sub-solve"
    )

    @property
    def nu_eff(self) -> float:
        """Effective viscosity (kinematic + eddy)."""
        return self.nu + self.nu_t

    @property
    def wind_vector_xy(self) -> Tuple[float, float]:
        """Compute horizontal wind direction unit vector (flow direction).

        Meteorological convention: wind_dir_deg is where wind comes FROM.
        Flow direction = wind_dir_deg + 180°.
        North = 0°, East = 90° → standard math angle measured CW from north.
        Convert: math_angle = 90 - met_angle (then to radians).
        """
        flow_deg = self.wind_dir_deg + 180.0
        # Met convention: 0=N, 90=E. Convert to math angle (CCW from +x)
        math_rad = np.deg2rad(90.0 - flow_deg)
        return (float(np.cos(math_rad)), float(np.sin(math_rad)))


# ---------------------------------------------------------------------------
# Helpers — boundary handling
# ---------------------------------------------------------------------------


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
    inlet_marker: int,
    outlet_marker: Optional[int],
    ground_tag: int = -1,
    top_tag: int = -2,
) -> dolfinx.mesh.MeshTags:
    """Collapse per-building markers into wind-specific categories.

    Returns MeshTags with values from :class:`BndCat`.
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
    mask_inlet = vals == inlet_marker
    mask_outlet = (
        vals == outlet_marker
        if outlet_marker is not None
        else np.zeros(len(vals), dtype=bool)
    )

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


def select_inlet_outlet(
    params: UrbanWindParameters,
    mesh: Optional[dolfinx.mesh.Mesh] = None,
    markers: Optional[dolfinx.mesh.MeshTags] = None,
) -> Tuple[int, Optional[int]]:
    """Choose inlet and outlet bbox markers from wind direction.

    Uses the known marker→normal mapping in :data:`BBOX_MARKER_NORMALS`.
    The top face (-2) is excluded.

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

    marker_normals = BBOX_MARKER_NORMALS
    if mesh is not None and markers is not None:
        inferred_normals = _infer_bbox_marker_normals(mesh, markers)
        if len(inferred_normals) >= 2:
            marker_normals = inferred_normals
            if mesh.comm.rank == 0:
                info(
                    "UrbanWind: inferred lateral bbox marker orientations: "
                    + ", ".join(
                        f"{m}->{tuple(np.round(n, 3))}"
                        for m, n in sorted(marker_normals.items())
                    )
                )
        elif mesh.comm.rank == 0:
            warning(
                "UrbanWind: could not infer lateral bbox marker orientations "
                "from mesh; falling back to built-in marker mapping."
            )

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


def _infer_bbox_marker_normals(
    mesh: dolfinx.mesh.Mesh,
    markers: dolfinx.mesh.MeshTags,
    top_tag: int = -2,
    ground_tag: int = -1,
) -> Dict[int, Tuple[float, float, float]]:
    """Infer lateral bbox marker normals from facet centroids.

    This makes inlet/outlet selection robust to mesh-generator marker
    permutations (e.g. when -3/-4/-5/-6 do not match the assumed order).
    """
    g = mesh.geometry.x
    if g.shape[0] > 0:
        local_xmin = float(np.min(g[:, 0]))
        local_xmax = float(np.max(g[:, 0]))
        local_ymin = float(np.min(g[:, 1]))
        local_ymax = float(np.max(g[:, 1]))
    else:
        local_xmin = float("inf")
        local_xmax = float("-inf")
        local_ymin = float("inf")
        local_ymax = float("-inf")
    xmin = float(mesh.comm.allreduce(local_xmin, op=MPI.MIN))
    xmax = float(mesh.comm.allreduce(local_xmax, op=MPI.MAX))
    ymin = float(mesh.comm.allreduce(local_ymin, op=MPI.MIN))
    ymax = float(mesh.comm.allreduce(local_ymax, op=MPI.MAX))

    vals = np.asarray(markers.values, dtype=np.int32)
    local_markers = sorted(
        set(int(v) for v in vals if v < ground_tag and int(v) != int(top_tag))
    )
    all_markers = mesh.comm.allgather(local_markers)
    lateral_markers = sorted(set().union(*[set(v) for v in all_markers]))

    fdim = mesh.topology.dim - 1
    inferred: Dict[int, Tuple[float, float, float]] = {}
    for marker in lateral_markers:
        facets = markers.find(int(marker))
        if len(facets) > 0:
            mids = dolfinx.mesh.compute_midpoints(mesh, fdim, facets)
            local_sum_x = float(np.sum(mids[:, 0]))
            local_sum_y = float(np.sum(mids[:, 1]))
            local_count = int(len(facets))
        else:
            local_sum_x = 0.0
            local_sum_y = 0.0
            local_count = 0

        sum_x = float(mesh.comm.allreduce(local_sum_x, op=MPI.SUM))
        sum_y = float(mesh.comm.allreduce(local_sum_y, op=MPI.SUM))
        count = int(mesh.comm.allreduce(local_count, op=MPI.SUM))
        if count == 0:
            continue

        cx = sum_x / count
        cy = sum_y / count
        dists = {
            "xmin": abs(cx - xmin),
            "xmax": abs(cx - xmax),
            "ymin": abs(cy - ymin),
            "ymax": abs(cy - ymax),
        }
        face = min(dists, key=dists.get)
        if face == "xmin":
            inferred[int(marker)] = (-1.0, 0.0, 0.0)
        elif face == "xmax":
            inferred[int(marker)] = (1.0, 0.0, 0.0)
        elif face == "ymin":
            inferred[int(marker)] = (0.0, -1.0, 0.0)
        else:
            inferred[int(marker)] = (0.0, 1.0, 0.0)

    return inferred


# ---------------------------------------------------------------------------
# Helpers — marker validation
# ---------------------------------------------------------------------------


def _validate_bbox_markers(
    mesh: dolfinx.mesh.Mesh,
    markers: dolfinx.mesh.MeshTags,
    inlet_marker: int,
    outlet_marker: Optional[int],
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

    # Check inlet marker
    if inlet_marker not in global_available:
        raise RuntimeError(
            f"UrbanWind: inlet marker {inlet_marker} not found in mesh tags.  "
            f"Available markers: {sorted(global_available)}.  "
            f"Expected BBOX_MARKER_NORMALS keys: {sorted(BBOX_MARKER_NORMALS.keys())}."
        )

    # Check outlet marker
    if (
        has_outlet
        and outlet_marker is not None
        and outlet_marker not in global_available
    ):
        raise RuntimeError(
            f"UrbanWind: outlet marker {outlet_marker} not found in mesh tags.  "
            f"Available markers: {sorted(global_available)}.  "
            f"Set closed_cavity=True if no outlet face exists."
        )

    # Warn if none of the expected bbox markers are present
    expected_bbox = set(BBOX_MARKER_NORMALS.keys())
    if not expected_bbox.intersection(global_available):
        warning(
            f"UrbanWind: none of the expected bbox markers "
            f"{sorted(expected_bbox)} found in mesh tags "
            f"{sorted(global_available)}.  Inlet/outlet selection may be wrong."
        )


# ---------------------------------------------------------------------------
# Helpers — inlet velocity profile
# ---------------------------------------------------------------------------


def make_inlet_velocity_expression(
    params: UrbanWindParameters,
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
        z_ref = params.u_ref_height
        denom = np.log((z_ref + z0) / z0)

        def _expr(x: np.ndarray) -> np.ndarray:
            n = x.shape[1]
            z = np.maximum(x[2], 0.0)
            mag = U_ref * np.log((z + z0) / z0) / denom
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


def _dolfinx_to_volume_mesh(
    dolfinx_mesh: dolfinx.mesh.Mesh,
    u_sol: Function,
    p_sol: Function,
    volume_mesh_dtcc: Any,
) -> Any:
    """Map dolfinx solution fields onto a dtcc-core VolumeMesh.

    Interpolates P2 velocity to P1, extracts arrays, and attaches
    ``velocity``, ``pressure``, and ``speed`` Fields.
    """
    from dtcc_core.model import Field as DtccField

    # Create P1 vector space and interpolate velocity
    V1 = FunctionSpace(dolfinx_mesh, "Lagrange", 1, dim=3)
    u1 = Function(V1)
    u1.interpolate(u_sol)

    # Pressure is already P1
    p1 = p_sol

    # ---- extract dof coordinates & values ----
    # Scalar P1 dof coords
    Q1 = p1.function_space
    dof_coords_q = Q1.tabulate_dof_coordinates()  # (ndof, 3)

    # Velocity dof coords (every component shares the same geometric dofs for
    # vector Lagrange elements; the first block of dof_coordinates corresponds
    # to the x-component and repeats for y,z)
    dof_coords_v = V1.tabulate_dof_coordinates()  # (ndof_per_comp, 3)

    # Number of dofs per scalar component
    ndof_comp = V1.dofmap.index_map.size_local
    # velocity stored as interleaved or blocked depending on V1 block size
    bs = V1.dofmap.index_map_bs
    u_arr = u1.x.array[: ndof_comp * bs].reshape(ndof_comp, bs)
    p_arr = p1.x.array[: Q1.dofmap.index_map.size_local].copy()

    # ---- map to dtcc-core vertex ordering ----
    dtcc_verts = np.asarray(volume_mesh_dtcc.vertices, dtype=np.float64)  # (M,3)

    # Build coordinate → index mapping via rounding for robustness
    decimals = 6
    dtcc_keys = np.round(dtcc_verts, decimals)
    fem_keys_v = np.round(dof_coords_v[:ndof_comp], decimals)
    fem_keys_q = np.round(dof_coords_q[: Q1.dofmap.index_map.size_local], decimals)

    # Fast lexsort-based reorder for velocity
    u_mapped = _reorder_by_coords(fem_keys_v, u_arr, dtcc_keys)
    p_mapped = _reorder_by_coords(fem_keys_q, p_arr.reshape(-1, 1), dtcc_keys).ravel()

    speed = np.linalg.norm(u_mapped, axis=1)

    # Attach fields
    volume_mesh_dtcc.fields = [
        DtccField(
            name="velocity",
            dim=3,
            values=u_mapped,
            unit="m/s",
            description="Velocity field from urban wind CFD solver",
        ),
        DtccField(
            name="pressure",
            dim=1,
            values=p_mapped,
            unit="m^2/s^2",
            description="Pressure field from urban wind CFD solver",
        ),
        DtccField(
            name="speed",
            dim=1,
            values=speed,
            unit="m/s",
            description="Wind speed |u| from urban wind CFD solver",
        ),
    ]
    return volume_mesh_dtcc


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


def _relative_change(a: float, b: float, eps: float = 1e-14) -> float:
    """Compute |a-b|/max(max(|a|,|b|), eps)."""
    denom = max(max(abs(a), abs(b)), eps)
    return abs(a - b) / denom


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class UrbanWindSimulator:
    """Incompressible Navier–Stokes solver for urban wind simulation.

    Implements the IPCS (Incremental Pressure-Correction Scheme) with
    Adams–Bashforth / Crank–Nicolson time discretisation.  The solver
    marches in pseudo-time until a steady state is detected or the step
    limit is reached.

    Usage
    -----
    >>> sim = UrbanWindSimulator(bounds=bounds)
    >>> out_mesh = sim.simulate()   # dtcc-core VolumeMesh with Fields

    >>> params = UrbanWindParameters(wind_speed=8.0, wind_dir_deg=240.0)
    >>> sim = UrbanWindSimulator(bounds=bounds, params=params)
    >>> out_mesh = sim.simulate()
    """

    def __init__(
        self,
        *,
        bounds: Optional[Any] = None,
        mesh_path: Optional[str] = None,
        mesh: Optional[dolfinx.mesh.Mesh] = None,
        markers: Optional[dolfinx.mesh.MeshTags] = None,
        params: Optional[UrbanWindParameters] = None,
        inlet_expression: Optional[callable] = None,
    ) -> None:
        self.bounds = bounds
        self.mesh_path = mesh_path
        self.mesh = mesh
        self.markers = markers
        self.params = params if params is not None else UrbanWindParameters()
        self.inlet_expression = inlet_expression

        # populated during simulation
        self.volume_mesh_dtcc: Optional[Any] = None
        self.num_buildings: int = 0
        self.category_markers: Optional[dolfinx.mesh.MeshTags] = None

    # ------------------------------------------------------------------ mesh
    def _build_mesh_from_bounds(self) -> None:
        comm = MPI.COMM_WORLD
        rank = comm.rank

        info("UrbanWind: Building volume mesh from bounds …")
        try:
            import dtcc_core.datasets as datasets
        except ImportError:
            raise ImportError("dtcc_core is required to build mesh from bounds.")

        # --- serial work: only rank 0 builds the mesh and writes to disk ---
        import tempfile

        tmp_path: Optional[str] = None
        if rank == 0:
            volume_mesh = datasets.city_volume_mesh(
                bounds=self.bounds,
                max_mesh_size=self.params.mesh_max_mesh_size,
                domain_height=self.params.mesh_domain_height,
                raster_cell_size=self.params.mesh_raster_cell_size,
                raster_radius=self.params.mesh_raster_radius,
            )
            self.volume_mesh_dtcc = volume_mesh

            with tempfile.NamedTemporaryFile(suffix=".xdmf", delete=False) as tmp:
                tmp_path = tmp.name
            info(f"UrbanWind: saving mesh to temporary file: {tmp_path}")
            volume_mesh.save(tmp_path)

        # broadcast the temp path so all ranks open the *same* file
        tmp_path = comm.bcast(tmp_path, root=0)
        comm.barrier()  # ensure file is written before any rank reads

        # --- parallel read: all ranks collectively load the mesh ---
        self.mesh, self.markers = load_mesh_with_markers(tmp_path)

        # barrier *after* read so no rank deletes before others finish
        comm.barrier()

        # rank 0 cleans up
        if rank == 0:
            Path(tmp_path).unlink(missing_ok=True)
            Path(tmp_path).with_suffix(".h5").unlink(missing_ok=True)

    def _load_if_needed(self) -> None:
        if self.mesh is not None and self.markers is not None:
            return
        if self.bounds is not None:
            self._build_mesh_from_bounds()
        elif self.mesh_path is not None:
            info(f"UrbanWind: Loading mesh from {self.mesh_path}")
            self.mesh, self.markers = load_mesh_with_markers(self.mesh_path)
        else:
            raise ValueError(
                "UrbanWindSimulator: provide bounds, mesh_path, or (mesh, markers)."
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
        return UrbanWindSimulator._circular_mean_deg(angles_deg)

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

    def _maybe_fetch_weather(self) -> None:
        """Optionally override wind_speed / wind_dir_deg from SMHI weather.

        Uses the dtcc-core ``SensorCollection.to_arrays()`` API.
        """
        if not self.params.use_weather:
            return

        comm = MPI.COMM_WORLD
        rank = comm.rank
        weather_data: Optional[Tuple[float, float]] = None

        # Only rank 0 performs the HTTP fetch
        if rank == 0:
            try:
                import dtcc_core.datasets as datasets
            except ImportError:
                warning("UrbanWind: dtcc_core not available; skipping weather fetch.")
                weather_data = None
                comm.bcast(weather_data, root=0)
                return

            info("UrbanWind: Fetching weather data …")
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
                    warning(
                        "UrbanWind: weather data incomplete; using manual wind params."
                    )
            except Exception as exc:
                warning(
                    f"UrbanWind: weather fetch failed ({exc}); using manual params."
                )

        # Broadcast result to all ranks
        weather_data = comm.bcast(weather_data, root=0)

        if weather_data is not None:
            ws, wd = weather_data
            info(f"UrbanWind: weather → speed={ws:.1f} m/s, dir={wd:.0f}°")
            self.params = self.params.model_copy(
                update={"wind_speed": ws, "wind_dir_deg": wd}
            )

    # -------------------------------------------------------- main simulate
    def simulate(self, *, output_path: Optional[str] = None) -> Any:
        """Run the urban wind simulation.

        Parameters
        ----------
        output_path
            Optional XDMF path to save the velocity/pressure solution.

        Returns
        -------
        dtcc_core VolumeMesh (if bounds were given) or tuple ``(u, p)``
            of dolfinx Functions when mesh was provided directly.
        """
        self._load_if_needed()
        assert self.mesh is not None and self.markers is not None

        # ---- translate mesh to origin ----
        # Real-world coordinates (e.g. SWEREF 99 TM: x ~ 320 000,
        # y ~ 6 400 000) cause catastrophic floating-point precision
        # loss in gradient / stiffness computations.  Shift the mesh
        # so (xmin, ymin, zmin) = (0, 0, 0).
        offset_to_origin(self.mesh)

        self._maybe_fetch_weather()

        params = self.params
        mesh = self.mesh
        rank0 = mesh.comm.rank == 0
        dt_val = params.dt
        dt_max = params.dt if params.dt_max is None else params.dt_max
        nu_eff = params.nu_eff
        side_top_open = params.side_top_boundary == "open"
        side_top_slip = params.side_top_boundary == "slip"
        use_weak_slip = side_top_slip and mesh.comm.size > 1
        weak_slip_tags: List[int] = []
        weak_slip_gamma: Optional[Constant] = None

        # ---- boundary setup ----
        self.num_buildings = infer_num_buildings(mesh, self.markers)
        info(f"UrbanWind: inferred {self.num_buildings} buildings")

        inlet_marker, outlet_marker = select_inlet_outlet(params, mesh=mesh, markers=self.markers)
        self._has_outlet = outlet_marker is not None and not params.closed_cavity

        # --- Validate that expected bbox markers exist in the mesh ---
        _validate_bbox_markers(
            mesh, self.markers, inlet_marker, outlet_marker, self._has_outlet
        )

        info(
            f"UrbanWind: inlet marker = {inlet_marker}, "
            f"outlet marker = {outlet_marker}"
            f"{'' if self._has_outlet else ' (closed cavity – no outlet)'}"
        )
        info(f"UrbanWind: side/top boundary model = {params.side_top_boundary}")

        self.category_markers = categorize_boundary(
            mesh,
            self.markers,
            self.num_buildings,
            inlet_marker=inlet_marker,
            outlet_marker=outlet_marker,
        )

        # ---- function spaces (Taylor–Hood) ----
        V = FunctionSpace(mesh, "Lagrange", params.velocity_degree, dim=3)
        Q = FunctionSpace(mesh, "Lagrange", params.pressure_degree)
        info(
            f"UrbanWind: V dofs = {V.dofmap.index_map.size_global}, "
            f"Q dofs = {Q.dofmap.index_map.size_global}"
        )

        # ---- functions / state ----
        u_n = Function(V, name="u_n")  # velocity at time n
        u_nm1 = Function(V, name="u_nm1")  # velocity at time n-1
        p_n = Function(Q, name="p_n")  # pressure at time n
        u_ = Function(V, name="u_star")  # tentative velocity
        phi = Function(Q, name="phi")  # pressure correction

        # Test / trial
        v = TestFunction(V)
        q = TestFunction(Q)
        u_trial = TrialFunction(V)
        p_trial = TrialFunction(Q)

        # ---- measures ----
        ds_cat = Measure("ds", domain=mesh, subdomain_data=self.category_markers)
        n_vec = FacetNormal(mesh)

        # ---- constants ----
        dt_c = Constant(mesh, PETSc.ScalarType(dt_val))
        nu_c = Constant(mesh, PETSc.ScalarType(nu_eff))

        # ---- inlet BC ----
        u_in_func = Function(V, name="u_inlet")
        if self.inlet_expression is not None:
            inlet_expr = self.inlet_expression
        else:
            inlet_expr = make_inlet_velocity_expression(params)
        ramp_steps = max(0, int(params.inlet_ramp_steps))
        if params.inlet_ramp_time is not None:
            ramp_time = float(params.inlet_ramp_time)
        elif ramp_steps > 0:
            ramp_time = float(ramp_steps) * float(params.dt)
        else:
            ramp_time = 0.0
        if ramp_time > 0.0:
            initial_scale = min(1.0, dt_val / ramp_time)

            def _inlet_expr_scaled(x: np.ndarray, _scale=initial_scale) -> np.ndarray:
                return _scale * inlet_expr(x)

            u_in_func.interpolate(_inlet_expr_scaled)
            info(
                f"UrbanWind: inlet ramp enabled over {ramp_time:.3f} s "
                f"(initial scale={initial_scale:.3f})"
            )
        else:
            u_in_func.interpolate(inlet_expr)

        if rank0:
            info("UrbanWind: setup phase 1/5 - locating boundary dofs")

        # Locate inlet dofs
        if rank0:
            info("UrbanWind: setup phase 1a/5 - inlet dofs")
        fdim = mesh.topology.dim - 1
        inlet_facets = self.category_markers.find(int(BndCat.INLET))
        # Check globally — some ranks may have zero local inlet facets
        n_inlet_global = mesh.comm.allreduce(len(inlet_facets), op=MPI.SUM)
        if n_inlet_global == 0:
            all_cats = set(int(v) for v in self.category_markers.values)
            all_cats_global = mesh.comm.allgather(sorted(all_cats))
            merged = sorted(set().union(*[set(c) for c in all_cats_global]))
            raise RuntimeError(
                "UrbanWind: no inlet facets found after boundary categorisation.  "
                f"Available category values (global): {merged}."
            )
        inlet_dofs = locate_dofs_topological(V, fdim, inlet_facets)
        bc_inlet = dirichletbc(u_in_func, inlet_dofs)

        # ---- wall BCs ----
        if rank0:
            info("UrbanWind: setup phase 1b/5 - wall and side/top BCs")
        bcs_vel: List[dolfinx.fem.DirichletBC] = [bc_inlet]

        solid_tags = [int(BndCat.WALL), int(BndCat.ROOF), int(BndCat.GROUND)]
        if params.wall_model == "noslip":
            # Strong Dirichlet u=0 on walls+roof+ground
            u_zero = Function(V)  # default zero
            for tag in solid_tags:
                facets = self.category_markers.find(tag)
                dofs = locate_dofs_topological(V, fdim, facets)
                bcs_vel.append(dirichletbc(u_zero, dofs))

        # Optional free-slip/no-penetration on side and top bbox faces.
        # We enforce axis-aligned normal components strongly:
        # top (-2): u_z = 0
        # x-faces (-3/-4): u_x = 0, y-faces (-5/-6): u_y = 0
        # for faces not selected as inlet/outlet.
        if side_top_slip:
            if use_weak_slip:
                weak_slip_tags = [int(BndCat.SIDE), int(BndCat.TOP)]
                if rank0:
                    warning(
                        "UrbanWind: using MPI-safe weak slip enforcement on side/top "
                        "(normal-velocity penalty), replacing strong component BCs."
                    )
            else:
                if rank0:
                    info("UrbanWind: setup phase 1c/5 - applying side/top slip BCs")
                top_facets = self.markers.find(-2)
                if len(top_facets) > 0:
                    if rank0:
                        info("UrbanWind: setup phase 1c.1 - top uz dofs")
                    dofs_z = locate_dofs_topological(V.sub(2), fdim, top_facets)
                    bcs_vel.append(dirichletbc(PETSc.ScalarType(0.0), dofs_z, V.sub(2)))

                side_markers = [-3, -4, -5, -6]
                for marker in side_markers:
                    if marker == inlet_marker or marker == outlet_marker:
                        continue
                    facets = self.markers.find(marker)
                    if len(facets) == 0:
                        continue
                    if marker in (-3, -4):
                        if rank0:
                            info(
                                f"UrbanWind: setup phase 1c.2 - side marker {marker} ux dofs"
                            )
                        dofs_x = locate_dofs_topological(V.sub(0), fdim, facets)
                        bcs_vel.append(
                            dirichletbc(PETSc.ScalarType(0.0), dofs_x, V.sub(0))
                        )
                    else:
                        if rank0:
                            info(
                                f"UrbanWind: setup phase 1c.3 - side marker {marker} uy dofs"
                            )
                        dofs_y = locate_dofs_topological(V.sub(1), fdim, facets)
                        bcs_vel.append(
                            dirichletbc(PETSc.ScalarType(0.0), dofs_y, V.sub(1))
                        )

        # Pressure BC
        if rank0:
            info("UrbanWind: setup phase 1d/5 - pressure BC setup")
        if self._has_outlet:
            # phi=0 on outlet face
            outlet_facets = self.category_markers.find(int(BndCat.OUTLET))
            # Check globally — some ranks may have zero local outlet facets
            n_outlet_global = mesh.comm.allreduce(len(outlet_facets), op=MPI.SUM)
            if n_outlet_global == 0:
                raise RuntimeError(
                    "UrbanWind: _has_outlet is True but no outlet facets found.  "
                    "Set closed_cavity=True if the domain has no outlet, or "
                    f"check outlet_marker.  Available marker values (global): "
                    f"{sorted(set(int(v) for v in self.markers.values))}."
                )
            outlet_dofs_q = locate_dofs_topological(Q, fdim, outlet_facets)
            bc_pressure = dirichletbc(PETSc.ScalarType(0.0), outlet_dofs_q, Q)
            bcs_pres = [bc_pressure]
            _pressure_nullspace = None
        else:
            # Closed cavity — no outlet.  Use a PETSc null-space
            # (constant pressure mode) instead of a bogus Dirichlet pin.
            bcs_pres = []
            _pressure_nullspace = PETSc.NullSpace().create(
                constant=True, comm=mesh.comm
            )

        hmin_local = mesh.hmin()
        hmin = float(mesh.comm.allreduce(hmin_local, op=MPI.MIN))
        if use_weak_slip:
            # Heuristic normal-velocity penalty for weak no-penetration.
            # Scale with both diffusion and pseudo-time inertia.
            gamma_val = max(20.0 * nu_eff / max(hmin, 1e-12), 5.0 / max(dt_val, 1e-12))
            weak_slip_gamma = Constant(mesh, PETSc.ScalarType(gamma_val))
            if rank0:
                info(
                    f"UrbanWind: weak slip penalty gamma={gamma_val:.4g} "
                    f"(hmin={hmin:.4g}, dt={dt_val:.4g})"
                )

        if rank0:
            info("UrbanWind: setup phase 2/5 - building variational forms")

        # ---- IPCS variational forms ----

        # Convecting velocity
        if params.convection_linearization == "ab2":
            # AB2 extrapolated: u_conv = 1.5*u_n - 0.5*u_{n-1}
            u_conv = 1.5 * u_n - 0.5 * u_nm1
        else:
            # Picard (lagged): u_conv = u_n  — robust for steady state
            u_conv = u_n

        # ---- Step 1: Tentative velocity ----
        # (u* - u_n)/dt + (u_AB · ∇)u* - ν∇²u* + ∇p_n = 0
        # Bilinear in u_trial, v:
        F1_lhs = (1.0 / dt_c) * inner(u_trial, v) * dx + nu_c * inner(
            grad(u_trial), grad(v)
        ) * dx
        if params.grad_div_gamma > 0.0:
            gamma_gd = Constant(mesh, PETSc.ScalarType(params.grad_div_gamma))
            F1_lhs += gamma_gd * div(u_trial) * div(v) * dx

        # Convection (u_conv · ∇)u — semi-implicit in u_trial
        if params.convective_form == "skew_symmetric":
            # Skew-symmetric form: 0.5[(u_conv·∇)u - (u_conv·∇v)^T u]
            # Conserves kinetic energy exactly (b(u,u)=0) but prevents
            # convective energy removal at outlets.  Only suitable when
            # sufficient diffusion (nu_t) is present to drain energy.
            F1_lhs += (
                0.5 * inner(dot(grad(u_trial), u_conv), v) * dx
                - 0.5 * inner(dot(grad(v), u_conv), u_trial) * dx
            )
        else:
            # Standard Galerkin convection — (u_conv·∇)u · v.
            # Naturally transports energy through open boundaries.
            F1_lhs += inner(dot(grad(u_trial), u_conv), v) * dx

        # Backflow stabilization on open boundaries:
        # if u_conv·n < 0 (inflow through an outflow/open boundary),
        # add damping proportional to |u_conv·n|.
        if params.backflow_beta > 0.0:
            beta_bf = Constant(mesh, PETSc.ScalarType(params.backflow_beta))
            un = dot(u_conv, n_vec)
            un_neg = 0.5 * (abs(un) - un)  # max(-un, 0)
            open_tags: List[int] = []
            if side_top_open:
                open_tags.extend([int(BndCat.SIDE), int(BndCat.TOP)])
            if self._has_outlet:
                open_tags.append(int(BndCat.OUTLET))
            for tag in open_tags:
                F1_lhs += beta_bf * un_neg * inner(u_trial, v) * ds_cat(tag)

        F1_rhs = (1.0 / dt_c) * inner(u_n, v) * dx - inner(grad(p_n), v) * dx

        # Wall friction weak terms (only for friction model)
        if params.wall_model == "friction":
            beta_c = Constant(mesh, PETSc.ScalarType(params.beta_wall))
            # Auto gamma_normal
            h_min = mesh.hmin()
            gamma_val = (
                params.gamma_normal
                if params.gamma_normal is not None
                else 50.0 * nu_eff / h_min
            )
            gamma_c = Constant(mesh, PETSc.ScalarType(gamma_val))
            info(f"UrbanWind: friction model — β={params.beta_wall}, γ={gamma_val:.4g}")

            for tag in solid_tags:
                # tangential friction: β * (u_t · v_t)  with u_t = u - (u·n)n
                F1_lhs += (
                    beta_c
                    * (inner(u_trial, v) - inner(dot(u_trial, n_vec) * n_vec, v))
                    * ds_cat(tag)
                )
                # normal penalty: γ * (u·n)(v·n)
                F1_lhs += (
                    gamma_c * inner(dot(u_trial, n_vec), dot(v, n_vec)) * ds_cat(tag)
                )

        if weak_slip_gamma is not None:
            for tag in weak_slip_tags:
                F1_lhs += (
                    weak_slip_gamma
                    * inner(dot(u_trial, n_vec), dot(v, n_vec))
                    * ds_cat(tag)
                )

        a1 = _fem_form(F1_lhs)
        L1 = _fem_form(F1_rhs)

        # ---- Step 2: Pressure correction  Δφ = (1/dt) div(u*) ----
        a2 = _fem_form(inner(grad(p_trial), grad(q)) * dx)
        L2 = _fem_form(-(1.0 / dt_c) * div(u_) * q * dx)

        # ---- Step 3: Velocity correction  u^{n+1} = u* - dt ∇φ ----
        a3_expr = inner(u_trial, v) * dx
        L3_expr = inner(u_, v) * dx - dt_c * inner(grad(phi), v) * dx
        if weak_slip_gamma is not None:
            for tag in weak_slip_tags:
                a3_expr += (
                    weak_slip_gamma
                    * inner(dot(u_trial, n_vec), dot(v, n_vec))
                    * ds_cat(tag)
                )
                L3_expr += (
                    weak_slip_gamma
                    * inner(dot(u_, n_vec), dot(v, n_vec))
                    * ds_cat(tag)
                )
        a3 = _fem_form(a3_expr)
        L3 = _fem_form(L3_expr)

        # Step 3 must enforce velocity BCs so corrected velocity
        # satisfies inlet profile and no-slip exactly.

        # ---- Assemble LHS matrices ----
        # A1 depends on u_n via convection — reassembled each step
        if rank0:
            info("UrbanWind: setup phase 3/5 - assembling initial matrices")
        A1 = _fem_petsc.create_matrix(a1)
        A1.zeroEntries()
        _fem_petsc.assemble_matrix_mat(A1, a1, bcs=bcs_vel)
        A1.assemble()
        # A2, A3 are truly constant
        A2 = _fem_petsc.assemble_matrix(a2, bcs=bcs_pres)
        A2.assemble()
        if _pressure_nullspace is not None:
            A2.setNullSpace(_pressure_nullspace)
        A3 = _fem_petsc.assemble_matrix(a3, bcs=bcs_vel)
        A3.assemble()

        # ---- KSP solvers ----
        if rank0:
            info("UrbanWind: setup phase 4/5 - configuring linear solvers")
        ksp1 = self._make_ksp(
            A1, params.petsc_velocity or DEFAULT_VELOCITY_PETSC, "vel"
        )
        ksp2 = self._make_ksp(
            A2, params.petsc_pressure or DEFAULT_PRESSURE_PETSC, "pres"
        )
        ksp3 = self._make_ksp(
            A3,
            {
                "ksp_type": "cg",
                "ksp_rtol": 1e-8,
                "ksp_monitor": None,
                "pc_type": "jacobi",
            },
            "corr",
        )

        # ---- Diagnostics forms ----
        if rank0:
            info("UrbanWind: setup phase 5/5 - preparing diagnostics and loop")
        one_c = Constant(mesh, PETSc.ScalarType(1.0))
        vol_form = _fem_form(one_c * dx)
        div_norm_form = _fem_form(div(u_n) * div(u_n) * dx)
        kinetic_form = _fem_form(0.5 * inner(u_n, u_n) * dx)
        q_in_form = _fem_form(dot(u_n, n_vec) * ds_cat(int(BndCat.INLET)))
        q_open_form = _fem_form(dot(u_n, n_vec) * ds_cat(int(BndCat.INLET)))
        slip_un2_form = None
        slip_area = 0.0
        if self._has_outlet:
            q_out_form = _fem_form(dot(u_n, n_vec) * ds_cat(int(BndCat.OUTLET)))
            q_open_measure = ds_cat(int(BndCat.INLET)) + ds_cat(int(BndCat.OUTLET))
            if side_top_open:
                q_open_measure += ds_cat(int(BndCat.SIDE)) + ds_cat(int(BndCat.TOP))
            q_open_form = _fem_form(
                dot(u_n, n_vec) * q_open_measure
            )
        else:
            q_out_form = None
        if side_top_slip:
            slip_measure = ds_cat(int(BndCat.SIDE)) + ds_cat(int(BndCat.TOP))
            slip_un2_form = _fem_form(dot(u_n, n_vec) * dot(u_n, n_vec) * slip_measure)
            slip_area_form = _fem_form(one_c * slip_measure)
            slip_area = float(
                mesh.comm.allreduce(
                    dolfinx.fem.assemble_scalar(slip_area_form), op=MPI.SUM
                )
            )

        volume = float(
            mesh.comm.allreduce(dolfinx.fem.assemble_scalar(vol_form), op=MPI.SUM)
        )

        # ---- Time loop ----
        converged_count = 0
        stat_ke_history: List[float] = []
        stat_flux_history: List[float] = []
        stat_div_history: List[float] = []
        stat_fluximb_history: List[float] = []
        stop_reason = "max_steps"
        rel = float("nan")
        div_rms = float("nan")
        flux_imbalance = float("nan")
        cfl = float("nan")
        ke_rel = float("nan")
        flux_rel = float("nan")
        div_curr = float("nan")
        fluximb_curr = float("nan")
        slip_un_rms = float("nan")
        if rank0:
            info(
                "UrbanWind: Starting pseudo-time loop "
                f"(mode={params.simulation_mode}) …"
            )
        t_sim = 0.0

        for step in range(1, params.max_steps + 1):
            dt_step = dt_val
            # Smooth startup for challenging large-domain cases:
            # ramp inlet forcing from 0 to full speed during first N steps.
            if ramp_time > 0.0:
                inlet_scale = min(1.0, (t_sim + dt_step) / ramp_time)

                def _inlet_expr_scaled_step(
                    x: np.ndarray, _scale=inlet_scale
                ) -> np.ndarray:
                    return _scale * inlet_expr(x)

                u_in_func.interpolate(_inlet_expr_scaled_step)

            # Reassemble A1 with updated convection velocity (u_n)
            A1.zeroEntries()
            _fem_petsc.assemble_matrix_mat(A1, a1, bcs=bcs_vel)
            A1.assemble()
            ksp1.setOperators(A1)

            # Step 1 — tentative velocity
            b1 = _fem_petsc.assemble_vector(L1)
            _fem_petsc.apply_lifting(b1, [a1], [bcs_vel])
            b1.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES,
                mode=PETSc.ScatterMode.REVERSE,
            )
            _fem_petsc.set_bc(b1, bcs_vel)
            ksp1.solve(b1, u_.x.petsc_vec)
            u_.x.scatter_forward()
            b1.destroy()
            ksp1_reason = ksp1.getConvergedReason()

            # Step 2 — pressure correction
            b2 = _fem_petsc.assemble_vector(L2)
            _fem_petsc.apply_lifting(b2, [a2], [bcs_pres])
            b2.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES,
                mode=PETSc.ScatterMode.REVERSE,
            )
            _fem_petsc.set_bc(b2, bcs_pres)
            if _pressure_nullspace is not None:
                _pressure_nullspace.remove(b2)
            ksp2.solve(b2, phi.x.petsc_vec)
            phi.x.scatter_forward()
            b2.destroy()

            # Update pressure: p^{n+1} = p^n + φ
            p_n.x.array[:] += phi.x.array
            p_n.x.scatter_forward()

            # Step 3 — velocity correction (enforce BCs)
            b3 = _fem_petsc.assemble_vector(L3)
            _fem_petsc.apply_lifting(b3, [a3], [bcs_vel])
            b3.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES,
                mode=PETSc.ScatterMode.REVERSE,
            )
            _fem_petsc.set_bc(b3, bcs_vel)
            ksp3.solve(b3, u_.x.petsc_vec)
            u_.x.scatter_forward()
            b3.destroy()

            # ---- Convergence check (owned DOFs only, no ghosts) ----
            n_owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
            diff = u_.x.array[:n_owned] - u_n.x.array[:n_owned]
            diff_norm = float(
                np.sqrt(mesh.comm.allreduce(np.dot(diff, diff), op=MPI.SUM))
            )
            u_norm = float(
                np.sqrt(
                    mesh.comm.allreduce(
                        np.dot(u_n.x.array[:n_owned], u_n.x.array[:n_owned]),
                        op=MPI.SUM,
                    )
                )
            )
            rel = diff_norm / max(max(u_norm, diff_norm), 1e-14)

            # Advance state
            omega = params.velocity_relaxation
            if ksp1_reason <= 0:
                omega = min(omega, params.spike_relaxation)
                if rank0:
                    info(
                        f"UrbanWind: velocity solve did not converge at step {step} "
                        f"(reason={ksp1_reason}); applying extra damping."
                    )
            if step >= params.spike_min_step and rel > params.spike_rel_threshold:
                omega = min(omega, params.spike_relaxation)
                if rank0:
                    info(
                        f"UrbanWind: spike damping active at step {step} "
                        f"(rel={rel:.3e}, omega={omega:.3f})"
                    )
            u_nm1.x.array[:] = u_n.x.array
            u_nm1.x.scatter_forward()
            if omega < 1.0:
                u_n.x.array[:] = omega * u_.x.array + (1.0 - omega) * u_n.x.array
            else:
                u_n.x.array[:] = u_.x.array
            u_n.x.scatter_forward()

            # ---- Diagnostics from updated state u_n ----
            div_norm_sq = float(
                mesh.comm.allreduce(
                    dolfinx.fem.assemble_scalar(div_norm_form), op=MPI.SUM
                )
            )
            div_rms = float(np.sqrt(max(div_norm_sq, 0.0) / max(volume, 1e-14)))

            kinetic = float(
                mesh.comm.allreduce(
                    dolfinx.fem.assemble_scalar(kinetic_form), op=MPI.SUM
                )
            )
            ke_mean = kinetic / max(volume, 1e-14)

            q_in = -float(
                mesh.comm.allreduce(dolfinx.fem.assemble_scalar(q_in_form), op=MPI.SUM)
            )
            q_open = float(
                mesh.comm.allreduce(dolfinx.fem.assemble_scalar(q_open_form), op=MPI.SUM)
            )
            flux_ref = abs(q_in)

            q_out = 0.0
            if q_out_form is not None:
                q_out = float(
                    mesh.comm.allreduce(
                        dolfinx.fem.assemble_scalar(q_out_form), op=MPI.SUM
                    )
                )
                flux_ref = abs(q_in) + abs(q_out)
            flux_imbalance = (
                abs(q_open) / max(flux_ref, 1e-14) if self._has_outlet else 0.0
            )
            if slip_un2_form is not None and slip_area > 0.0:
                slip_un2 = float(
                    mesh.comm.allreduce(
                        dolfinx.fem.assemble_scalar(slip_un2_form), op=MPI.SUM
                    )
                )
                slip_un_rms = float(np.sqrt(max(slip_un2, 0.0) / max(slip_area, 1e-14)))
            else:
                slip_un_rms = float("nan")

            n_owned_comp = V.dofmap.index_map.size_local
            bs = V.dofmap.index_map_bs
            u_arr = u_n.x.array[: n_owned_comp * bs].reshape(n_owned_comp, bs)
            u_mag_max_local = (
                float(np.max(np.linalg.norm(u_arr, axis=1))) if n_owned_comp > 0 else 0.0
            )
            u_mag_max = float(mesh.comm.allreduce(u_mag_max_local, op=MPI.MAX))
            cfl = (u_mag_max * dt_val) / max(hmin, 1e-14)

            # ---- Prepare convergence diagnostics for logging/stopping ----
            steady_is_converged = False
            steady_streak_next = converged_count
            stat_ready = False
            stat_is_converged = False
            stat_checks_start = params.stat_warmup_steps + 2 * params.stat_window

            if params.simulation_mode == "steady":
                steady_is_converged = (
                    rel < params.steady_tolerance
                    and div_rms < params.divergence_tolerance
                    and (
                        (not self._has_outlet)
                        or flux_imbalance < params.flux_imbalance_tolerance
                    )
                )
                if step >= params.min_steps and steady_is_converged:
                    steady_streak_next = converged_count + 1
                else:
                    steady_streak_next = 0
            else:
                stat_ke_history.append(ke_mean)
                stat_flux_history.append(q_out if self._has_outlet else q_open)
                stat_div_history.append(div_rms)
                stat_fluximb_history.append(flux_imbalance)
                window = params.stat_window
                if step >= stat_checks_start:
                    ke_prev = float(np.mean(stat_ke_history[-2 * window : -window]))
                    ke_curr = float(np.mean(stat_ke_history[-window:]))
                    flux_prev = float(
                        np.mean(stat_flux_history[-2 * window : -window])
                    )
                    flux_curr = float(np.mean(stat_flux_history[-window:]))
                    div_curr = float(np.mean(stat_div_history[-window:]))
                    fluximb_curr = float(np.mean(stat_fluximb_history[-window:]))

                    ke_rel = _relative_change(ke_curr, ke_prev)
                    flux_rel = _relative_change(flux_curr, flux_prev)
                    stat_ready = True
                    stat_is_converged = (
                        ke_rel < params.stat_tolerance
                        and flux_rel < params.stat_tolerance
                        and div_curr < params.stat_divergence_tolerance
                        and fluximb_curr < params.stat_flux_imbalance_tolerance
                    )

            if step <= params.log_initial_steps or step % params.log_every_steps == 0:
                its1 = ksp1.getIterationNumber()
                its2 = ksp2.getIterationNumber()
                its3 = ksp3.getIterationNumber()
                if rank0:
                    slip_part = (
                        f"  slip_un={slip_un_rms:.3e}" if not np.isnan(slip_un_rms) else ""
                    )
                    info(
                        f"UrbanWind: step {step:5d}  rel={rel:.3e}  "
                        f"div_rms={div_rms:.3e}  flux_imb={flux_imbalance:.3e}  "
                        f"CFL={cfl:.2f}{slip_part}  "
                        f"KSP its: vel={its1} pres={its2} corr={its3}"
                    )
                    if params.simulation_mode == "steady":
                        rel_ok = rel < params.steady_tolerance
                        div_ok = div_rms < params.divergence_tolerance
                        flux_ok = (not self._has_outlet) or (
                            flux_imbalance < params.flux_imbalance_tolerance
                        )
                        info(
                            "UrbanWind: criteria steady  "
                            f"rel={rel:.3e}/{params.steady_tolerance:.3e} "
                            f"[{'OK' if rel_ok else 'WAIT'}]  "
                            f"div={div_rms:.3e}/{params.divergence_tolerance:.3e} "
                            f"[{'OK' if div_ok else 'WAIT'}]  "
                            f"flux={flux_imbalance:.3e}/{params.flux_imbalance_tolerance:.3e} "
                            f"[{'OK' if flux_ok else 'WAIT'}]  "
                            f"streak={steady_streak_next}/{params.steady_window}  "
                            f"min_steps={step}/{params.min_steps}"
                        )
                    else:
                        if stat_ready:
                            dke_ok = ke_rel < params.stat_tolerance
                            dflux_ok = flux_rel < params.stat_tolerance
                            divw_ok = div_curr < params.stat_divergence_tolerance
                            fluxw_ok = (
                                fluximb_curr < params.stat_flux_imbalance_tolerance
                            )
                            info(
                                "UrbanWind: criteria stat    "
                                f"dKE={ke_rel:.3e}/{params.stat_tolerance:.3e} "
                                f"[{'OK' if dke_ok else 'WAIT'}]  "
                                f"dFlux={flux_rel:.3e}/{params.stat_tolerance:.3e} "
                                f"[{'OK' if dflux_ok else 'WAIT'}]"
                            )
                            info(
                                "UrbanWind: criteria stat    "
                                f"mean_div={div_curr:.3e}/{params.stat_divergence_tolerance:.3e} "
                                f"[{'OK' if divw_ok else 'WAIT'}]  "
                                f"mean_flux_imb={fluximb_curr:.3e}/{params.stat_flux_imbalance_tolerance:.3e} "
                                f"[{'OK' if fluxw_ok else 'WAIT'}]  "
                                f"window={params.stat_window} warmup={params.stat_warmup_steps}"
                            )
                        else:
                            steps_left = max(0, stat_checks_start - step)
                            info(
                                "UrbanWind: criteria stat    "
                                f"windowed checks start in {steps_left} step(s) "
                                f"(warmup={params.stat_warmup_steps}, "
                                f"window={params.stat_window})"
                            )
                    if cfl > params.cfl_target:
                        info(
                            f"UrbanWind: CFL advisory at step {step}: "
                            f"{cfl:.2f} > target {params.cfl_target:.2f}"
                        )

            # Stopping criteria
            if params.simulation_mode == "steady":
                if step >= params.min_steps and steady_is_converged:
                    converged_count += 1
                    if converged_count >= params.steady_window:
                        if rank0:
                            info(
                                f"UrbanWind: Steady state reached at step {step} "
                                f"(rel={rel:.3e}, div_rms={div_rms:.3e}, "
                                f"flux_imb={flux_imbalance:.3e})"
                            )
                        stop_reason = "steady_criteria"
                        break
                else:
                    converged_count = 0
            else:
                if stat_ready and stat_is_converged:
                    if rank0:
                        info(
                            f"UrbanWind: Statistical stationarity reached at step {step} "
                            f"(dKE={ke_rel:.3e}, dFlux={flux_rel:.3e}, "
                            f"mean_div_rms={div_curr:.3e}, "
                            f"mean_flux_imb={fluximb_curr:.3e})"
                        )
                    stop_reason = "statistical_stationarity"
                    break

            # Adaptive pseudo-time stepping (for next step)
            if params.adaptive_dt:
                dt_new = dt_val
                if cfl > params.cfl_target:
                    ratio = max(params.cfl_target / max(cfl, 1e-12), 0.05)
                    dt_new = max(params.dt_min, dt_val * params.cfl_reduce_safety * ratio)
                    if cfl > params.cfl_hard_limit:
                        dt_new = max(params.dt_min, 0.5 * dt_new)
                if ksp1_reason <= 0:
                    dt_new = max(params.dt_min, min(dt_new, 0.5 * dt_val))
                elif cfl < 0.5 * params.cfl_target:
                    dt_new = min(dt_max, dt_val * params.cfl_increase_factor)
                if dt_new != dt_val:
                    old_dt = dt_val
                    dt_val = dt_new
                    dt_c.value = PETSc.ScalarType(dt_val)
                    if rank0 and (step <= params.log_initial_steps or step % params.log_every_steps == 0):
                        info(
                            f"UrbanWind: adaptive dt update at step {step}: "
                            f"{old_dt:.3e} -> {dt_val:.3e}"
                        )
            t_sim += dt_step
        else:
            if rank0:
                info(
                    f"UrbanWind: Reached max_steps={params.max_steps} "
                    f"(final rel={rel:.3e})"
                )
            step = params.max_steps

        if rank0:
            summary = (
                f"UrbanWind: Summary  reason={stop_reason}  step={step}  "
                f"rel={rel:.3e}  div_rms={div_rms:.3e}  flux_imb={flux_imbalance:.3e}  "
                f"CFL={cfl:.2f}  dt={dt_val:.3e}  t={t_sim:.3e}"
            )
            if params.simulation_mode == "statistical_steady":
                summary += (
                    f"  dKE={ke_rel:.3e}  dFlux={flux_rel:.3e}  "
                    f"mean_div_rms_window={div_curr:.3e}  "
                    f"mean_flux_imb_window={fluximb_curr:.3e}"
                )
            info(summary)

        # ---- Cleanup KSPs ----
        ksp1.destroy()
        ksp2.destroy()
        ksp3.destroy()

        # ---- Output ----
        if output_path is not None:
            from dolfinx.io import XDMFFile

            # XDMF requires functions of the same degree as the mesh (P1).
            # Interpolate P2 velocity down to P1 for output.
            V1_out = FunctionSpace(mesh, "Lagrange", 1, dim=3)
            u_out = Function(V1_out, name="velocity")
            u_out.interpolate(u_n)

            p_out = Function(Q, name="pressure")
            p_out.x.array[:] = p_n.x.array
            p_out.x.scatter_forward()

            # Write velocity and pressure to separate XDMF files so that
            # ParaView does not mix them up when applying glyph filters.
            import os

            base, ext = os.path.splitext(output_path)
            vel_path = f"{base}_velocity{ext}"
            pres_path = f"{base}_pressure{ext}"

            with XDMFFile(mesh.comm, vel_path, "w") as xdmf:
                xdmf.write_mesh(mesh)
                xdmf.write_function(u_out)
            with XDMFFile(mesh.comm, pres_path, "w") as xdmf:
                xdmf.write_mesh(mesh)
                xdmf.write_function(p_out)
            info(f"UrbanWind: Saved solution to {vel_path} and {pres_path}")

        # ---- Convert to dtcc-core VolumeMesh ----
        if self.volume_mesh_dtcc is not None:
            if mesh.comm.size > 1:
                if rank0:
                    warning(
                        "UrbanWind: dtcc-core VolumeMesh output is not "
                        "supported in MPI mode (output mapping uses local DOFs). "
                        "Returning dolfinx fields instead."
                    )
                return u_n, p_n
            return _dolfinx_to_volume_mesh(mesh, u_n, p_n, self.volume_mesh_dtcc)

        # Fallback when mesh was provided directly (no dtcc-core mesh)
        return u_n, p_n

    # -------------------------------------------------------- PETSc helpers
    @staticmethod
    def _make_ksp(
        A: PETSc.Mat,
        opts: Dict[str, Any],
        prefix: str,
    ) -> PETSc.KSP:
        ksp = PETSc.KSP().create(A.getComm())
        ksp.setOperators(A)
        ksp.setOptionsPrefix(f"urban_wind_{prefix}_")
        popts = PETSc.Options()
        for k, v in opts.items():
            if v is None:
                continue
            full_key = f"urban_wind_{prefix}_{k}"
            popts[full_key] = str(v)
        ksp.setFromOptions()
        return ksp


__all__ = [
    "BndCat",
    "UrbanWindParameters",
    "UrbanWindSimulator",
    "select_inlet_outlet",
    "BBOX_MARKER_NORMALS",
]
