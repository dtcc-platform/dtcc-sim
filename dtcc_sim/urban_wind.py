"""
Urban Wind CFD Solver (IPCS / ABCN Fractional-Step Method)

Solves the incompressible Navier–Stokes equations in pseudo-time using an
Incremental Pressure-Correction Scheme (IPCS) with Adams–Bashforth /
Crank–Nicolson (ABCN) time discretization on DTCC city air-volume meshes.

Physical model
--------------
    ∂u/∂t + (u·∇)u = −∇p/ρ + ν_eff Δu      (momentum)
    ∇·u = 0                                    (continuity)

The solver marches in pseudo-time until a steady state is reached or the
maximum number of steps is exhausted.

Output
------
A ``dtcc_core.model.VolumeMesh`` with attached ``Field`` objects:
- ``velocity`` (dim=3, m/s)
- ``pressure`` (dim=1, Pa)
- ``speed``    (dim=1, m/s)

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
    SIDE = 7   # lateral bbox faces that are neither inlet nor outlet


# Marker → bbox face mapping (dtcc-core volumemesh convention)
BBOX_MARKER_NORMALS: Dict[int, Tuple[float, float, float]] = {
    -3: (-1.0, 0.0, 0.0),   # xmin face
    -4: (1.0, 0.0, 0.0),    # xmax face
    -5: (0.0, -1.0, 0.0),   # ymin face
    -6: (0.0, 1.0, 0.0),    # ymax face
    -2: (0.0, 0.0, 1.0),    # top face
}


# ---------------------------------------------------------------------------
# Default PETSc options
# ---------------------------------------------------------------------------

DEFAULT_VELOCITY_PETSC: Dict[str, Any] = {
    "ksp_type": "gmres",
    "ksp_rtol": 1.0e-6,
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
}

DEFAULT_PRESSURE_PETSC: Dict[str, Any] = {
    "ksp_type": "cg",
    "ksp_rtol": 1.0e-6,
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
    steady_tolerance: float = Field(
        1.0e-4, description="Relative velocity change for steady-state"
    )
    min_steps: int = Field(50, description="Minimum steps before early stopping", ge=0)
    steady_window: int = Field(
        5, description="Consecutive converged steps before stopping", ge=1
    )

    # ---- 2.5 Solver scheme ----
    scheme: Literal["IPCS_ABCN"] = Field("IPCS_ABCN", description="Solver scheme")
    convective_form: Literal["standard", "skew_symmetric"] = Field(
        "skew_symmetric", description="Convection treatment"
    )
    pressure_correction_xi: float = Field(
        0.0, description="Open-boundary locking parameter ξ"
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
        None, description="Force outlet boundary marker (skip auto detection)"
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


def infer_num_buildings(
    mesh: dolfinx.mesh.Mesh, markers: dolfinx.mesh.MeshTags
) -> int:
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
    outlet_marker: int,
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
    mask_outlet = vals == outlet_marker

    # Remaining bbox faces (not inlet, outlet or top)
    mask_side = (
        (vals < ground_tag)
        & ~mask_top
        & ~mask_inlet
        & ~mask_outlet
    )

    cat[mask_wall] = int(BndCat.WALL)
    cat[mask_roof] = int(BndCat.ROOF)
    cat[mask_ground] = int(BndCat.GROUND)
    cat[mask_inlet] = int(BndCat.INLET)
    cat[mask_outlet] = int(BndCat.OUTLET)
    cat[mask_top] = int(BndCat.TOP)
    cat[mask_side] = int(BndCat.SIDE)

    # catch-all
    mask_other = ~(
        mask_wall | mask_roof | mask_ground | mask_top
        | mask_inlet | mask_outlet | mask_side
    )
    cat[mask_other] = int(BndCat.SIDE)

    order = np.argsort(idx)
    return dolfinx.mesh.meshtags(mesh, fdim, idx[order], cat[order])


# ---------------------------------------------------------------------------
# Helpers — inlet / outlet selection
# ---------------------------------------------------------------------------

def select_inlet_outlet(
    params: UrbanWindParameters,
) -> Tuple[int, int]:
    """Choose inlet and outlet bbox markers from wind direction.

    Uses the known marker→normal mapping in :data:`BBOX_MARKER_NORMALS`.
    The top face (-2) is excluded.

    Returns
    -------
    (inlet_marker, outlet_marker)
    """
    if params.inlet_marker is not None and params.outlet_marker is not None:
        return params.inlet_marker, params.outlet_marker

    wx, wy = params.wind_vector_xy
    w = np.array([wx, wy])

    # Only consider lateral faces (exclude top = -2)
    best_inlet_marker = -3
    best_outlet_marker = -3
    best_inlet_dot = 0.0   # most negative dot → inlet
    best_outlet_dot = 0.0  # most positive dot → outlet

    first = True
    for marker, (nx, ny, nz) in BBOX_MARKER_NORMALS.items():
        if marker == -2:  # skip top
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

    inlet = params.inlet_marker if params.inlet_marker is not None else best_inlet_marker
    outlet = params.outlet_marker if params.outlet_marker is not None else best_outlet_marker

    return inlet, outlet


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
        DtccField(name="velocity", dim=3, values=u_mapped, unit="m/s",
                  description="Velocity field from urban wind CFD solver"),
        DtccField(name="pressure", dim=1, values=p_mapped, unit="Pa",
                  description="Pressure field from urban wind CFD solver"),
        DtccField(name="speed", dim=1, values=speed, unit="m/s",
                  description="Wind speed |u| from urban wind CFD solver"),
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
    ) -> None:
        self.bounds = bounds
        self.mesh_path = mesh_path
        self.mesh = mesh
        self.markers = markers
        self.params = params if params is not None else UrbanWindParameters()

        # populated during simulation
        self.volume_mesh_dtcc: Optional[Any] = None
        self.num_buildings: int = 0
        self.category_markers: Optional[dolfinx.mesh.MeshTags] = None

    # ------------------------------------------------------------------ mesh
    def _build_mesh_from_bounds(self) -> None:
        info("UrbanWind: Building volume mesh from bounds …")
        try:
            import dtcc_core.datasets as datasets
        except ImportError:
            raise ImportError(
                "dtcc_core is required to build mesh from bounds."
            )

        volume_mesh = datasets.city_volume_mesh(
            bounds=self.bounds,
            max_mesh_size=self.params.mesh_max_mesh_size,
            domain_height=self.params.mesh_domain_height,
            raster_cell_size=self.params.mesh_raster_cell_size,
            raster_radius=self.params.mesh_raster_radius,
        )
        self.volume_mesh_dtcc = volume_mesh

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".xdmf", delete=False) as tmp:
            tmp_path = tmp.name
        info(f"UrbanWind: saving mesh to temporary file: {tmp_path}")
        volume_mesh.save(tmp_path)
        self.mesh, self.markers = load_mesh_with_markers(tmp_path)

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
    def _maybe_fetch_weather(self) -> None:
        """Optionally override wind_speed / wind_dir_deg from SMHI weather."""
        if not self.params.use_weather:
            return
        try:
            import dtcc_core.datasets as datasets
        except ImportError:
            warning("UrbanWind: dtcc_core not available; skipping weather fetch.")
            return

        info("UrbanWind: Fetching weather data …")
        try:
            sensors = datasets.weather(
                bounds=self.bounds,
                parameters=["wind_speed", "wind_direction"],
            )
            ws_vals = []
            wd_vals = []
            for s in sensors.stations():
                for ts in s.timeseries:
                    if ts.name == "wind_speed" and len(ts.values) > 0:
                        ws_vals.append(ts.values[-1])
                    elif ts.name == "wind_direction" and len(ts.values) > 0:
                        wd_vals.append(ts.values[-1])

            if ws_vals and wd_vals:
                agg = self.params.weather_aggregation
                if agg == "nearest":
                    ws = ws_vals[0]
                    wd = wd_vals[0]
                elif agg == "mean":
                    ws = float(np.mean(ws_vals))
                    wd = float(np.mean(wd_vals))
                elif agg == "median":
                    ws = float(np.median(ws_vals))
                    wd = float(np.median(wd_vals))
                else:
                    ws, wd = ws_vals[0], wd_vals[0]

                info(f"UrbanWind: weather → speed={ws:.1f} m/s, dir={wd:.0f}°")
                # Mutating Pydantic model — create a fresh copy
                self.params = self.params.model_copy(
                    update={"wind_speed": ws, "wind_dir_deg": wd}
                )
            else:
                warning("UrbanWind: weather data incomplete; using manual wind params.")
        except Exception as exc:
            warning(f"UrbanWind: weather fetch failed ({exc}); using manual params.")

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

        self._maybe_fetch_weather()

        params = self.params
        mesh = self.mesh
        dt_val = params.dt
        nu_eff = params.nu_eff

        # ---- boundary setup ----
        self.num_buildings = infer_num_buildings(mesh, self.markers)
        info(f"UrbanWind: inferred {self.num_buildings} buildings")

        inlet_marker, outlet_marker = select_inlet_outlet(params)
        info(f"UrbanWind: inlet marker = {inlet_marker}, outlet marker = {outlet_marker}")

        self.category_markers = categorize_boundary(
            mesh, self.markers, self.num_buildings,
            inlet_marker=inlet_marker, outlet_marker=outlet_marker,
        )

        # ---- function spaces (Taylor–Hood) ----
        V = FunctionSpace(mesh, "Lagrange", params.velocity_degree, dim=3)
        Q = FunctionSpace(mesh, "Lagrange", params.pressure_degree)
        info(
            f"UrbanWind: V dofs = {V.dofmap.index_map.size_global}, "
            f"Q dofs = {Q.dofmap.index_map.size_global}"
        )

        # ---- functions / state ----
        u_n = Function(V, name="u_n")    # velocity at time n
        u_nm1 = Function(V, name="u_nm1")  # velocity at time n-1
        p_n = Function(Q, name="p_n")    # pressure at time n
        u_ = Function(V, name="u_star")  # tentative velocity
        phi = Function(Q, name="phi")    # pressure correction

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
        inlet_expr = make_inlet_velocity_expression(params)
        u_in_func.interpolate(inlet_expr)

        # Locate inlet dofs
        fdim = mesh.topology.dim - 1
        inlet_facets = self.category_markers.find(int(BndCat.INLET))
        inlet_dofs = locate_dofs_topological(V, fdim, inlet_facets)
        bc_inlet = dirichletbc(u_in_func, inlet_dofs)

        # ---- wall BCs ----
        bcs_vel: List[dolfinx.fem.DirichletBC] = [bc_inlet]

        solid_tags = [int(BndCat.WALL), int(BndCat.ROOF), int(BndCat.GROUND)]
        if params.wall_model == "noslip":
            # Strong Dirichlet u=0 on walls+roof+ground
            u_zero = Function(V)  # default zero
            for tag in solid_tags:
                facets = self.category_markers.find(tag)
                dofs = locate_dofs_topological(V, fdim, facets)
                bcs_vel.append(dirichletbc(u_zero, dofs))

        # Pressure BC: phi=0 at outlet
        outlet_facets = self.category_markers.find(int(BndCat.OUTLET))
        outlet_dofs_q = locate_dofs_topological(Q, fdim, outlet_facets)
        bc_pressure = dirichletbc(
            PETSc.ScalarType(0.0), outlet_dofs_q, Q
        )
        bcs_pres = [bc_pressure]

        # ---- IPCS variational forms ----

        # AB2 extrapolated convecting velocity (after step 0, use AB2)
        # For step 0, u_nm1 = u_n = 0 so convection is zero anyway.
        # u_AB = 1.5*u_n - 0.5*u_nm1
        u_AB = 1.5 * u_n - 0.5 * u_nm1

        # ---- Step 1: Tentative velocity ----
        # (u* - u_n)/dt + (u_AB · ∇)u* - ν∇²u* + ∇p_n = 0
        # Bilinear in u_trial, v:
        F1_lhs = (
            (1.0 / dt_c) * inner(u_trial, v) * dx
            + nu_c * inner(grad(u_trial), grad(v)) * dx
        )

        # Convection (explicit AB2 in convecting velocity, implicit in u_trial)
        if params.convective_form == "skew_symmetric":
            F1_lhs += (
                0.5 * inner(dot(grad(u_trial), u_AB), v) * dx
                + 0.5 * inner(dot(grad(v), u_AB), u_trial) * dx
            )
        else:
            F1_lhs += inner(dot(grad(u_trial), u_AB), v) * dx

        F1_rhs = (
            (1.0 / dt_c) * inner(u_n, v) * dx
            - inner(grad(p_n), v) * dx
        )

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
                F1_lhs += beta_c * (
                    inner(u_trial, v)
                    - inner(dot(u_trial, n_vec) * n_vec, v)
                ) * ds_cat(tag)
                # normal penalty: γ * (u·n)(v·n)
                F1_lhs += gamma_c * inner(
                    dot(u_trial, n_vec), dot(v, n_vec)
                ) * ds_cat(tag)

        a1 = _fem_form(F1_lhs)
        L1 = _fem_form(F1_rhs)

        # ---- Step 2: Pressure correction  Δφ = (1/dt) div(u*) ----
        a2 = _fem_form(inner(grad(p_trial), grad(q)) * dx)
        L2 = _fem_form(-(1.0 / dt_c) * div(u_) * q * dx)

        # ---- Step 3: Velocity correction  u^{n+1} = u* - dt ∇φ ----
        a3 = _fem_form(inner(u_trial, v) * dx)
        L3 = _fem_form(inner(u_, v) * dx - dt_c * inner(grad(phi), v) * dx)

        # ---- Assemble LHS matrices (constant across steps) ----
        A1 = _fem_petsc.assemble_matrix(a1, bcs=bcs_vel)
        A1.assemble()
        A2 = _fem_petsc.assemble_matrix(a2, bcs=bcs_pres)
        A2.assemble()
        A3 = _fem_petsc.assemble_matrix(a3)
        A3.assemble()

        # ---- KSP solvers ----
        ksp1 = self._make_ksp(
            A1, params.petsc_velocity or DEFAULT_VELOCITY_PETSC, "vel"
        )
        ksp2 = self._make_ksp(
            A2, params.petsc_pressure or DEFAULT_PRESSURE_PETSC, "pres"
        )
        ksp3 = self._make_ksp(
            A3, {"ksp_type": "cg", "ksp_rtol": 1e-8, "pc_type": "jacobi"}, "corr"
        )

        # ---- Time loop ----
        converged_count = 0
        info("UrbanWind: Starting pseudo-time loop …")

        for step in range(1, params.max_steps + 1):
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

            # Step 2 — pressure correction
            b2 = _fem_petsc.assemble_vector(L2)
            _fem_petsc.apply_lifting(b2, [a2], [bcs_pres])
            b2.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES,
                mode=PETSc.ScatterMode.REVERSE,
            )
            _fem_petsc.set_bc(b2, bcs_pres)
            ksp2.solve(b2, phi.x.petsc_vec)
            phi.x.scatter_forward()
            b2.destroy()

            # Update pressure: p^{n+1} = p^n + φ
            p_n.x.array[:] += phi.x.array
            p_n.x.scatter_forward()

            # Step 3 — velocity correction
            b3 = _fem_petsc.assemble_vector(L3)
            b3.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES,
                mode=PETSc.ScatterMode.REVERSE,
            )
            ksp3.solve(b3, u_.x.petsc_vec)
            u_.x.scatter_forward()
            b3.destroy()

            # ---- Convergence check ----
            diff = u_.x.array - u_n.x.array
            diff_norm = float(np.sqrt(mesh.comm.allreduce(
                np.dot(diff, diff), op=MPI.SUM
            )))
            u_norm = float(np.sqrt(mesh.comm.allreduce(
                np.dot(u_n.x.array, u_n.x.array), op=MPI.SUM
            )))
            rel = diff_norm / max(u_norm, 1e-14)

            if step % 10 == 0 or step <= 5:
                info(f"UrbanWind: step {step:5d}  rel={rel:.3e}")

            # Advance state
            u_nm1.x.array[:] = u_n.x.array
            u_nm1.x.scatter_forward()
            u_n.x.array[:] = u_.x.array
            u_n.x.scatter_forward()

            # Steady-state check
            if step >= params.min_steps and rel < params.steady_tolerance:
                converged_count += 1
                if converged_count >= params.steady_window:
                    info(
                        f"UrbanWind: Steady state reached at step {step} "
                        f"(rel={rel:.3e} < tol={params.steady_tolerance})"
                    )
                    break
            else:
                converged_count = 0
        else:
            info(
                f"UrbanWind: Reached max_steps={params.max_steps} "
                f"(final rel={rel:.3e})"
            )

        # ---- Cleanup KSPs ----
        ksp1.destroy()
        ksp2.destroy()
        ksp3.destroy()

        # ---- Output ----
        if output_path is not None:
            from dolfinx.io import XDMFFile
            with XDMFFile(mesh.comm, output_path, "w") as xdmf:
                xdmf.write_mesh(mesh)
                xdmf.write_function(u_n)
                xdmf.write_function(p_n)
            info(f"UrbanWind: Saved solution to {output_path}")

        # ---- Convert to dtcc-core VolumeMesh ----
        if self.volume_mesh_dtcc is not None:
            return _dolfinx_to_volume_mesh(
                mesh, u_n, p_n, self.volume_mesh_dtcc
            )

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
            full_key = f"urban_wind_{prefix}_{k}"
            if v is None:
                popts[full_key] = ""
            else:
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
