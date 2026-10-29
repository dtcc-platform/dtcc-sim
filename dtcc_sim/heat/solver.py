"""
Heat equation solver

A steady-state diffusion model for air temperature in a 3D city volume.
Supports multiple boundary condition types (Dirichlet, Neumann, Robin)
for different urban surfaces (walls, roofs, ground, open boundaries).
"""

from __future__ import annotations

from ..timing import phase as timing_phase, timed, timed_run
from ..meshing import prepare_city_mesh, publish_mesh_provenance
from ..output import _output_directory, _solution_path, _export_city_surface
from ..coordinates import input_crs, local_mesh_coordinates, write_coordinate_frame
from .parameters import (
    BndCat, BCSpec, HeatParameters, DirichletBCSpec, NeumannBCSpec, RobinBCSpec,
)

from typing import Any, Mapping, Optional
from pathlib import Path

import numpy as np
import dolfinx

from dtcc_sim.fenics import *  # FEniCSx wrapper
from dtcc_sim.fenics import _mesh_from_dtcc
from dtcc_sim.smooth_reconstruction import _attach_scalar_field_to_volume_mesh


DEFAULT_PETSC_OPTIONS: Mapping[str, Any] = {
    "ksp_monitor_short": None,
    "ksp_converged_reason": None,
    "ksp_type": "cg",
    "ksp_rtol": 1.0e-6,
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
}


# -----------------------------------------------------------------------------
# Helpers: global marker stats + collapsing to 4 categories
# -----------------------------------------------------------------------------


def _global_max_positive_marker(
    mesh: dolfinx.mesh.Mesh, markers: dolfinx.mesh.MeshTags
) -> int:
    """Find the global maximum positive marker value across all processes."""
    vals = np.asarray(markers.values, dtype=np.int32)
    if np.any(vals >= 0):
        local_max = int(vals[vals >= 0].max())
    else:
        local_max = -1
    return int(mesh.comm.allreduce(local_max, op=MPI.MAX))


def infer_num_buildings(mesh: dolfinx.mesh.Mesh, markers: dolfinx.mesh.MeshTags) -> int:
    """
    Infer the number of buildings from marker scheme.

    Assumes marker scheme:
      walls: 0..N-1
      roofs: N..2N-1
      other boundaries: negative tags

    Args:
        mesh: The mesh
        markers: Facet markers

    Returns:
        Number of buildings
    """
    max_pos = _global_max_positive_marker(mesh, markers)
    if max_pos < 0:
        return 0
    return (max_pos + 1) // 2


def collapse_boundary_markers(
    mesh: dolfinx.mesh.Mesh,
    markers: dolfinx.mesh.MeshTags,
    num_buildings: int,
    ground_tag: int = -1,
) -> dolfinx.mesh.MeshTags:
    """
    Create collapsed facet MeshTags with 4 categories: WALL/ROOF/GROUND/OPEN.

    This reduces O(num_buildings) subdomain terms in UFL forms to O(1).

    Args:
        mesh: The mesh
        markers: Original facet markers
        num_buildings: Number of buildings
        ground_tag: Marker value for ground facets (default: -1)

    Returns:
        New MeshTags with collapsed categories
    """
    fdim = mesh.topology.dim - 1

    idx = np.asarray(markers.indices, dtype=np.int32)
    vals = np.asarray(markers.values, dtype=np.int32)

    N = int(num_buildings)
    cat = np.empty_like(vals, dtype=np.int32)

    # Categorize facets
    mask_wall = (vals >= 0) & (vals < N)
    mask_roof = (vals >= N) & (vals < 2 * N)
    mask_ground = vals == ground_tag
    mask_open = vals < ground_tag  # typically -2,-3,... (bbox sides/top)

    cat[mask_wall] = int(BndCat.WALL)
    cat[mask_roof] = int(BndCat.ROOF)
    cat[mask_ground] = int(BndCat.GROUND)
    cat[mask_open] = int(BndCat.OPEN)

    # Anything unexpected: treat as OPEN (safe default)
    mask_other = ~(mask_wall | mask_roof | mask_ground | mask_open)
    cat[mask_other] = int(BndCat.OPEN)

    # meshtags expects sorted indices for robust .find()
    order = np.argsort(idx)
    return dolfinx.mesh.meshtags(mesh, fdim, idx[order], cat[order])


# -----------------------------------------------------------------------------
# Simulator
# -----------------------------------------------------------------------------


class HeatSolver:
    """
    Heat equation solver: Steady-state thermal analysis for 3D urban environments.

    Physical Model
    --------------
    This simulator solves the steady-state heat diffusion equation with optional
    relaxation term to model air temperature distribution in urban volumes:

        -∇·(κ∇T) + σ(T - T_ambient) = 0    in Ω (urban domain)

    where:
        - T: temperature field [°C or K]
        - κ: thermal diffusivity [m²/s] - models heat conduction/diffusion in air
        - σ: relaxation coefficient [1/s] - models thermal exchange with ambient air mass
        - T_ambient: ambient/background temperature [°C or K]

    The diffusion term (-∇·(κ∇T)) represents heat conduction through the air, while
    the relaxation term (σ(T - T_ambient)) models heat exchange with the surrounding
    ambient air mass (e.g., through large-scale atmospheric mixing, advection effects
    simplified as a sink/source term).

    Boundary Conditions
    -------------------
    Three types of boundary conditions are supported on urban surfaces:

    1. **Dirichlet BC**: Fixed temperature
       T = T_prescribed    on ∂Ω

    2. **Neumann BC**: Prescribed heat flux
       -κ ∂T/∂n = q    on ∂Ω

    3. **Robin BC**: Convective heat transfer (Newton's law of cooling)
       -κ ∂T/∂n = h(T - T_ref)    on ∂Ω

       where h is the heat transfer coefficient [W/(m²·K)] and T_ref is the
       reference temperature (e.g., building surface temperature).

    Urban surfaces are categorized as:
        - **Walls**: Building vertical surfaces
        - **Roofs**: Building horizontal surfaces (top)
        - **Ground**: Terrain surface
        - **Open**: Far-field/domain boundaries

    Applications
    ------------
    - **Urban Heat Island (UHI) analysis**: Quantify temperature elevation in cities
    - **Building energy assessment**: Evaluate heat losses and thermal comfort
    - **Outdoor thermal comfort**: Study pedestrian-level temperature distribution
    - **Urban climate adaptation**: Inform urban planning and design decisions
    - **Mitigation strategies**: Evaluate green infrastructure, cool roofs, ventilation

    Physical Interpretation
    -----------------------
    Buildings act as heat sources/sinks with surface temperatures typically different
    from ambient air. The Robin boundary condition models convective heat exchange
    between building surfaces and surrounding air. The steady-state solution represents
    the equilibrium temperature distribution under given thermal conditions.

    Typical parameter ranges:
        - κ ~ 0.5-5.0 m²/s (effective thermal diffusivity of urban air)
        - σ ~ 0.0-0.5 1/s (relaxation rate, 0 = pure diffusion)
        - h ~ 2-10 W/(m²·K) for natural convection
        - h ~ 10-50 W/(m²·K) for forced convection (wind)

    Usage Examples
    --------------
    >>> # From bounds (auto-generates mesh)
    >>> sim = HeatSolver(bounds=bounds)
    >>> T = sim.simulate()

    >>> # From pre-computed mesh
    >>> sim = HeatSolver(mesh_path="mesh.xdmf")
    >>> T = sim.simulate()

    >>> # Heat wave scenario with custom parameters
    >>> params = HeatParameters(
    ...     kappa=2.0,           # Enhanced mixing
    ...     T_ambient=20.0,      # 20°C ambient
    ...     wall_value=35.0,     # Hot building surfaces
    ...     wall_h=8.0,          # Moderate convection
    ... )
    >>> sim = HeatSolver(bounds=bounds, params=params)
    >>> T = sim.simulate(output_path="heatwave.xdmf")

    Notes
    -----
    - The simulator uses FEniCSx for finite element discretization
    - Linear (P1) or quadratic (P2) Lagrange elements supported
    - Parallel computation via MPI (inherited from FEniCSx)
    - Boundary markers are automatically collapsed from per-building to
      per-category to reduce computational cost

    See Also
    --------
    HeatParameters : Parameter specification and validation
    dtcc_sim.datasets.UrbanHeatSimulationDataset : Dataset wrapper interface
    """

    def __init__(
        self,
        *,
        bounds: Optional[Any] = None,
        mesh_path: Optional[str] = None,
        mesh: Optional[dolfinx.mesh.Mesh] = None,
        markers: Optional[dolfinx.mesh.MeshTags] = None,
        output_dir: Optional[str] = None,
        params: Optional[HeatParameters] = None,
        petsc_options: Optional[Mapping[str, Any]] = None,
        mesh_crs: Optional[str] = None,
    ) -> None:
        """Initialize the urban heat simulator.

        Args:
            bounds: DTCC Bounds object or bounds tuple [minx, miny, maxx, maxy]
            mesh_path: Path to XDMF mesh file with markers
            mesh: Pre-loaded dolfinx mesh
            markers: Pre-loaded facet markers
            output_dir: Optional directory for solution exports; repeated runs replace them.
            params: Simulation parameters
            petsc_options: PETSc solver options

        Note: Provide either bounds, mesh_path, or (mesh, markers).
        """
        self.output_dir = _output_directory(output_dir)
        self.bounds = bounds
        self.mesh_path = mesh_path
        self.mesh = mesh
        self.mesh_crs = input_crs(mesh, generated=bounds is not None, declared=mesh_crs)
        self.markers = markers

        self.params = params if params is not None else HeatParameters()
        self.petsc_options = (
            dict(petsc_options)
            if petsc_options is not None
            else dict(DEFAULT_PETSC_OPTIONS)
        )

        self.num_buildings: Optional[int] = None
        self.category_markers: Optional[dolfinx.mesh.MeshTags] = None
        self.solution: Optional[Function] = None
        self.volume_mesh_dtcc: Optional[Any] = None
        self.diagnostics: dict[str, Any] = {}

    def _build_mesh_from_bounds(self) -> None:
        """Build volume mesh from bounds using dtcc_core.datasets.city_volume_mesh."""
        info("Building a volume mesh for heat diffusion...")

        # Build volume mesh using the dataset
        volume_mesh, self.mesh_record = prepare_city_mesh(self.bounds, self.params)
        self.volume_mesh_dtcc = volume_mesh

        # Save to temporary file and load with FEniCSx
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".xdmf", delete=False) as tmp:
            tmp_path = tmp.name

        info(f"Saving the generated heat mesh to {tmp_path}")
        volume_mesh.save(tmp_path)

        # Load with FEniCSx
        self.mesh, self.markers = load_mesh_with_markers(tmp_path)

        # Clean up temporary file
        Path(tmp_path).unlink(missing_ok=True)
        h5_path = Path(tmp_path).with_suffix(".h5")
        h5_path.unlink(missing_ok=True)

    @timed("mesh_preparation")
    def _load_if_needed(self) -> None:
        """Load or build mesh if not already available."""
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
            info(f"Loading the heat mesh from {self.mesh_path}")
            self.mesh, self.markers = load_mesh_with_markers(self.mesh_path)
        else:
            raise ValueError(
                "HeatSolver: provide bounds=..., mesh_path=..., or (mesh=..., markers=...)."
            )

    def _apply_bc_spec(
        self,
        V: dolfinx.fem.FunctionSpace,
        u: Any,
        v: Any,
        ds_cat: Any,
        cat_markers: dolfinx.mesh.MeshTags,
        spec: BCSpec,
        category: BndCat,
        a: Any,
        L: Any,
        bcs: list,
    ) -> tuple[Any, Any, list]:
        """Apply a boundary condition specification to the variational forms.

        Args:
            V: Function space
            u: Trial function
            v: Test function
            ds_cat: Measure over boundary categories
            cat_markers: Collapsed category markers
            spec: Boundary condition specification
            category: Boundary category
            a: Bilinear form
            L: Linear form
            bcs: List of Dirichlet BCs

        Returns:
            Updated (a, L, bcs) tuple
        """
        tag = int(category)

        if isinstance(spec, DirichletBCSpec):
            bcs.append(
                DirichletBC(V, spec.value, markers=cat_markers, marker_value=tag)
            )
            return a, L, bcs

        if isinstance(spec, NeumannBCSpec):
            q = Constant(self.mesh, float(spec.flux))
            L = L + q * v * ds_cat(tag)
            return a, L, bcs

        if isinstance(spec, RobinBCSpec):
            h = Constant(self.mesh, float(spec.h))
            Tref = Constant(self.mesh, float(spec.value))
            a = a + h * u * v * ds_cat(tag)
            L = L + h * Tref * v * ds_cat(tag)
            return a, L, bcs

        raise TypeError(f"Unsupported BC spec type: {type(spec)}")

    @timed_run
    def simulate(self, *, output_path: Optional[str] = None) -> Any:
        """Run the urban heat simulation.

        Args:
            output_path: Optional path to save solution (XDMF format)

        Returns:
            A DTCC ``VolumeMesh`` with a temperature ``Field`` when the
            simulator builds the mesh from bounds, otherwise the temperature
            field as a dolfinx Function.
        """
        output_path = _solution_path(
            self.output_dir, output_path, "solution.xdmf", input_mesh=self.mesh_path,
        )
        self._load_if_needed()
        assert self.mesh is not None and self.markers is not None

        crs = input_crs(self.volume_mesh_dtcc, generated=self.bounds is not None, declared=self.mesh_crs)
        with local_mesh_coordinates(self.mesh, crs=crs) as frame:
            self.coordinate_frame = frame
            self._solve_local_heat()
            self.diagnostics["coordinate_frame"] = frame.as_dict()
            result = self._attach_temperature_field()
        with timing_phase("export"):
            if output_path is not None:
                info(f"Saving temperature to {output_path}")
                self.solution.save(output_path)
                write_coordinate_frame(output_path, self.coordinate_frame, comm=self.mesh.comm)
                _export_city_surface(
                    output_path, native_mesh=self.volume_mesh_dtcc, mesh=self.mesh,
                    markers=getattr(self, "markers", None), mesh_path=self.mesh_path,
                    frame=self.coordinate_frame,
                )
            publish_mesh_provenance(self, result, output_path)
        return result

    @timed("solver")
    def _solve_local_heat(self):
        """Assemble, solve and measure temperature in local coordinates."""
        # Infer buildings + collapse tags
        self.num_buildings = infer_num_buildings(self.mesh, self.markers)
        info(f"Detected heat-domain buildings: {self.num_buildings}")

        self.category_markers = collapse_boundary_markers(
            self.mesh, self.markers, self.num_buildings
        )

        # Function space
        V = FunctionSpace(self.mesh, "Lagrange", self.params.degree)
        info(
            f"The temperature space has {V.dofmap.index_map.size_global} degrees of freedom"
        )

        # Measures over collapsed boundary categories
        ds_cat = Measure("ds", domain=self.mesh, subdomain_data=self.category_markers)

        # Unknown/test functions
        u = TrialFunction(V)
        v = TestFunction(V)

        # Coefficients
        kappa = Constant(self.mesh, float(self.params.kappa))
        sigma = Constant(self.mesh, float(self.params.sigma))
        Tamb = Constant(self.mesh, float(self.params.T_ambient))

        # Base bilinear/linear forms
        a = kappa * inner(grad(u), grad(v)) * dx + sigma * inner(u, v) * dx
        L = sigma * inner(Tamb, v) * dx

        # Apply boundary conditions for each surface type
        bcs: list = []
        for surface_name, category in [
            ("wall", BndCat.WALL),
            ("roof", BndCat.ROOF),
            ("ground", BndCat.GROUND),
            ("open", BndCat.OPEN),
        ]:
            spec = self.params.get_bc_spec(surface_name)
            a, L, bcs = self._apply_bc_spec(
                V, u, v, ds_cat, self.category_markers, spec, category, a, L, bcs
            )

        # Solve
        info("Solving the stationary heat equation...")
        self.solution = solve(a == L, bcs=bcs, petsc_options=self.petsc_options)
        info("The heat solve is complete")
        self._record_diagnostics(V)

    @timed("validation")
    def _record_diagnostics(self, V: dolfinx.fem.FunctionSpace) -> None:
        """Record finite, local solver diagnostics available after a solve."""
        if self.mesh is None or self.solution is None:
            raise RuntimeError("UrbanHeat diagnostics require mesh and solution.")

        values = np.real(np.asarray(self.solution.x.array))
        local_count = int(values.size)
        if local_count:
            local_min = float(np.min(values))
            local_max = float(np.max(values))
            local_sum = float(np.sum(values))
            local_l2_sq = float(np.dot(values, values))
        else:
            local_min = float("inf")
            local_max = float("-inf")
            local_sum = 0.0
            local_l2_sq = 0.0

        comm = self.mesh.comm
        global_count = int(comm.allreduce(local_count, op=MPI.SUM))
        global_min = float(comm.allreduce(local_min, op=MPI.MIN))
        global_max = float(comm.allreduce(local_max, op=MPI.MAX))
        global_sum = float(comm.allreduce(local_sum, op=MPI.SUM))
        global_l2_sq = float(comm.allreduce(local_l2_sq, op=MPI.SUM))

        category_counts: dict[str, int] = {}
        if self.category_markers is not None:
            marker_values = np.asarray(self.category_markers.values, dtype=np.int32)
            for category in BndCat:
                local_category_count = int(
                    np.count_nonzero(marker_values == int(category))
                )
                category_counts[category.name.lower()] = int(
                    comm.allreduce(local_category_count, op=MPI.SUM)
                )

        self.diagnostics = {
            "degrees_of_freedom": int(V.dofmap.index_map.size_global),
            "num_buildings": int(self.num_buildings or 0),
            "boundary_category_counts": category_counts,
            "temperature_min": global_min if global_count else None,
            "temperature_max": global_max if global_count else None,
            "temperature_mean": (
                global_sum / global_count if global_count else None
            ),
            "temperature_l2_norm": float(np.sqrt(global_l2_sq)),
            "solver_convergence": "not_exposed_by_dtcc_sim_fenics_solve_wrapper",
            "residual_l2_norm": None,
            "residual_status": "not_available_from_dtcc_sim_fenics_solve_wrapper",
            "petsc_options": dict(self.petsc_options),
        }

    @timed("postprocessing")
    def _attach_temperature_field(self) -> Any:
        if self.mesh is None or self.solution is None:
            raise RuntimeError(
                "HeatSolver cannot attach a temperature field before "
                "a mesh and solution are available."
            )
        if self.volume_mesh_dtcc is None:
            return self.solution

        result = _attach_scalar_field_to_volume_mesh(
            self.mesh,
            self.solution,
            self.volume_mesh_dtcc,
            name="temperature",
            unit="degC",
        )
        setattr(result, "simulation_diagnostics", dict(self.diagnostics))
        return result
